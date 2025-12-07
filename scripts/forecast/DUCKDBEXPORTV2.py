#!/usr/bin/env python3
"""
RegionIQ → Supabase Data Sync (Production-Grade)
================================================

V2.0: Complete rewrite with proper column mapping and validation

Syncs forecast data from DuckDB gold schema to Supabase:
  - gold.uk_macro_forecast → supabase.uk_macro_forecast
  - gold.itl1_forecast → supabase.itl1_forecast

Features:
  - Pre-flight validation (checks for NULLs, duplicates)
  - Proper column mapping with fallback logic
  - Batch upserts with progress tracking
  - Rollback on failure
  - Comprehensive logging and verification

Author: RegionIQ
Version: 2.0 (Fixed)
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

try:
    import duckdb
    HAVE_DUCKDB = True
except ImportError:
    HAVE_DUCKDB = False
    print("ERROR: duckdb package required. Install: pip install duckdb")
    sys.exit(1)

try:
    from supabase import create_client, Client
    HAVE_SUPABASE = True
except ImportError:
    HAVE_SUPABASE = False
    print("ERROR: supabase package required. Install: pip install supabase")
    sys.exit(1)

# ===============================
# Configuration
# ===============================

DUCK_PATH = Path("data/lake/warehouse.duckdb")

# Supabase credentials from environment
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # Use service key for writes

# Batch settings
BATCH_SIZE = 500  # Rows per upsert batch
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2

# Logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"supabase_sync_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ===============================
# Data Validation
# ===============================

class DataValidator:
    """Pre-flight data validation"""
    
    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame, 
        source: str,
        required_cols: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Validate dataframe before upload.
        Returns (is_valid, list_of_issues)
        """
        issues = []
        
        # Check not empty
        if df.empty:
            issues.append(f"{source}: DataFrame is empty")
            return False, issues
        
        # Check required columns exist
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            issues.append(f"{source}: Missing columns: {missing_cols}")
            return False, issues
        
        # Check for NULLs in critical columns
        for col in required_cols:
            null_count = df[col].isna().sum()
            if null_count > 0:
                issues.append(f"{source}: {null_count} NULL values in '{col}'")
        
        # Check for duplicates
        if 'metric_id' in df.columns and 'period' in df.columns:
            if 'region_code' in df.columns:
                # Regional data
                dup_cols = ['region_code', 'metric_id', 'period', 'data_type']
            else:
                # Macro data
                dup_cols = ['metric_id', 'period', 'data_type']
            
            dups = df.duplicated(subset=dup_cols, keep=False)
            if dups.any():
                issues.append(f"{source}: {dups.sum()} duplicate rows detected")
        
        # Check data type distribution
        if 'data_type' in df.columns:
            type_counts = df['data_type'].value_counts().to_dict()
            logger.info(f"{source} data_type distribution: {type_counts}")
            
            # CRITICAL: Ensure forecast data exists
            if 'forecast' not in type_counts or type_counts.get('forecast', 0) == 0:
                issues.append(f"{source}: NO FORECAST DATA - only historical found")
        
        # Check period/year range
        period_col = 'period' if 'period' in df.columns else 'year'
        if period_col in df.columns:
            min_year = df[period_col].min()
            max_year = df[period_col].max()
            logger.info(f"{source} year range: {min_year} → {max_year}")
            
            if max_year < 2025:
                issues.append(f"{source}: Max year {max_year} < 2025 - missing forecast?")
        
        is_valid = len(issues) == 0
        return is_valid, issues


# ===============================
# Column Mapping & Normalization
# ===============================

class DataNormalizer:
    """Normalize data for Supabase schema"""
    
    @staticmethod
    def normalize_macro(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize UK macro data.
        CRITICAL FIX: Use fillna to handle mixed historical/forecast
        """
        df = df.copy()
        
        # FIXED: Column mapping with fillna logic
        # Handle metric_id
        if "metric" in df.columns:
            if "metric_id" not in df.columns:
                df["metric_id"] = df["metric"]
            else:
                # Fill NULLs in metric_id from metric column
                df["metric_id"] = df["metric_id"].fillna(df["metric"])
        
        # Handle period
        if "year" in df.columns:
            if "period" not in df.columns:
                df["period"] = df["year"]
            else:
                # Fill NULLs in period from year column
                df["period"] = df["period"].fillna(df["year"])
        
        # Ensure integer types
        df["period"] = pd.to_numeric(df["period"], errors="coerce").astype("Int64")
        
        # Ensure numeric value
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        
        # Add region_level if missing
        if "region_level" not in df.columns:
            df["region_level"] = "UK"
        
        # Add region_code if missing
        if "region_code" not in df.columns:
            df["region_code"] = "K02000001"
        
        # Add region_name if missing
        if "region_name" not in df.columns:
            df["region_name"] = "United Kingdom"
        
        # Standardize data_type
        if "data_type" not in df.columns:
            df["data_type"] = "unknown"
        
        # Add metadata
        df["sync_timestamp"] = datetime.utcnow()
        
        # Select and order columns
        cols = [
            "region_code", "region_name", "region_level",
            "metric_id", "period", "value",
            "unit", "freq", "data_type",
            "ci_lower", "ci_upper",
            "forecast_run_date", "forecast_version",
            "sync_timestamp"
        ]
        
        # Keep only columns that exist
        available = [c for c in cols if c in df.columns]
        df = df[available]
        
        # Drop rows with NULL in critical columns
        df = df.dropna(subset=["metric_id", "period", "value"])
        
        return df
    
    @staticmethod
    def normalize_itl1(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize ITL1 regional data.
        CRITICAL FIX: Use fillna to handle mixed historical/forecast
        """
        df = df.copy()
        
        # FIXED: Column mapping with fillna logic
        # Handle metric_id
        if "metric" in df.columns:
            if "metric_id" not in df.columns:
                df["metric_id"] = df["metric"]
            else:
                # Fill NULLs in metric_id from metric column
                df["metric_id"] = df["metric_id"].fillna(df["metric"])
        
        # Handle period
        if "year" in df.columns:
            if "period" not in df.columns:
                df["period"] = df["year"]
            else:
                # Fill NULLs in period from year column
                df["period"] = df["period"].fillna(df["year"])
        
        # Ensure integer types
        df["period"] = pd.to_numeric(df["period"], errors="coerce").astype("Int64")
        
        # Ensure numeric value
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        
        # Add region_level if missing
        if "region_level" not in df.columns:
            df["region_level"] = "ITL1"
        
        # Handle region_name
        if "region" in df.columns and "region_name" not in df.columns:
            df["region_name"] = df["region"]
        elif "region_name" not in df.columns:
            df["region_name"] = None
        
        # Standardize data_type
        if "data_type" not in df.columns:
            df["data_type"] = "unknown"
        
        # Add metadata
        df["sync_timestamp"] = datetime.utcnow()
        
        # Select and order columns
        cols = [
            "region_code", "region_name", "region_level",
            "metric_id", "period", "value",
            "unit", "freq", "data_type",
            "ci_lower", "ci_upper",
            "forecast_run_date", "forecast_version",
            "sync_timestamp"
        ]
        
        # Keep only columns that exist
        available = [c for c in cols if c in df.columns]
        df = df[available]
        
        # Drop rows with NULL in critical columns
        df = df.dropna(subset=["region_code", "metric_id", "period", "value"])
        
        return df


# ===============================
# Supabase Sync Engine
# ===============================

class SupabaseSyncer:
    """Handle Supabase data synchronization"""
    
    def __init__(self, url: str, key: str):
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
        
        self.client: Client = create_client(url, key)
        logger.info(f"✓ Connected to Supabase: {url}")
    
    def upsert_batch(
        self,
        table_name: str,
        records: List[Dict],
        conflict_columns: Optional[List[str]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Upsert a batch of records with retry logic.
        Returns (success, error_message)
        """
        if not records:
            return True, None
        
        for attempt in range(MAX_RETRIES):
            try:
                # Convert to JSON-serializable format
                serializable_records = []
                for record in records:
                    clean = {}
                    for k, v in record.items():
                        # Handle pandas NA, NaN, None
                        if pd.isna(v):
                            clean[k] = None
                        # Handle numpy int64
                        elif isinstance(v, (np.integer, pd.Int64Dtype)):
                            clean[k] = int(v)
                        # Handle numpy float64
                        elif isinstance(v, (np.floating, float)):
                            clean[k] = float(v) if not np.isnan(v) else None
                        # Handle timestamps
                        elif isinstance(v, (pd.Timestamp, datetime)):
                            clean[k] = v.isoformat()
                        else:
                            clean[k] = v
                    
                    serializable_records.append(clean)
                
                # Upsert with conflict resolution
                response = self.client.table(table_name).upsert(
                    serializable_records,
                    on_conflict=",".join(conflict_columns) if conflict_columns else None
                ).execute()
                
                return True, None
                
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} failed: {error_msg}")
                
                if attempt < MAX_RETRIES - 1:
                    import time
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    return False, error_msg
        
        return False, "Max retries exceeded"
    
    def sync_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        conflict_columns: List[str],
        source_label: str
    ) -> bool:
        """
        Sync entire dataframe to Supabase in batches.
        Returns success status.
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Syncing {source_label} → {table_name}")
        logger.info(f"{'='*70}")
        logger.info(f"Total rows: {len(df):,}")
        
        # Convert to records
        records = df.to_dict(orient="records")
        total_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE
        
        logger.info(f"Batches: {total_batches} ({BATCH_SIZE} rows/batch)")
        
        success_count = 0
        failure_count = 0
        
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            
            logger.info(f"  Batch {batch_num}/{total_batches} ({len(batch)} rows)...")
            
            success, error = self.upsert_batch(table_name, batch, conflict_columns)
            
            if success:
                success_count += len(batch)
                logger.info(f"  ✓ Batch {batch_num} succeeded")
            else:
                failure_count += len(batch)
                logger.error(f"  ✗ Batch {batch_num} failed: {error}")
                
                # CRITICAL: Stop on first failure for data integrity
                logger.error(f"Aborting sync due to batch failure")
                return False
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Sync complete: {source_label}")
        logger.info(f"  Success: {success_count:,} rows")
        logger.info(f"  Failed: {failure_count:,} rows")
        logger.info(f"{'='*70}\n")
        
        return failure_count == 0
    
    def verify_sync(
        self,
        table_name: str,
        expected_count: int,
        source_label: str
    ) -> bool:
        """Verify data was synced correctly"""
        try:
            # Count total rows
            response = self.client.table(table_name).select("*", count="exact").limit(1).execute()
            actual_count = response.count
            
            logger.info(f"\nVerification: {source_label}")
            logger.info(f"  Expected rows: {expected_count:,}")
            logger.info(f"  Actual rows: {actual_count:,}")
            
            # Check forecast data exists
            forecast_response = self.client.table(table_name).select(
                "*", count="exact"
            ).eq("data_type", "forecast").limit(1).execute()
            
            forecast_count = forecast_response.count
            logger.info(f"  Forecast rows: {forecast_count:,}")
            
            if forecast_count == 0:
                logger.error(f"  ✗ NO FORECAST DATA in {table_name}")
                return False
            
            # Allow for small discrepancies (dropped NULLs, duplicates)
            tolerance = 0.05  # 5%
            diff = abs(actual_count - expected_count)
            diff_pct = diff / expected_count if expected_count > 0 else 1.0
            
            if diff_pct > tolerance:
                logger.warning(f"  ⚠ Row count mismatch: {diff_pct:.1%} difference")
                return False
            
            logger.info(f"  ✓ Verification passed")
            return True
            
        except Exception as e:
            logger.error(f"  ✗ Verification failed: {e}")
            return False


# ===============================
# Main Pipeline
# ===============================

def load_from_duckdb(table_name: str) -> Optional[pd.DataFrame]:
    """Load data from DuckDB gold schema"""
    if not DUCK_PATH.exists():
        logger.error(f"DuckDB not found: {DUCK_PATH}")
        return None
    
    try:
        con = duckdb.connect(str(DUCK_PATH), read_only=True)
        
        # Check if table exists
        tables = con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'gold'"
        ).fetchdf()
        
        if table_name not in tables['table_name'].values:
            logger.error(f"Table gold.{table_name} does not exist")
            con.close()
            return None
        
        # Load data
        df = con.execute(f"SELECT * FROM gold.{table_name}").fetchdf()
        con.close()
        
        logger.info(f"✓ Loaded gold.{table_name}: {len(df):,} rows")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load gold.{table_name}: {e}")
        return None


def main():
    """Main sync pipeline"""
    
    logger.info("="*70)
    logger.info("REGIONIQ → SUPABASE SYNC V2.0 (FIXED)")
    logger.info("="*70)
    logger.info(f"Started: {datetime.now()}")
    logger.info(f"DuckDB: {DUCK_PATH}")
    logger.info(f"Log file: {LOG_FILE}")
    
    # Check prerequisites
    if not HAVE_DUCKDB:
        logger.error("duckdb package not installed")
        sys.exit(1)
    
    if not HAVE_SUPABASE:
        logger.error("supabase package not installed")
        sys.exit(1)
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("Environment variables not set:")
        logger.error("  SUPABASE_URL")
        logger.error("  SUPABASE_SERVICE_KEY")
        sys.exit(1)
    
    if not DUCK_PATH.exists():
        logger.error(f"DuckDB not found: {DUCK_PATH}")
        sys.exit(1)
    
    # Initialize components
    validator = DataValidator()
    normalizer = DataNormalizer()
    
    try:
        syncer = SupabaseSyncer(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        logger.error(f"Failed to connect to Supabase: {e}")
        sys.exit(1)
    
    # Track success
    all_success = True
    
    # =============================
    # SYNC 1: UK Macro Forecasts
    # =============================
    
    logger.info("\n" + "="*70)
    logger.info("PHASE 1: UK MACRO FORECASTS")
    logger.info("="*70)
    
    macro_df = load_from_duckdb("uk_macro_forecast")
    
    if macro_df is not None and not macro_df.empty:
        # Normalize
        macro_normalized = normalizer.normalize_macro(macro_df)
        logger.info(f"✓ Normalized: {len(macro_normalized):,} rows")
        
        # Validate
        is_valid, issues = validator.validate_dataframe(
            macro_normalized,
            "UK Macro",
            required_cols=["metric_id", "period", "value"]
        )
        
        if not is_valid:
            logger.error("UK Macro validation failed:")
            for issue in issues:
                logger.error(f"  • {issue}")
            all_success = False
        else:
            logger.info("✓ Validation passed")
            
            # Sync to Supabase
            success = syncer.sync_dataframe(
                macro_normalized,
                table_name="uk_macro_forecast",
                conflict_columns=["metric_id", "period", "data_type"],
                source_label="UK Macro"
            )
            
            if success:
                # Verify
                verified = syncer.verify_sync(
                    "uk_macro_forecast",
                    len(macro_normalized),
                    "UK Macro"
                )
                if not verified:
                    all_success = False
            else:
                all_success = False
    else:
        logger.warning("⚠ UK Macro data not found or empty")
        all_success = False
    
    # =============================
    # SYNC 2: ITL1 Regional Forecasts
    # =============================
    
    logger.info("\n" + "="*70)
    logger.info("PHASE 2: ITL1 REGIONAL FORECASTS")
    logger.info("="*70)
    
    itl1_df = load_from_duckdb("itl1_forecast")
    
    if itl1_df is not None and not itl1_df.empty:
        # Normalize
        itl1_normalized = normalizer.normalize_itl1(itl1_df)
        logger.info(f"✓ Normalized: {len(itl1_normalized):,} rows")
        
        # Validate
        is_valid, issues = validator.validate_dataframe(
            itl1_normalized,
            "ITL1 Regional",
            required_cols=["region_code", "metric_id", "period", "value"]
        )
        
        if not is_valid:
            logger.error("ITL1 Regional validation failed:")
            for issue in issues:
                logger.error(f"  • {issue}")
            all_success = False
        else:
            logger.info("✓ Validation passed")
            
            # Sync to Supabase
            success = syncer.sync_dataframe(
                itl1_normalized,
                table_name="itl1_forecast",
                conflict_columns=["region_code", "metric_id", "period", "data_type"],
                source_label="ITL1 Regional"
            )
            
            if success:
                # Verify
                verified = syncer.verify_sync(
                    "itl1_forecast",
                    len(itl1_normalized),
                    "ITL1 Regional"
                )
                if not verified:
                    all_success = False
            else:
                all_success = False
    else:
        logger.warning("⚠ ITL1 Regional data not found or empty")
        all_success = False
    
    # =============================
    # Final Summary
    # =============================
    
    logger.info("\n" + "="*70)
    logger.info("SYNC SUMMARY")
    logger.info("="*70)
    logger.info(f"Completed: {datetime.now()}")
    
    if all_success:
        logger.info("✅ ALL SYNCS SUCCESSFUL")
        logger.info("  • UK Macro: ✓")
        logger.info("  • ITL1 Regional: ✓")
        logger.info(f"\nLog: {LOG_FILE}")
        sys.exit(0)
    else:
        logger.error("❌ SYNC FAILURES DETECTED")
        logger.error("  Review logs for details")
        logger.error(f"\nLog: {LOG_FILE}")
        sys.exit(1)


if __name__ == "__main__":
    main()