#!/usr/bin/env python3
"""
Unified LAD to ITL Aggregation Script (Production v1.2 - Labour Market Rates)

Takes LAD-level data for ALL metrics and aggregates up to ITL3, ITL2, ITL1 levels.
Creates consistent bottom-up hierarchy for forecasting.

Inputs:
- silver.lad_population_history
- silver.lad_population_16_64_history
- silver.lad_employment_history
- silver.lad_gdhi_history
- silver.lad_gva_history
- silver.lad_employment_rate_history
- silver.lad_unemployment_rate_history

Outputs:
- silver.lad_history (unified LAD data)
- silver.itl3_history (bottom-up, canonical)
- silver.itl2_history (bottom-up, canonical)
- silver.itl1_unified_bottomup (validation only)
- Comparison reports vs top-down ITL1 if available
- data/logs/lad_aggregation_summary.json (pipeline status)

Rate metrics (employment_rate, unemployment_rate) are aggregated using
population-weighted averages rather than sums.
"""

import json
import logging
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple

try:
    import duckdb
    HAVE_DUCKDB = True
except ImportError:
    HAVE_DUCKDB = False

# -----------------------------
# Configuration
# -----------------------------
SILVER_DIR = Path("data/silver")
SILVER_DIR.mkdir(parents=True, exist_ok=True)

LAKE_DIR = Path("data/lake")
DUCK_PATH = LAKE_DIR / "warehouse.duckdb"

LOG_DIR = Path("data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

VINTAGE = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("unified_aggregation")

# ============================================================================
# PIPELINE REPORTER
# ============================================================================

class PipelineReporter:
    """
    Structured pipeline status reporter for RegionIQ forecasting governance.
    
    Tracks warnings, critical errors, and metrics for orchestrator consumption.
    Emits machine-readable JSON summaries for automated monitoring.
    """
    
    def __init__(self, stage_name: str):
        self.stage = stage_name
        self.start_time = time.time()
        self.warnings = []
        self.critical_errors = []
        self.metrics = {}
        self.status = "success"  # success | warning | failed
    
    def add_warning(self, message: str):
        """Log a warning (non-blocking issue)"""
        self.warnings.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": message
        })
        log.warning(f"⚠️  {message}")
        if self.status == "success":
            self.status = "warning"
    
    def add_critical_error(self, message: str):
        """Log a critical error (pipeline-blocking issue)"""
        self.critical_errors.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": message
        })
        log.error(f"❌ CRITICAL: {message}")
        self.status = "failed"
    
    def add_metric(self, key: str, value):
        """Add a metric to the summary"""
        self.metrics[key] = value
    
    def finalize(self) -> Dict:
        """Generate final pipeline summary"""
        duration = time.time() - self.start_time
        
        summary = {
            "stage": self.stage,
            "status": self.status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(duration, 2),
            "warnings_count": len(self.warnings),
            "critical_errors_count": len(self.critical_errors),
            "warnings": self.warnings,
            "critical_errors": self.critical_errors,
            "metrics": self.metrics
        }
        
        return summary
    
    def save_and_exit(self, filename: str):
        """Save summary to JSON and exit with appropriate code"""
        summary = self.finalize()
        
        output_path = LOG_DIR / filename
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        log.info(f"\n✓ Pipeline summary → {output_path}")
        
        # Exit with appropriate code
        exit_code = 0 if self.status in ["success", "warning"] else 1
        
        if self.status == "success":
            log.info("✅ Pipeline completed successfully")
        elif self.status == "warning":
            log.warning("⚠️  Pipeline completed with warnings")
        else:
            log.error("❌ Pipeline failed with critical errors")
        
        sys.exit(exit_code)

# Metric configurations
METRICS = {
    'population': {
        'table': 'lad_population_history',
        'metric_id': 'population_total',
        'unit': 'persons',
        'display_name': 'Population',
        'is_rate': False
    },
    'population_16_64': {
        'table': 'lad_population_16_64_history',
        'metric_id': 'population_16_64',
        'unit': 'persons',
        'display_name': 'Working Age Population (16-64)',
        'is_rate': False
    },
    'employment': {
        'table': 'lad_employment_history',
        'metric_id': 'emp_total_jobs',
        'unit': 'jobs',
        'display_name': 'Employment',
        'is_rate': False
    },
    'gdhi': {
        'table': 'lad_gdhi_history',
        'metric_id': 'gdhi_total_mn_gbp',
        'unit': 'GBP_m',
        'display_name': 'GDHI Total',
        'is_rate': False
    },
    'gva': {
        'table': 'lad_gva_history',
        'metric_id': 'nominal_gva_mn_gbp',
        'unit': 'GBP_m',
        'display_name': 'GVA',
        'is_rate': False
    },
    'employment_rate': {
        'table': 'lad_employment_rate_history',
        'metric_id': 'employment_rate_pct',
        'unit': 'percent',
        'display_name': 'Employment Rate',
        'is_rate': True
    },
    'unemployment_rate': {
        'table': 'lad_unemployment_rate_history',
        'metric_id': 'unemployment_rate_pct',
        'unit': 'percent',
        'display_name': 'Unemployment Rate',
        'is_rate': True
    }
}

# Separate lists for aggregation logic
ADDITIVE_METRICS = [k for k, v in METRICS.items() if not v['is_rate']]
RATE_METRICS = [k for k, v in METRICS.items() if v['is_rate']]

# -----------------------------
# Helper Functions
# -----------------------------

def load_lad_metric(metric_key: str, reporter: PipelineReporter) -> pd.DataFrame:
    """Load LAD data for a specific metric from DuckDB or CSV"""
    config = METRICS[metric_key]
    table_name = config['table']
    
    if HAVE_DUCKDB and DUCK_PATH.exists():
        log.info(f"Loading {config['display_name']} from DuckDB...")
        con = duckdb.connect(str(DUCK_PATH), read_only=True)
        try:
            df = con.execute(f"SELECT * FROM silver.{table_name}").fetchdf()
            log.info(f"  Loaded {len(df)} observations from DuckDB")
            return df
        except Exception as e:
            reporter.add_warning(f"DuckDB load failed for {metric_key}: {e}, trying CSV...")
        finally:
            con.close()
    
    # Fallback to CSV
    csv_path = SILVER_DIR / f"{table_name}.csv"
    if not csv_path.exists():
        # Rate metrics may not exist yet - warning not critical
        if config['is_rate']:
            reporter.add_warning(f"Rate metric data not found: {csv_path}")
        else:
            reporter.add_critical_error(f"LAD data not found: {csv_path}")
        return pd.DataFrame()
    
    log.info(f"Loading {config['display_name']} from CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    log.info(f"  Loaded {len(df)} observations from CSV")
    return df


def load_all_lad_data(reporter: PipelineReporter) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and combine all LAD metrics into dataframes.
    
    Returns:
        Tuple of (additive_data, rate_data) DataFrames
    """
    additive_data = []
    rate_data = []
    
    for metric_key, config in METRICS.items():
        df = load_lad_metric(metric_key, reporter)
        if not df.empty:
            if config['is_rate']:
                rate_data.append(df)
            else:
                additive_data.append(df)
        else:
            if not config['is_rate']:
                reporter.add_critical_error(f"No data loaded for {metric_key}")
    
    # Combine additive metrics
    if not additive_data:
        reporter.add_critical_error("No LAD data found for any additive metric!")
        return pd.DataFrame(), pd.DataFrame()
    
    combined_additive = pd.concat(additive_data, ignore_index=True)
    log.info(f"\n✓ Combined additive LAD data: {len(combined_additive)} total observations")
    log.info(f"  Metrics: {combined_additive['metric_id'].unique().tolist()}")
    log.info(f"  LADs: {combined_additive['region_code'].nunique()}")
    log.info(f"  Year range: {combined_additive['period'].min()} - {combined_additive['period'].max()}")
    
    reporter.add_metric("lad_additive_observations", len(combined_additive))
    reporter.add_metric("lad_unique_codes", combined_additive['region_code'].nunique())
    reporter.add_metric("lad_additive_metrics_loaded", combined_additive['metric_id'].nunique())
    reporter.add_metric("lad_year_range", f"{combined_additive['period'].min()}-{combined_additive['period'].max()}")
    
    # Combine rate metrics (may be empty)
    combined_rates = pd.DataFrame()
    if rate_data:
        combined_rates = pd.concat(rate_data, ignore_index=True)
        log.info(f"\n✓ Combined rate LAD data: {len(combined_rates)} total observations")
        log.info(f"  Metrics: {combined_rates['metric_id'].unique().tolist()}")
        log.info(f"  LADs: {combined_rates['region_code'].nunique()}")
        log.info(f"  Year range: {combined_rates['period'].min()} - {combined_rates['period'].max()}")
        
        reporter.add_metric("lad_rate_observations", len(combined_rates))
        reporter.add_metric("lad_rate_metrics_loaded", combined_rates['metric_id'].nunique())
    else:
        log.info("\n⚠️  No rate metrics loaded (employment_rate, unemployment_rate)")
        reporter.add_warning("No rate metrics loaded")
    
    # Write unified silver.lad_history (mirrors ITL1/2/3 pattern)
    unified_lad = pd.concat([combined_additive, combined_rates], ignore_index=True) if not combined_rates.empty else combined_additive
    try:
        con = duckdb.connect(str(DUCK_PATH))
        con.execute("CREATE SCHEMA IF NOT EXISTS silver")
        
        # Check if data_quality column exists in existing table and preserve it
        existing_cols = con.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = 'silver' AND table_name = 'lad_history'
        """).fetchdf()
        
        has_data_quality = not existing_cols.empty and 'data_quality' in existing_cols['column_name'].values
        
        if has_data_quality and 'data_quality' not in unified_lad.columns:
            # Preserve data_quality from existing table for rate metrics
            try:
                existing_with_quality = con.execute("""
                    SELECT region_code, metric_id, period, data_quality
                    FROM silver.lad_history
                    WHERE metric_id IN ('unemployment_rate_pct', 'employment_rate_pct')
                """).fetchdf()
                
                if not existing_with_quality.empty:
                    unified_lad = unified_lad.merge(
                        existing_with_quality,
                        on=['region_code', 'metric_id', 'period'],
                        how='left'
                    )
                    log.info("  ✓ Preserved data_quality column from existing table")
            except Exception as e:
                log.warning(f"  Could not preserve data_quality: {e}")
        
        # Add data_quality = 'ONS' for rows that don't have it
        if 'data_quality' not in unified_lad.columns:
            unified_lad['data_quality'] = 'ONS'
        else:
            unified_lad['data_quality'] = unified_lad['data_quality'].fillna('ONS')
        
        con.register("df_tmp", unified_lad)
        con.execute("CREATE OR REPLACE TABLE silver.lad_history AS SELECT * FROM df_tmp")
        con.close()
        log.info(f"✓ Wrote silver.lad_history ({len(unified_lad)} rows)")
        reporter.add_metric("lad_unified_rows", len(unified_lad))
    except Exception as e:
        log.warning(f"Failed to write silver.lad_history: {e}")
    
    return combined_additive, combined_rates


def calculate_historical_derived_metrics(df: pd.DataFrame, reporter: PipelineReporter) -> pd.DataFrame:
    """
    Calculate derived metrics from aggregated historical data.
    Same logic as forecast script, but for historical years.
    
    Calculates:
    - gdhi_per_head_gbp = (gdhi_total_mn_gbp * 1e6) / population_total
    - productivity_gbp_per_job = (nominal_gva_mn_gbp * 1e6) / emp_total_jobs
    
    Returns:
        DataFrame with original data + derived metrics appended
    """
    derived_rows = []
    
    # Group by region and period
    for (region_code, period), group in df.groupby(['region_code', 'period']):
        # Pivot metrics for this region-year
        metrics = {row['metric_id']: row['value'] for _, row in group.iterrows()}
        
        # Get metadata (should be same for all rows in group)
        region_name = group['region_name'].iloc[0]
        region_level = group['region_level'].iloc[0]
        vintage = group['vintage'].iloc[0]
        geo_hierarchy = group['geo_hierarchy'].iloc[0]
        
        # Calculate GDHI per head
        if 'gdhi_total_mn_gbp' in metrics and 'population_total' in metrics:
            gdhi_total = metrics['gdhi_total_mn_gbp']
            population = metrics['population_total']
            
            if population > 0:
                gdhi_per_head = (gdhi_total * 1e6) / population
                
                derived_rows.append({
                    'region_code': region_code,
                    'region_name': region_name,
                    'region_level': region_level,
                    'metric_id': 'gdhi_per_head_gbp',
                    'period': period,
                    'value': gdhi_per_head,
                    'unit': 'GBP',
                    'freq': 'A',
                    'source': 'calculated_from_aggregated',
                    'vintage': vintage,
                    'geo_hierarchy': geo_hierarchy
                })
        
        # Calculate productivity (GVA per job)
        if 'nominal_gva_mn_gbp' in metrics and 'emp_total_jobs' in metrics:
            gva_total = metrics['nominal_gva_mn_gbp']
            jobs = metrics['emp_total_jobs']
            
            if jobs > 0:
                productivity = (gva_total * 1e6) / jobs
                
                derived_rows.append({
                    'region_code': region_code,
                    'region_name': region_name,
                    'region_level': region_level,
                    'metric_id': 'productivity_gbp_per_job',
                    'period': period,
                    'value': productivity,
                    'unit': 'GBP',
                    'freq': 'A',
                    'source': 'calculated_from_aggregated',
                    'vintage': vintage,
                    'geo_hierarchy': geo_hierarchy
                })
    
    if not derived_rows:
        reporter.add_warning("No derived metrics calculated (missing base metrics)")
        return df
    
    derived_df = pd.DataFrame(derived_rows)
    
    # Count by metric type
    derived_counts = derived_df['metric_id'].value_counts()
    for metric_id, count in derived_counts.items():
        log.info(f"  ✓ Calculated {count:,} {metric_id} observations")
        reporter.add_metric(f"derived_{metric_id}_count", count)
    
    # Combine with original data
    result = pd.concat([df, derived_df], ignore_index=True)
    result = result.sort_values(['metric_id', 'region_code', 'period']).reset_index(drop=True)
    
    return result


def aggregate_additive_to_level(
    df: pd.DataFrame, 
    level_code_col: str,
    level_name_col: str,
    region_level: str,
    reporter: PipelineReporter
) -> pd.DataFrame:
    """
    Aggregate additive LAD data to a higher geography level.
    Uses SUM for additive metrics (population, jobs, GVA, GDHI).
    
    Args:
        df: LAD data with parent geography columns and multiple metrics
        level_code_col: Column name for target level code (e.g., 'itl3_code')
        level_name_col: Column name for target level name (e.g., 'itl3_name')
        region_level: String identifier for output (e.g., 'ITL3')
        reporter: Pipeline reporter for warnings/errors
    
    Returns:
        Aggregated dataframe in tidy schema with all additive metrics
    """
    # Check required columns
    required = [level_code_col, level_name_col, 'period', 'value', 'metric_id']
    missing = [c for c in required if c not in df.columns]
    if missing:
        reporter.add_critical_error(f"Missing columns for {region_level} aggregation: {missing}")
        return pd.DataFrame()
    
    # Remove rows with null parent geography codes
    df_clean = df.dropna(subset=[level_code_col, level_name_col]).copy()
    
    if len(df_clean) < len(df):
        dropped = len(df) - len(df_clean)
        reporter.add_warning(f"Dropped {dropped} rows with missing {level_code_col}")
    
    # Aggregate by geography, period, and metric (SUM for additive)
    agg = df_clean.groupby([
        level_code_col, level_name_col, 'period', 'metric_id', 'unit'
    ], as_index=False).agg({
        'value': 'sum'
    })
    
    # Rename to standard schema
    agg = agg.rename(columns={
        level_code_col: 'region_code',
        level_name_col: 'region_name'
    })
    
    # Add metadata
    agg['region_level'] = region_level
    agg['freq'] = 'A'
    agg['source'] = 'NOMIS_LAD_aggregated'
    agg['vintage'] = VINTAGE
    
    # Add geo_hierarchy
    if region_level == 'ITL3':
        agg['geo_hierarchy'] = 'ITL3>ITL2>ITL1'
    elif region_level == 'ITL2':
        agg['geo_hierarchy'] = 'ITL2>ITL1'
    elif region_level == 'ITL1':
        agg['geo_hierarchy'] = 'ITL1'
    
    # Reorder columns to match schema
    cols = ['region_code', 'region_name', 'region_level', 'metric_id',
            'period', 'value', 'unit', 'freq', 'source', 'vintage', 'geo_hierarchy']
    agg = agg[cols]
    
    # Sort
    agg = agg.sort_values(['metric_id', 'region_code', 'period']).reset_index(drop=True)
    
    log.info(f"  {region_level} (additive): {agg['region_code'].nunique()} regions × {agg['metric_id'].nunique()} metrics = {len(agg)} observations")
    
    return agg


def aggregate_rates_to_level(
    rate_df: pd.DataFrame,
    pop_df: pd.DataFrame,
    level_code_col: str,
    level_name_col: str,
    region_level: str,
    reporter: PipelineReporter
) -> pd.DataFrame:
    """
    Aggregate rate metrics to a higher geography level using population-weighted averages.
    
    Rate at aggregate level = Σ(rate_i × pop_i) / Σ(pop_i)
    
    Args:
        rate_df: LAD rate data (employment_rate, unemployment_rate)
        pop_df: LAD population data for weighting
        level_code_col: Column name for target level code
        level_name_col: Column name for target level name
        region_level: String identifier for output
        reporter: Pipeline reporter
    
    Returns:
        Aggregated rate dataframe
    """
    if rate_df.empty:
        return pd.DataFrame()
    
    # Check required columns
    required = [level_code_col, level_name_col, 'period', 'value', 'metric_id', 'region_code']
    missing = [c for c in required if c not in rate_df.columns]
    if missing:
        reporter.add_critical_error(f"Missing columns for {region_level} rate aggregation: {missing}")
        return pd.DataFrame()
    
    # Get population for weighting
    pop_subset = pop_df[pop_df['metric_id'] == 'population_total'][
        ['region_code', 'period', 'value']
    ].rename(columns={'value': 'population'})
    
    # Get max and min population years available
    if pop_subset.empty:
        reporter.add_warning(f"No population data available for {region_level} rate weighting")
        return pd.DataFrame()
    
    max_pop_year = pop_subset['period'].max()
    min_pop_year = pop_subset['period'].min()
    
    # For rate years beyond max population year, use max population year's weights
    # For rate years before min population year, use min population year's weights
    rate_df_adj = rate_df.copy()
    rate_df_adj['pop_period'] = rate_df_adj['period'].clip(lower=min_pop_year, upper=max_pop_year)
    
    # Rename pop_subset period column for merge
    pop_subset_for_merge = pop_subset.rename(columns={'period': 'pop_period'})
    
    # Merge rates with population using adjusted period
    rate_with_pop = rate_df_adj.merge(
        pop_subset_for_merge,
        on=['region_code', 'pop_period'],
        how='left'
    )
    
    # Log if we're using proxy population years
    rate_years_beyond_pop = rate_df_adj[rate_df_adj['period'] > max_pop_year]['period'].nunique()
    if rate_years_beyond_pop > 0:
        log.info(f"  ℹ️  Using {max_pop_year} population weights for {rate_years_beyond_pop} year(s) beyond population data")
    
    rate_years_before_pop = rate_df_adj[rate_df_adj['period'] < min_pop_year]['period'].nunique()
    if rate_years_before_pop > 0:
        log.info(f"  ℹ️  Using {min_pop_year} population weights for {rate_years_before_pop} year(s) before population data")
    
    # Check for missing population weights (should be rare now - only if LAD not in population data)
    missing_pop = rate_with_pop['population'].isna().sum()
    if missing_pop > 0:
        reporter.add_warning(f"{missing_pop} rate observations missing population weights (LAD not in population data)")
        # Drop rows without population (can't weight them)
        rate_with_pop = rate_with_pop.dropna(subset=['population'])
    
    # Drop the temporary pop_period column before further processing
    rate_with_pop = rate_with_pop.drop(columns=['pop_period'])
    
    if rate_with_pop.empty:
        reporter.add_warning(f"No rate data with population weights for {region_level}")
        return pd.DataFrame()
    
    # Calculate weighted values
    rate_with_pop['weighted_value'] = rate_with_pop['value'] * rate_with_pop['population']
    
    # Remove rows with null parent geography codes
    rate_with_pop = rate_with_pop.dropna(subset=[level_code_col, level_name_col])
    
    # Aggregate: sum(weighted_value) / sum(population)
    agg = rate_with_pop.groupby([
        level_code_col, level_name_col, 'period', 'metric_id', 'unit'
    ], as_index=False).agg({
        'weighted_value': 'sum',
        'population': 'sum'
    })
    
    # Calculate final weighted average
    agg['value'] = agg['weighted_value'] / agg['population']
    agg = agg.drop(columns=['weighted_value', 'population'])
    
    # Rename to standard schema
    agg = agg.rename(columns={
        level_code_col: 'region_code',
        level_name_col: 'region_name'
    })
    
    # Add metadata
    agg['region_level'] = region_level
    agg['freq'] = 'A'
    agg['source'] = 'NOMIS_LAD_aggregated_popweighted'
    agg['vintage'] = VINTAGE
    
    # Add geo_hierarchy
    if region_level == 'ITL3':
        agg['geo_hierarchy'] = 'ITL3>ITL2>ITL1'
    elif region_level == 'ITL2':
        agg['geo_hierarchy'] = 'ITL2>ITL1'
    elif region_level == 'ITL1':
        agg['geo_hierarchy'] = 'ITL1'
    
    # Reorder columns
    cols = ['region_code', 'region_name', 'region_level', 'metric_id',
            'period', 'value', 'unit', 'freq', 'source', 'vintage', 'geo_hierarchy']
    agg = agg[cols]
    
    # Sort
    agg = agg.sort_values(['metric_id', 'region_code', 'period']).reset_index(drop=True)
    
    log.info(f"  {region_level} (rates): {agg['region_code'].nunique()} regions × {agg['metric_id'].nunique()} metrics = {len(agg)} observations")
    
    return agg


def aggregate_to_level(
    additive_df: pd.DataFrame,
    rate_df: pd.DataFrame,
    level_code_col: str,
    level_name_col: str,
    region_level: str,
    reporter: PipelineReporter
) -> pd.DataFrame:
    """
    Aggregate all LAD data to a higher geography level.
    
    - Additive metrics (pop, jobs, GVA, GDHI): SUM
    - Rate metrics (emp_rate, unemp_rate): Population-weighted average
    
    Returns combined dataframe with all metrics.
    """
    # Aggregate additive metrics
    additive_agg = aggregate_additive_to_level(
        additive_df, level_code_col, level_name_col, region_level, reporter
    )
    
    # Aggregate rate metrics (if available)
    rate_agg = pd.DataFrame()
    if not rate_df.empty:
        rate_agg = aggregate_rates_to_level(
            rate_df, additive_df, level_code_col, level_name_col, region_level, reporter
        )
    
    # Combine
    if rate_agg.empty:
        combined = additive_agg
    else:
        combined = pd.concat([additive_agg, rate_agg], ignore_index=True)
    
    combined = combined.sort_values(['metric_id', 'region_code', 'period']).reset_index(drop=True)
    
    reporter.add_metric(f"{region_level.lower()}_regions", combined['region_code'].nunique())
    reporter.add_metric(f"{region_level.lower()}_metrics", combined['metric_id'].nunique())
    reporter.add_metric(f"{region_level.lower()}_observations", len(combined))
    
    return combined


def write_to_duckdb(df: pd.DataFrame, table_name: str, reporter: PipelineReporter):
    """Write aggregated data to DuckDB silver schema"""
    if not HAVE_DUCKDB:
        reporter.add_warning("duckdb not installed; skipping DuckDB write")
        return
    
    con = duckdb.connect(str(DUCK_PATH))
    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS silver")
        con.register("df_tmp", df)
        con.execute(f"CREATE OR REPLACE TABLE silver.{table_name} AS SELECT * FROM df_tmp")
        log.info(f"  ✓ Wrote {len(df)} rows to silver.{table_name}")
    except Exception as e:
        reporter.add_critical_error(f"DuckDB write failed for silver.{table_name}: {e}")
    finally:
        con.close()


def compare_with_topdown(bottomup_itl1: pd.DataFrame, reporter: PipelineReporter):
    """Compare bottom-up ITL1 with existing top-down ITL1 for all metrics"""
    
    # Try to load top-down data
    topdown = None
    
    comparable_metrics = [
        'population_total', 'emp_total_jobs', 'gdhi_total_mn_gbp', 'nominal_gva_mn_gbp'
    ]
    
    if HAVE_DUCKDB and DUCK_PATH.exists():
        try:
            con = duckdb.connect(str(DUCK_PATH), read_only=True)
            topdown = con.execute(f"""
                SELECT region_code, period, metric_id, value 
                FROM silver.itl1_history 
                WHERE metric_id IN ({','.join([f"'{m}'" for m in comparable_metrics])})
            """).fetchdf()
            con.close()
        except:
            pass
    
    if topdown is None or topdown.empty:
        # Try CSV fallback
        csv_path = SILVER_DIR / "itl1_unified_history.csv"
        if csv_path.exists():
            topdown_all = pd.read_csv(csv_path)
            topdown = topdown_all[topdown_all['metric_id'].isin(comparable_metrics)].copy()
    
    if topdown is None or topdown.empty:
        log.info("\n  No top-down ITL1 data available for comparison")
        reporter.add_metric("topdown_comparison", "not_available")
        return
    
    log.info("\n" + "="*70)
    log.info("BOTTOM-UP vs TOP-DOWN COMPARISON")
    log.info("="*70)
    log.info(f"Top-down data: {len(topdown)} observations")
    
    # Filter bottom-up to comparable metrics
    bottomup_comparable = bottomup_itl1[bottomup_itl1['metric_id'].isin(comparable_metrics)]
    
    # Merge on region_code, period, and metric_id
    comparison = bottomup_comparable.merge(
        topdown[['region_code', 'period', 'metric_id', 'value']],
        on=['region_code', 'period', 'metric_id'],
        how='inner',
        suffixes=('_bottomup', '_topdown')
    )
    
    if comparison.empty:
        reporter.add_warning("No overlapping data between bottom-up and top-down")
        return
    
    # Calculate differences
    comparison['diff'] = comparison['value_bottomup'] - comparison['value_topdown']
    comparison['diff_pct'] = (comparison['diff'] / comparison['value_topdown'] * 100).abs()
    
    # Overall summary
    avg_diff = comparison['diff_pct'].mean()
    median_diff = comparison['diff_pct'].median()
    max_diff = comparison['diff_pct'].max()
    
    log.info(f"\nMatched {len(comparison)} observations across comparable metrics")
    log.info(f"  Average absolute difference: {avg_diff:.3f}%")
    log.info(f"  Median absolute difference: {median_diff:.3f}%")
    log.info(f"  Maximum absolute difference: {max_diff:.3f}%")
    
    reporter.add_metric("topdown_avg_diff_pct", round(avg_diff, 3))
    reporter.add_metric("topdown_median_diff_pct", round(median_diff, 3))
    reporter.add_metric("topdown_max_diff_pct", round(max_diff, 3))
    
    # Add warning if differences are large
    if max_diff > 2.0:
        reporter.add_warning(f"Large top-down discrepancy detected: {max_diff:.2f}% maximum difference")
    
    # By metric summary
    log.info("\n" + "="*70)
    log.info("ACCURACY BY METRIC")
    log.info("="*70)
    
    for metric_id in comparison['metric_id'].unique():
        metric_comp = comparison[comparison['metric_id'] == metric_id]
        
        # Get display name
        display_name = next(
            (v['display_name'] for k, v in METRICS.items() if v['metric_id'] == metric_id),
            metric_id
        )
        
        avg_diff = metric_comp['diff_pct'].mean()
        max_diff = metric_comp['diff_pct'].max()
        n_large = (metric_comp['diff_pct'] > 1.0).sum()
        
        status = "✅" if max_diff < 1.0 else "⚠️ "
        log.info(f"{status} {display_name}:")
        log.info(f"   Avg: {avg_diff:.3f}%, Max: {max_diff:.3f}%, >1%: {n_large}/{len(metric_comp)}")
        
        # Store in metrics
        reporter.add_metric(f"topdown_{metric_id}_avg_diff_pct", round(avg_diff, 3))
        reporter.add_metric(f"topdown_{metric_id}_max_diff_pct", round(max_diff, 3))
        
        # Show worst cases if any
        if n_large > 0:
            worst = metric_comp.nlargest(3, 'diff_pct')
            for _, row in worst.iterrows():
                log.info(
                    f"     {row['region_code']} ({row['period']}): "
                    f"{row['diff_pct']:.2f}% diff"
                )


def validate_duckdb_tables(expected_counts: Dict[str, int], reporter: PipelineReporter) -> bool:
    """
    Validate that DuckDB silver tables were created correctly.
    
    Args:
        expected_counts: Dict mapping table names to expected row counts
        reporter: Pipeline reporter for tracking issues
    
    Returns:
        True if all validations pass, False otherwise
    """
    if not HAVE_DUCKDB or not DUCK_PATH.exists():
        reporter.add_warning("DuckDB not available - skipping validation")
        return True
    
    log.info("\n" + "="*70)
    log.info("DUCKDB TABLE VALIDATION")
    log.info("="*70)
    
    con = duckdb.connect(str(DUCK_PATH), read_only=True)
    validation_passed = True
    
    try:
        # Expected metrics: 5 base + 2 derived + 2 rates = 9 total (if rates available)
        expected_base_metrics = set([
            'population_total',
            'population_16_64',
            'emp_total_jobs', 
            'gdhi_total_mn_gbp', 
            'nominal_gva_mn_gbp'
        ])
        
        expected_derived_metrics = set([
            'gdhi_per_head_gbp',
            'productivity_gbp_per_job'
        ])
        
        expected_rate_metrics = set([
            'employment_rate_pct',
            'unemployment_rate_pct'
        ])
        
        for table_name, expected_count in expected_counts.items():
            log.info(f"\n✓ Validating silver.{table_name}...")
            
            # Check table exists
            try:
                result = con.execute(f"SELECT COUNT(*) FROM silver.{table_name}").fetchone()
                actual_count = result[0]
            except Exception as e:
                reporter.add_critical_error(f"Table silver.{table_name} does not exist: {e}")
                validation_passed = False
                continue
            
            # Check row count
            if actual_count != expected_count:
                reporter.add_critical_error(
                    f"Row count mismatch in {table_name}: expected {expected_count}, got {actual_count}"
                )
                validation_passed = False
            else:
                log.info(f"  ✓ Row count: {actual_count:,}")
            
            # Check metrics present
            metrics_result = con.execute(f"""
                SELECT DISTINCT metric_id 
                FROM silver.{table_name}
            """).fetchdf()
            actual_metrics = set(metrics_result['metric_id'].tolist())
            
            # Check base metrics (mandatory)
            missing_base = expected_base_metrics - actual_metrics
            if missing_base:
                reporter.add_critical_error(f"Missing base metrics in {table_name}: {missing_base}")
                validation_passed = False
            else:
                log.info(f"  ✓ All 5 base metrics present")
            
            # Check derived metrics (should be present if base metrics are)
            missing_derived = expected_derived_metrics - actual_metrics
            if missing_derived:
                reporter.add_warning(f"Missing derived metrics in {table_name}: {missing_derived}")
            else:
                log.info(f"  ✓ All 2 derived metrics present")
            
            # Check rate metrics (optional - may not be available)
            present_rates = expected_rate_metrics & actual_metrics
            if present_rates:
                log.info(f"  ✓ Rate metrics present: {present_rates}")
            else:
                log.info(f"  ℹ️  No rate metrics (employment_rate, unemployment_rate)")
            
            log.info(f"  → Total metrics: {len(actual_metrics)}")
            
            # Check for NULLs in critical columns
            null_check = con.execute(f"""
                SELECT 
                    SUM(CASE WHEN region_code IS NULL THEN 1 ELSE 0 END) as null_region_code,
                    SUM(CASE WHEN metric_id IS NULL THEN 1 ELSE 0 END) as null_metric_id,
                    SUM(CASE WHEN period IS NULL THEN 1 ELSE 0 END) as null_period,
                    SUM(CASE WHEN value IS NULL THEN 1 ELSE 0 END) as null_value
                FROM silver.{table_name}
            """).fetchone()
            
            total_nulls = sum(null_check)
            if total_nulls > 0:
                reporter.add_critical_error(
                    f"Found {total_nulls} NULL values in {table_name} critical columns"
                )
                validation_passed = False
            else:
                log.info(f"  ✓ No NULL values in critical columns")
            
            # Get regions and year range
            stats = con.execute(f"""
                SELECT 
                    COUNT(DISTINCT region_code) as n_regions,
                    MIN(period) as min_year,
                    MAX(period) as max_year
                FROM silver.{table_name}
            """).fetchone()
            
            log.info(f"  ✓ Regions: {stats[0]}, Years: {stats[1]}-{stats[2]}")
    
    finally:
        con.close()
    
    if validation_passed:
        log.info("\n" + "="*70)
        log.info("✅ ALL VALIDATIONS PASSED")
        log.info("="*70)
    else:
        log.error("\n" + "="*70)
        log.error("❌ VALIDATION FAILURES DETECTED")
        log.error("="*70)
    
    return validation_passed


def print_summary(additive_data: pd.DataFrame, rate_data: pd.DataFrame,
                  itl3_data: pd.DataFrame, itl2_data: pd.DataFrame, 
                  itl1_data: pd.DataFrame):
    """Print final summary of aggregation"""
    
    log.info("\n" + "="*70)
    log.info("AGGREGATION SUMMARY")
    log.info("="*70)
    
    # Combine for LAD stats
    lad_combined = pd.concat([additive_data, rate_data], ignore_index=True) if not rate_data.empty else additive_data
    
    # Overall stats
    summary_data = []
    for level, df, label in [
        ('LAD', lad_combined, 'LAD (Input)'),
        ('ITL3', itl3_data, 'ITL3'),
        ('ITL2', itl2_data, 'ITL2'),
        ('ITL1', itl1_data, 'ITL1')
    ]:
        if not df.empty:
            summary_data.append({
                'Level': label,
                'Regions': df['region_code'].nunique(),
                'Metrics': df['metric_id'].nunique(),
                'Years': f"{df['period'].min()}-{df['period'].max()}",
                'Observations': len(df)
            })
    
    if summary_data:
        summary = pd.DataFrame(summary_data)
        log.info("\n" + summary.to_string(index=False))
    
    # Metric-level detail
    log.info("\n" + "="*70)
    log.info("OBSERVATIONS BY METRIC")
    log.info("="*70)
    
    for level_name, df in [('ITL3', itl3_data), ('ITL2', itl2_data), ('ITL1', itl1_data)]:
        if not df.empty:
            log.info(f"\n{level_name}:")
            metric_counts = df.groupby('metric_id').size()
            for metric_id, count in metric_counts.items():
                display_name = next(
                    (v['display_name'] for k, v in METRICS.items() if v['metric_id'] == metric_id),
                    metric_id
                )
                log.info(f"  {display_name}: {count:,} observations")


# -----------------------------
# Main Pipeline
# -----------------------------

def main():
    # Initialize pipeline reporter
    reporter = PipelineReporter("lad_to_itl_aggregation")
    
    log.info("="*70)
    log.info("UNIFIED LAD TO ITL AGGREGATION v1.2 (LABOUR MARKET RATES)")
    log.info("="*70)
    log.info(f"Vintage: {VINTAGE}")
    log.info("Strategy: Bottom-up aggregation from LAD → ITL3 → ITL2 → ITL1")
    log.info("Additive metrics: Population, Population 16-64, Employment, GDHI, GVA (SUM)")
    log.info("Rate metrics: Employment Rate, Unemployment Rate (population-weighted avg)")
    log.info("Derived metrics: GDHI per head, Productivity")
    log.info("="*70)
    
    # Load all LAD data
    log.info("\n[1/6] Loading LAD data...")
    additive_data, rate_data = load_all_lad_data(reporter)
    
    if additive_data.empty:
        reporter.add_critical_error("No additive LAD data loaded")
        reporter.save_and_exit("lad_aggregation_summary.json")
    
    # Check for required parent geography columns
    required_cols = ['itl3_code', 'itl3_name', 'itl2_code', 'itl2_name', 
                     'itl1_code', 'itl1_name']
    missing_cols = [c for c in required_cols if c not in additive_data.columns]
    
    if missing_cols:
        reporter.add_critical_error(f"LAD data missing parent geography columns: {missing_cols}")
        reporter.save_and_exit("lad_aggregation_summary.json")
    
    # Aggregate to ITL3
    log.info("\n[2/6] Aggregating to ITL3...")
    itl3_data = aggregate_to_level(
        additive_data, rate_data,
        level_code_col='itl3_code',
        level_name_col='itl3_name',
        region_level='ITL3',
        reporter=reporter
    )
    
    if not itl3_data.empty:
        # Calculate derived metrics for historical data
        log.info("  Calculating derived metrics...")
        itl3_data = calculate_historical_derived_metrics(itl3_data, reporter)
        
        itl3_csv = SILVER_DIR / "itl3_unified_history.csv"
        itl3_data.to_csv(itl3_csv, index=False)
        log.info(f"  ✓ Saved CSV → {itl3_csv}")
        
        # Write to DuckDB with CANONICAL name: itl3_history
        write_to_duckdb(itl3_data, "itl3_history", reporter)
    else:
        reporter.add_critical_error("ITL3 aggregation produced no data")
    
    # Aggregate to ITL2
    log.info("\n[3/6] Aggregating to ITL2...")
    itl2_data = aggregate_to_level(
        additive_data, rate_data,
        level_code_col='itl2_code',
        level_name_col='itl2_name',
        region_level='ITL2',
        reporter=reporter
    )
    
    if not itl2_data.empty:
        # Calculate derived metrics for historical data
        log.info("  Calculating derived metrics...")
        itl2_data = calculate_historical_derived_metrics(itl2_data, reporter)
        
        itl2_csv = SILVER_DIR / "itl2_unified_history.csv"
        itl2_data.to_csv(itl2_csv, index=False)
        log.info(f"  ✓ Saved CSV → {itl2_csv}")
        
        # Write to DuckDB with CANONICAL name: itl2_history
        write_to_duckdb(itl2_data, "itl2_history", reporter)
    else:
        reporter.add_critical_error("ITL2 aggregation produced no data")
    
    # Aggregate to ITL1 (bottom-up)
    log.info("\n[4/6] Aggregating to ITL1...")
    itl1_data = aggregate_to_level(
        additive_data, rate_data,
        level_code_col='itl1_code',
        level_name_col='itl1_name',
        region_level='ITL1',
        reporter=reporter
    )
    
    if not itl1_data.empty:
        # Calculate derived metrics for historical data
        log.info("  Calculating derived metrics...")
        itl1_data = calculate_historical_derived_metrics(itl1_data, reporter)
        
        itl1_csv = SILVER_DIR / "itl1_unified_bottomup.csv"
        itl1_data.to_csv(itl1_csv, index=False)
        log.info(f"  ✓ Saved CSV → {itl1_csv}")
        
        # Keep special name for comparison table
        write_to_duckdb(itl1_data, "itl1_unified_bottomup", reporter)
        
        # Compare with top-down
        log.info("\n[5/6] Comparing with top-down ITL1...")
        compare_with_topdown(itl1_data, reporter)
    else:
        reporter.add_critical_error("ITL1 aggregation produced no data")
    
    # Validate DuckDB tables - HARD FAIL if validation fails
    log.info("\n[6/6] Validating DuckDB tables...")
    expected_counts = {}
    if not itl3_data.empty:
        expected_counts['itl3_history'] = len(itl3_data)
    if not itl2_data.empty:
        expected_counts['itl2_history'] = len(itl2_data)
    if not itl1_data.empty:
        expected_counts['itl1_unified_bottomup'] = len(itl1_data)
    
    validation_passed = validate_duckdb_tables(expected_counts, reporter)
    
    if not validation_passed:
        reporter.add_critical_error("Validation failed - silver tables have issues")
        log.error("\n" + "="*70)
        log.error("❌ AGGREGATION FAILED - VALIDATION ERRORS")
        log.error("="*70)
        log.error("Silver tables NOT updated properly - fix data issues and re-run")
        log.error("DO NOT proceed to forecasting until validation passes")
        reporter.save_and_exit("lad_aggregation_summary.json")
    
    # Print summary
    print_summary(additive_data, rate_data, itl3_data, itl2_data, itl1_data)
    
    # Final status
    log.info("\n" + "="*70)
    log.info("✅ UNIFIED BOTTOM-UP AGGREGATION COMPLETE")
    log.info("="*70)
    
    log.info("\nDuckDB Tables Created:")
    log.info("  - silver.lad_history (unified LAD data)")
    log.info("  - silver.itl3_history (canonical: 5 base + 2 derived + rates)")
    log.info("  - silver.itl2_history (canonical: 5 base + 2 derived + rates)")
    log.info("  - silver.itl1_unified_bottomup (validation only)")
    
    log.info("\nCSV Files Created:")
    log.info("  - data/silver/itl3_unified_history.csv")
    log.info("  - data/silver/itl2_unified_history.csv")
    log.info("  - data/silver/itl1_unified_bottomup.csv")
    
    log.info("\nMetric Aggregation Methods:")
    log.info("  - Additive (SUM): population_total, population_16_64, emp_total_jobs, gdhi_total_mn_gbp, nominal_gva_mn_gbp")
    log.info("  - Rates (pop-weighted avg): employment_rate_pct, unemployment_rate_pct")
    log.info("  - Derived (calculated): gdhi_per_head_gbp, productivity_gbp_per_job")
    
    log.info("\nNext steps:")
    log.info("  → Validation passed - silver tables are clean")
    log.info("  → Forecast scripts will read silver.itl2_history and silver.itl3_history")
    log.info("  → Historical + forecast data will have consistent derived metrics")
    log.info("="*70)
    
    # Save summary and exit
    reporter.save_and_exit("lad_aggregation_summary.json")


if __name__ == "__main__":
    main()