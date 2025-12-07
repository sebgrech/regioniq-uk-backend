#!/usr/bin/env python3
"""
Unified LAD to ITL Aggregation Script (Production v1.0)

Takes LAD-level data for ALL metrics and aggregates up to ITL3, ITL2, ITL1 levels.
Creates consistent bottom-up hierarchy for forecasting.

Inputs:
- silver.lad_population_history
- silver.lad_employment_history
- silver.lad_gdhi_history
- silver.lad_gva_history

Outputs:
- silver.itl3_unified_history (bottom-up)
- silver.itl2_unified_history (bottom-up)
- silver.itl1_unified_bottomup (bottom-up)
- Comparison reports vs top-down ITL1 if available
"""

import logging
import pandas as pd
import numpy as np
import sys
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

VINTAGE = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("unified_aggregation")

# Metric configurations
METRICS = {
    'population': {
        'table': 'lad_population_history',
        'metric_id': 'population_total',
        'unit': 'persons',
        'display_name': 'Population'
    },
    'employment': {
        'table': 'lad_employment_history',
        'metric_id': 'emp_total_jobs',
        'unit': 'jobs',
        'display_name': 'Employment'
    },
    'gdhi': {
        'table': 'lad_gdhi_history',
        'metric_id': 'gdhi_total_mn_gbp',
        'unit': 'GBP_m',
        'display_name': 'GDHI Total'
    },
    'gva': {
        'table': 'lad_gva_history',
        'metric_id': 'nominal_gva_mn_gbp',
        'unit': 'GBP_m',
        'display_name': 'GVA'
    }
}

# -----------------------------
# Helper Functions
# -----------------------------

def load_lad_metric(metric_key: str) -> pd.DataFrame:
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
            log.warning(f"  DuckDB load failed: {e}, trying CSV...")
        finally:
            con.close()
    
    # Fallback to CSV
    csv_path = SILVER_DIR / f"{table_name}.csv"
    if not csv_path.exists():
        log.warning(f"  ⚠️  LAD data not found: {csv_path}")
        return pd.DataFrame()
    
    log.info(f"Loading {config['display_name']} from CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    log.info(f"  Loaded {len(df)} observations from CSV")
    return df


def load_all_lad_data() -> pd.DataFrame:
    """Load and combine all LAD metrics into one dataframe"""
    all_data = []
    
    for metric_key in METRICS.keys():
        df = load_lad_metric(metric_key)
        if not df.empty:
            all_data.append(df)
    
    if not all_data:
        raise RuntimeError("No LAD data found for any metric!")
    
    combined = pd.concat(all_data, ignore_index=True)
    log.info(f"\n✓ Combined LAD data: {len(combined)} total observations")
    log.info(f"  Metrics: {combined['metric_id'].unique().tolist()}")
    log.info(f"  LADs: {combined['region_code'].nunique()}")
    log.info(f"  Year range: {combined['period'].min()} - {combined['period'].max()}")
    
    return combined


def calculate_historical_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
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
        log.info("  ⚠️  No derived metrics calculated (missing base metrics)")
        return df
    
    derived_df = pd.DataFrame(derived_rows)
    
    # Count by metric type
    derived_counts = derived_df['metric_id'].value_counts()
    for metric_id, count in derived_counts.items():
        log.info(f"  ✓ Calculated {count:,} {metric_id} observations")
    
    # Combine with original data
    result = pd.concat([df, derived_df], ignore_index=True)
    result = result.sort_values(['metric_id', 'region_code', 'period']).reset_index(drop=True)
    
    return result


def aggregate_to_level(
    df: pd.DataFrame, 
    level_code_col: str,
    level_name_col: str,
    region_level: str
) -> pd.DataFrame:
    """
    Aggregate LAD data to a higher geography level for ALL metrics.
    
    Args:
        df: LAD data with parent geography columns and multiple metrics
        level_code_col: Column name for target level code (e.g., 'itl3_code')
        level_name_col: Column name for target level name (e.g., 'itl3_name')
        region_level: String identifier for output (e.g., 'ITL3')
    
    Returns:
        Aggregated dataframe in tidy schema with all metrics
    """
    # Check required columns
    required = [level_code_col, level_name_col, 'period', 'value', 'metric_id']
    missing = [c for c in required if c not in df.columns]
    if missing:
        log.error(f"Missing columns for {region_level} aggregation: {missing}")
        return pd.DataFrame()
    
    # Remove rows with null parent geography codes
    df_clean = df.dropna(subset=[level_code_col, level_name_col]).copy()
    
    if len(df_clean) < len(df):
        log.warning(f"  Dropped {len(df) - len(df_clean)} rows with missing {level_code_col}")
    
    # Aggregate by geography, period, and metric
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
    
    log.info(f"  {region_level}: {agg['region_code'].nunique()} regions × {agg['metric_id'].nunique()} metrics = {len(agg)} observations")
    
    return agg


def write_to_duckdb(df: pd.DataFrame, table_name: str):
    """Write aggregated data to DuckDB silver schema"""
    if not HAVE_DUCKDB:
        log.warning("  duckdb not installed; skipping DuckDB write")
        return
    
    con = duckdb.connect(str(DUCK_PATH))
    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS silver")
        con.register("df_tmp", df)
        con.execute(f"CREATE OR REPLACE TABLE silver.{table_name} AS SELECT * FROM df_tmp")
        log.info(f"  ✓ Wrote {len(df)} rows to silver.{table_name}")
    finally:
        con.close()


def compare_with_topdown(bottomup_itl1: pd.DataFrame):
    """Compare bottom-up ITL1 with existing top-down ITL1 for all metrics"""
    
    # Try to load top-down data
    topdown = None
    
    if HAVE_DUCKDB and DUCK_PATH.exists():
        try:
            con = duckdb.connect(str(DUCK_PATH), read_only=True)
            topdown = con.execute("""
                SELECT region_code, period, metric_id, value 
                FROM silver.itl1_history 
                WHERE metric_id IN ('population_total', 'emp_total_jobs', 'gdhi_total_mn_gbp', 'nominal_gva_mn_gbp')
            """).fetchdf()
            con.close()
        except:
            pass
    
    if topdown is None or topdown.empty:
        # Try CSV fallback
        csv_path = SILVER_DIR / "itl1_unified_history.csv"
        if csv_path.exists():
            topdown_all = pd.read_csv(csv_path)
            topdown = topdown_all[topdown_all['metric_id'].isin([
                'population_total', 'emp_total_jobs', 'gdhi_total_mn_gbp', 'nominal_gva_mn_gbp'
            ])].copy()
    
    if topdown is None or topdown.empty:
        log.info("\n  No top-down ITL1 data available for comparison")
        return
    
    log.info("\n" + "="*70)
    log.info("BOTTOM-UP vs TOP-DOWN COMPARISON")
    log.info("="*70)
    log.info(f"Top-down data: {len(topdown)} observations")
    
    # Merge on region_code, period, and metric_id
    comparison = bottomup_itl1.merge(
        topdown[['region_code', 'period', 'metric_id', 'value']],
        on=['region_code', 'period', 'metric_id'],
        how='inner',
        suffixes=('_bottomup', '_topdown')
    )
    
    if comparison.empty:
        log.warning("No overlapping data between bottom-up and top-down")
        return
    
    # Calculate differences
    comparison['diff'] = comparison['value_bottomup'] - comparison['value_topdown']
    comparison['diff_pct'] = (comparison['diff'] / comparison['value_topdown'] * 100).abs()
    
    # Overall summary
    log.info(f"\nMatched {len(comparison)} observations across all metrics")
    log.info(f"  Average absolute difference: {comparison['diff_pct'].mean():.3f}%")
    log.info(f"  Median absolute difference: {comparison['diff_pct'].median():.3f}%")
    log.info(f"  Maximum absolute difference: {comparison['diff_pct'].max():.3f}%")
    
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
        
        # Show worst cases if any
        if n_large > 0:
            worst = metric_comp.nlargest(3, 'diff_pct')
            for _, row in worst.iterrows():
                log.info(
                    f"     {row['region_code']} ({row['period']}): "
                    f"{row['diff_pct']:.2f}% diff"
                )


def validate_duckdb_tables(expected_counts: Dict[str, int]) -> bool:
    """
    Validate that DuckDB silver tables were created correctly.
    
    Args:
        expected_counts: Dict mapping table names to expected row counts
    
    Returns:
        True if all validations pass, False otherwise
    """
    if not HAVE_DUCKDB or not DUCK_PATH.exists():
        log.warning("DuckDB not available - skipping validation")
        return True
    
    log.info("\n" + "="*70)
    log.info("DUCKDB TABLE VALIDATION")
    log.info("="*70)
    
    con = duckdb.connect(str(DUCK_PATH), read_only=True)
    validation_passed = True
    
    try:
        # Expected metrics: 4 base + 2 derived
        expected_base_metrics = set([
            'population_total', 
            'emp_total_jobs', 
            'gdhi_total_mn_gbp', 
            'nominal_gva_mn_gbp'
        ])
        
        expected_derived_metrics = set([
            'gdhi_per_head_gbp',
            'productivity_gbp_per_job'
        ])
        
        expected_all_metrics = expected_base_metrics | expected_derived_metrics
        
        for table_name, expected_count in expected_counts.items():
            log.info(f"\n✓ Validating silver.{table_name}...")
            
            # Check table exists
            try:
                result = con.execute(f"SELECT COUNT(*) FROM silver.{table_name}").fetchone()
                actual_count = result[0]
            except:
                log.error(f"  ❌ Table does not exist!")
                validation_passed = False
                continue
            
            # Check row count
            if actual_count != expected_count:
                log.error(f"  ❌ Row count mismatch: expected {expected_count}, got {actual_count}")
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
                log.error(f"  ❌ Missing base metrics: {missing_base}")
                validation_passed = False
            else:
                log.info(f"  ✓ All 4 base metrics present")
            
            # Check derived metrics (should be present if base metrics are)
            missing_derived = expected_derived_metrics - actual_metrics
            if missing_derived:
                log.warning(f"  ⚠️  Missing derived metrics: {missing_derived}")
                # Not a hard fail, but warn
            else:
                log.info(f"  ✓ All 2 derived metrics present")
            
            log.info(f"  → Total metrics: {len(actual_metrics)}/6")
            
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
                log.error(f"  ❌ Found {total_nulls} NULL values in critical columns")
                log.error(f"     region_code: {null_check[0]}, metric_id: {null_check[1]}, "
                         f"period: {null_check[2]}, value: {null_check[3]}")
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


def print_summary(lad_data: pd.DataFrame, itl3_data: pd.DataFrame, 
                  itl2_data: pd.DataFrame, itl1_data: pd.DataFrame):
    """Print final summary of aggregation"""
    
    log.info("\n" + "="*70)
    log.info("AGGREGATION SUMMARY")
    log.info("="*70)
    
    # Overall stats
    summary_data = []
    for level, df, label in [
        ('LAD', lad_data, 'LAD (Input)'),
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
    log.info("="*70)
    log.info("UNIFIED LAD TO ITL AGGREGATION v1.0")
    log.info("="*70)
    log.info(f"Vintage: {VINTAGE}")
    log.info("Strategy: Bottom-up aggregation from LAD → ITL3 → ITL2 → ITL1")
    log.info("Metrics: Population, Employment, GDHI, GVA + derived ratios")
    log.info("="*70)
    
    # Load all LAD data
    log.info("\n[1/6] Loading LAD data...")
    lad_data = load_all_lad_data()
    
    # Check for required parent geography columns
    required_cols = ['itl3_code', 'itl3_name', 'itl2_code', 'itl2_name', 
                     'itl1_code', 'itl1_name']
    missing_cols = [c for c in required_cols if c not in lad_data.columns]
    
    if missing_cols:
        log.error(f"LAD data missing parent geography columns: {missing_cols}")
        log.error("Please ensure all LAD ingest scripts include the lookup file")
        return
    
    # Aggregate to ITL3
    log.info("\n[2/6] Aggregating to ITL3...")
    itl3_data = aggregate_to_level(
        lad_data,
        level_code_col='itl3_code',
        level_name_col='itl3_name',
        region_level='ITL3'
    )
    
    if not itl3_data.empty:
        # Calculate derived metrics for historical data
        log.info("  Calculating derived metrics...")
        itl3_data = calculate_historical_derived_metrics(itl3_data)
        
        itl3_csv = SILVER_DIR / "itl3_unified_history.csv"
        itl3_data.to_csv(itl3_csv, index=False)
        log.info(f"  ✓ Saved CSV → {itl3_csv}")
        
        # Write to DuckDB with CANONICAL name: itl3_history (not itl3_unified_history)
        write_to_duckdb(itl3_data, "itl3_history")
    
    # Aggregate to ITL2
    log.info("\n[3/6] Aggregating to ITL2...")
    itl2_data = aggregate_to_level(
        lad_data,
        level_code_col='itl2_code',
        level_name_col='itl2_name',
        region_level='ITL2'
    )
    
    if not itl2_data.empty:
        # Calculate derived metrics for historical data
        log.info("  Calculating derived metrics...")
        itl2_data = calculate_historical_derived_metrics(itl2_data)
        
        itl2_csv = SILVER_DIR / "itl2_unified_history.csv"
        itl2_data.to_csv(itl2_csv, index=False)
        log.info(f"  ✓ Saved CSV → {itl2_csv}")
        
        # Write to DuckDB with CANONICAL name: itl2_history (not itl2_unified_history)
        write_to_duckdb(itl2_data, "itl2_history")
    
    # Aggregate to ITL1 (bottom-up)
    log.info("\n[4/6] Aggregating to ITL1...")
    itl1_data = aggregate_to_level(
        lad_data,
        level_code_col='itl1_code',
        level_name_col='itl1_name',
        region_level='ITL1'
    )
    
    if not itl1_data.empty:
        # Calculate derived metrics for historical data
        log.info("  Calculating derived metrics...")
        itl1_data = calculate_historical_derived_metrics(itl1_data)
        
        itl1_csv = SILVER_DIR / "itl1_unified_bottomup.csv"
        itl1_data.to_csv(itl1_csv, index=False)
        log.info(f"  ✓ Saved CSV → {itl1_csv}")
        
        # Keep special name for comparison table
        write_to_duckdb(itl1_data, "itl1_unified_bottomup")
        
        # Compare with top-down
        log.info("\n[5/6] Comparing with top-down ITL1...")
        compare_with_topdown(itl1_data)
    
    # Validate DuckDB tables - HARD FAIL if validation fails
    log.info("\n[6/6] Validating DuckDB tables...")
    expected_counts = {}
    if not itl3_data.empty:
        expected_counts['itl3_history'] = len(itl3_data)
    if not itl2_data.empty:
        expected_counts['itl2_history'] = len(itl2_data)
    if not itl1_data.empty:
        expected_counts['itl1_unified_bottomup'] = len(itl1_data)
    
    validation_passed = validate_duckdb_tables(expected_counts)
    
    if not validation_passed:
        log.error("\n" + "="*70)
        log.error("❌ AGGREGATION FAILED - VALIDATION ERRORS")
        log.error("="*70)
        log.error("Silver tables NOT updated properly - fix data issues and re-run")
        log.error("DO NOT proceed to forecasting until validation passes")
        sys.exit(1)  # Non-zero exit = pipeline stops
    
    # Print summary
    print_summary(lad_data, itl3_data, itl2_data, itl1_data)
    
    # Final status
    log.info("\n" + "="*70)
    log.info("✅ UNIFIED BOTTOM-UP AGGREGATION COMPLETE")
    log.info("="*70)
    
    log.info("\nDuckDB Tables Created:")
    log.info("  - silver.itl3_history (canonical: 4 base + 2 derived metrics)")
    log.info("  - silver.itl2_history (canonical: 4 base + 2 derived metrics)")
    log.info("  - silver.itl1_unified_bottomup (validation only)")
    
    log.info("\nCSV Files Created:")
    log.info("  - data/silver/itl3_unified_history.csv")
    log.info("  - data/silver/itl2_unified_history.csv")
    log.info("  - data/silver/itl1_unified_bottomup.csv")
    
    log.info("\nDerived Metrics Calculated:")
    log.info("  - gdhi_per_head_gbp = (gdhi_total_mn_gbp * 1e6) / population_total")
    log.info("  - productivity_gbp_per_job = (nominal_gva_mn_gbp * 1e6) / emp_total_jobs")
    
    log.info("\nNext steps:")
    log.info("  → Validation passed - silver tables are clean")
    log.info("  → Forecast scripts will read silver.itl2_history and silver.itl3_history")
    log.info("  → Historical + forecast data will have consistent derived metrics")
    log.info("="*70)


if __name__ == "__main__":
    main()