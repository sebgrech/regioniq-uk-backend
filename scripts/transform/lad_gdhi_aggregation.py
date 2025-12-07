#!/usr/bin/env python3
"""
LAD GDHI Aggregation Script (Production v1.0)

Takes LAD-level GDHI (total £m) data and aggregates up to ITL3, ITL2, ITL1 levels.
Creates consistent bottom-up hierarchy for GDHI forecasting.

IMPORTANT: This aggregates GDHI TOTAL only.
GDHI per head is derived AFTER reconciliation as: gdhi_total / population

Inputs:
- silver.lad_gdhi_history (from lad_gdhi_ingest.py)

Outputs:
- silver.itl3_gdhi_history
- silver.itl2_gdhi_history  
- silver.itl1_gdhi_history (bottom-up)
- Comparison report vs top-down ITL1 if available
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

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
log = logging.getLogger("gdhi_aggregation")

# -----------------------------
# Helper Functions
# -----------------------------

def load_lad_data() -> pd.DataFrame:
    """Load LAD GDHI data from DuckDB or CSV"""
    if HAVE_DUCKDB and DUCK_PATH.exists():
        log.info("Loading LAD GDHI from DuckDB...")
        con = duckdb.connect(str(DUCK_PATH), read_only=True)
        try:
            df = con.execute("SELECT * FROM silver.lad_gdhi_history").fetchdf()
            log.info(f"Loaded {len(df)} LAD observations from DuckDB")
            return df
        finally:
            con.close()
    else:
        # Fallback to CSV
        csv_path = SILVER_DIR / "lad_gdhi_history.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"LAD GDHI data not found: {csv_path}")
        
        log.info(f"Loading LAD GDHI from CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        log.info(f"Loaded {len(df)} LAD observations from CSV")
        return df


def aggregate_to_level(
    df: pd.DataFrame, 
    level_code_col: str,
    level_name_col: str,
    region_level: str
) -> pd.DataFrame:
    """
    Aggregate LAD GDHI to a higher geography level.
    
    Args:
        df: LAD GDHI data with parent geography columns
        level_code_col: Column name for target level code (e.g., 'itl3_code')
        level_name_col: Column name for target level name (e.g., 'itl3_name')
        region_level: String identifier for output (e.g., 'ITL3')
    
    Returns:
        Aggregated dataframe in tidy schema
    """
    # Check required columns
    required = [level_code_col, level_name_col, 'period', 'value']
    missing = [c for c in required if c not in df.columns]
    if missing:
        log.error(f"Missing columns for {region_level} aggregation: {missing}")
        return pd.DataFrame()
    
    # Remove rows with null parent geography codes
    df_clean = df.dropna(subset=[level_code_col, level_name_col]).copy()
    
    if len(df_clean) < len(df):
        log.warning(f"Dropped {len(df) - len(df_clean)} rows with missing {level_code_col}")
    
    # Aggregate by geography and period
    agg = df_clean.groupby([level_code_col, level_name_col, 'period'], as_index=False).agg({
        'value': 'sum'
    })
    
    # Rename to standard schema
    agg = agg.rename(columns={
        level_code_col: 'region_code',
        level_name_col: 'region_name'
    })
    
    # Add metadata
    agg['region_level'] = region_level
    agg['metric_id'] = 'gdhi_total_mn_gbp'
    agg['unit'] = 'GBP_m'
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
    agg = agg.sort_values(['region_code', 'period']).reset_index(drop=True)
    
    log.info(f"{region_level}: Aggregated to {agg['region_code'].nunique()} regions, {len(agg)} observations")
    
    return agg


def write_to_duckdb(df: pd.DataFrame, table_name: str):
    """Write aggregated data to DuckDB silver schema"""
    if not HAVE_DUCKDB:
        log.warning("duckdb not installed; skipping DuckDB write")
        return
    
    con = duckdb.connect(str(DUCK_PATH))
    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS silver")
        con.register("df_tmp", df)
        con.execute(f"CREATE OR REPLACE TABLE silver.{table_name} AS SELECT * FROM df_tmp")
        log.info(f"✓ Wrote {len(df)} rows to silver.{table_name}")
    finally:
        con.close()


def compare_with_topdown(bottomup_itl1: pd.DataFrame):
    """Compare bottom-up ITL1 with existing top-down ITL1 if available"""
    # Try DuckDB first
    if HAVE_DUCKDB and DUCK_PATH.exists():
        try:
            log.info("\n" + "="*70)
            log.info("BOTTOM-UP vs TOP-DOWN COMPARISON")
            log.info("="*70)
            
            con = duckdb.connect(str(DUCK_PATH), read_only=True)
            topdown = con.execute("""
                SELECT region_code, period, value 
                FROM silver.itl1_history 
                WHERE metric_id = 'gdhi_total_mn_gbp'
            """).fetchdf()
            con.close()
            
        except:
            # Fallback to CSV
            csv_path = SILVER_DIR / "itl1_unified_history.csv"
            if not csv_path.exists():
                log.info("No top-down ITL1 data found for comparison")
                return
            
            topdown_all = pd.read_csv(csv_path)
            topdown = topdown_all[topdown_all['metric_id'] == 'gdhi_total_mn_gbp'].copy()
    else:
        # CSV only
        csv_path = SILVER_DIR / "itl1_unified_history.csv"
        if not csv_path.exists():
            log.info("No top-down ITL1 data available for comparison")
            return
        
        topdown_all = pd.read_csv(csv_path)
        topdown = topdown_all[topdown_all['metric_id'] == 'gdhi_total_mn_gbp'].copy()
    
    if topdown.empty:
        log.info("No GDHI data in top-down ITL1 for comparison")
        return
    
    log.info(f"Loaded {len(topdown)} top-down ITL1 GDHI observations")
    
    # Merge on region_code and period
    comparison = bottomup_itl1.merge(
        topdown[['region_code', 'period', 'value']],
        on=['region_code', 'period'],
        how='inner',
        suffixes=('_bottomup', '_topdown')
    )
    
    if comparison.empty:
        log.warning("No overlapping periods between bottom-up and top-down")
        return
    
    # Calculate differences
    comparison['diff'] = comparison['value_bottomup'] - comparison['value_topdown']
    comparison['diff_pct'] = (comparison['diff'] / comparison['value_topdown'] * 100).abs()
    
    # Summary statistics
    log.info(f"\nComparison for {len(comparison)} matched observations:")
    log.info(f"  Average absolute difference: {comparison['diff_pct'].mean():.3f}%")
    log.info(f"  Median absolute difference: {comparison['diff_pct'].median():.3f}%")
    log.info(f"  Maximum absolute difference: {comparison['diff_pct'].max():.3f}%")
    log.info(f"  Min absolute difference: {comparison['diff_pct'].min():.3f}%")
    
    # Flag regions with differences >1%
    large_diff = comparison[comparison['diff_pct'] > 1.0]
    if not large_diff.empty:
        log.warning(f"\n⚠️  {len(large_diff)} observations with >1% difference:")
        
        worst = large_diff.nlargest(5, 'diff_pct')
        for _, row in worst.iterrows():
            log.warning(
                f"  {row['region_code']} ({row['period']}): "
                f"{row['diff_pct']:.2f}% diff "
                f"(BU: £{row['value_bottomup']:,.0f}m, TD: £{row['value_topdown']:,.0f}m)"
            )
    else:
        log.info("\n✅ EXCELLENT: All differences <1%")
    
    # By region summary
    log.info("\n" + "="*70)
    log.info("ACCURACY BY REGION")
    log.info("="*70)
    
    region_summary = comparison.groupby('region_code').agg({
        'diff_pct': ['mean', 'max'],
        'period': 'count'
    }).round(3)
    region_summary.columns = ['Avg_Diff_%', 'Max_Diff_%', 'Years']
    
    for region_code, row in region_summary.sort_values('Max_Diff_%', ascending=False).iterrows():
        status = "✓" if row['Max_Diff_%'] < 1.0 else "⚠️ "
        log.info(f"{status} {region_code}: Avg {row['Avg_Diff_%']:.3f}%, Max {row['Max_Diff_%']:.3f}%, {int(row['Years'])} years")


# -----------------------------
# Main Pipeline
# -----------------------------

def main():
    log.info("="*70)
    log.info("LAD GDHI TO ITL AGGREGATION v1.0")
    log.info("="*70)
    log.info(f"Vintage: {VINTAGE}")
    log.info("Strategy: Bottom-up aggregation from LAD → ITL3 → ITL2 → ITL1")
    log.info("Metric: GDHI total (£m) only - per head derived later")
    
    # Load LAD data
    lad_data = load_lad_data()
    
    # Check for required parent geography columns
    required_cols = ['itl3_code', 'itl3_name', 'itl2_code', 'itl2_name', 
                     'itl1_code', 'itl1_name']
    missing_cols = [c for c in required_cols if c not in lad_data.columns]
    
    if missing_cols:
        log.error(f"LAD data missing parent geography columns: {missing_cols}")
        log.error("Please re-run lad_gdhi_ingest.py with the lookup file")
        return
    
    # Verify metric
    if 'metric_id' in lad_data.columns:
        metrics = lad_data['metric_id'].unique()
        if 'gdhi_total_mn_gbp' not in metrics:
            log.error(f"Expected gdhi_total_mn_gbp metric, found: {metrics}")
            return
    
    log.info(f"\nInput: {len(lad_data)} LAD GDHI observations")
    log.info(f"Unique LADs: {lad_data['region_code'].nunique()}")
    log.info(f"Year range: {lad_data['period'].min()} - {lad_data['period'].max()}")
    
    # Aggregate to ITL3
    log.info("\n--- Aggregating to ITL3 ---")
    itl3_data = aggregate_to_level(
        lad_data,
        level_code_col='itl3_code',
        level_name_col='itl3_name',
        region_level='ITL3'
    )
    
    if not itl3_data.empty:
        itl3_csv = SILVER_DIR / "itl3_gdhi_history.csv"
        itl3_data.to_csv(itl3_csv, index=False)
        log.info(f"✓ Saved CSV → {itl3_csv}")
        
        write_to_duckdb(itl3_data, "itl3_gdhi_history")
    
    # Aggregate to ITL2
    log.info("\n--- Aggregating to ITL2 ---")
    itl2_data = aggregate_to_level(
        lad_data,
        level_code_col='itl2_code',
        level_name_col='itl2_name',
        region_level='ITL2'
    )
    
    if not itl2_data.empty:
        itl2_csv = SILVER_DIR / "itl2_gdhi_history.csv"
        itl2_data.to_csv(itl2_csv, index=False)
        log.info(f"✓ Saved CSV → {itl2_csv}")
        
        write_to_duckdb(itl2_data, "itl2_gdhi_history")
    
    # Aggregate to ITL1 (bottom-up)
    log.info("\n--- Aggregating to ITL1 ---")
    itl1_data = aggregate_to_level(
        lad_data,
        level_code_col='itl1_code',
        level_name_col='itl1_name',
        region_level='ITL1'
    )
    
    if not itl1_data.empty:
        itl1_csv = SILVER_DIR / "itl1_gdhi_bottomup.csv"
        itl1_data.to_csv(itl1_csv, index=False)
        log.info(f"✓ Saved CSV → {itl1_csv}")
        
        write_to_duckdb(itl1_data, "itl1_gdhi_bottomup")
        
        # Compare with top-down
        compare_with_topdown(itl1_data)
    
    # Summary report
    log.info("\n" + "="*70)
    log.info("AGGREGATION SUMMARY")
    log.info("="*70)
    
    summary_data = []
    for level, df in [('LAD', lad_data), ('ITL3', itl3_data), ('ITL2', itl2_data), ('ITL1', itl1_data)]:
        if not df.empty:
            summary_data.append({
                'Level': level,
                'Regions': df['region_code'].nunique(),
                'Years': f"{df['period'].min()}-{df['period'].max()}",
                'Observations': len(df),
                'Avg per region': f"{len(df) / df['region_code'].nunique():.1f}"
            })
    
    if summary_data:
        summary = pd.DataFrame(summary_data)
        log.info("\n" + summary.to_string(index=False))
    
    log.info("\n✅ Bottom-up GDHI aggregation complete!")
    log.info("="*70)


if __name__ == "__main__":
    main()