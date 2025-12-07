#!/usr/bin/env python3
"""
LAD GDHI Ingest from NOMIS API → DuckDB + CSV (Production v1.0)

Fetches Local Authority District (LAD) level GDHI (total £m) data from NOMIS.
Handles ALL boundary changes 2019-2023 with complete concordance mapping.

IMPORTANT: Only ingests GDHI total (£m), NOT per head.
Per head will be derived AFTER reconciliation as: gdhi_total / population

Dataset: NM_185_1 (Regional GDHI)
- measure=1: GDHI total (£ million)
- component_of_gdhi=0: Total (not broken down by component)

Boundary Changes Handled:
- Cumbria (Apr 2023): 6 districts → 2 unitaries
- Northamptonshire (Apr 2021): 7 districts → 2 unitaries
- Buckinghamshire (Apr 2020): 4 districts → 1 unitary
- North Yorkshire (Apr 2023): 7 districts → 1 unitary
- Somerset (Apr 2023): 4 districts → 1 unitary

Outputs:
- data/raw/gdhi/lad_gdhi_nomis.csv (raw)
- data/silver/lad_gdhi_history.csv (tidy, 2023 boundaries)
- warehouse.duckdb:
    bronze.gdhi_lad_raw
    silver.lad_gdhi_history
"""

import io
import logging
import pandas as pd
import requests
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
RAW_DIR = Path("data/raw/gdhi")
RAW_DIR.mkdir(parents=True, exist_ok=True)

SILVER_DIR = Path("data/silver")
SILVER_DIR.mkdir(parents=True, exist_ok=True)
SILVER_CSV = SILVER_DIR / "lad_gdhi_history.csv"

LAKE_DIR = Path("data/lake")
LAKE_DIR.mkdir(parents=True, exist_ok=True)
DUCK_PATH = LAKE_DIR / "warehouse.duckdb"

LOOKUP_PATH = Path("data/reference/master_2025_geography_lookup.csv")

VINTAGE = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# NOMIS API URL for LAD GDHI (total £m only)
# measure=1: GDHI total (£m)
# measure=2: GDHI per head (£) - NOT INGESTED, will be derived later
NOMIS_URL = (
    "https://www.nomisweb.co.uk/api/v01/dataset/NM_185_1.data.csv"
    "?geography=1778384897...1778384901,1778384941,1778384950,1778385143...1778385146,"
    "1778385159,1778384902...1778384905,1778384942,1778384943,1778384956,1778384957,"
    "1778385033...1778385044,1778385124...1778385138,1778384906...1778384910,1778384958,"
    "1778385139...1778385142,1778385154...1778385158,1778384911...1778384914,1778384954,"
    "1778384955,1778384965...1778384972,1778385045...1778385058,1778385066...1778385072,"
    "1778384915...1778384917,1778384944,1778385078...1778385085,1778385100...1778385104,"
    "1778385112...1778385117,1778385147...1778385153,1778384925...1778384928,1778384948,"
    "1778384949,1778384960...1778384964,1778384986...1778384997,1778385015...1778385020,"
    "1778385059...1778385065,1778385086...1778385088,1778385118...1778385123,"
    "1778385160...1778385192,1778384929...1778384940,1778384953,1778384981...1778384985,"
    "1778385004...1778385014,1778385021...1778385032,1778385073...1778385077,"
    "1778385089...1778385099,1778385105...1778385111,1778384918...1778384924,"
    "1778384945...1778384947,1778384951,1778384952,1778384973...1778384980,"
    "1778384998...1778385003,1778384959,1778385193...1778385257"
    "&component_of_gdhi=0"
    "&measure=1"
    "&measures=20100"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("lad_gdhi_ingest")

# TLC to ONS code mapping
TLC_TO_ONS = {
    'TLC': 'E12000001',  'TLD': 'E12000002',  'TLE': 'E12000003',
    'TLF': 'E12000004',  'TLG': 'E12000005',  'TLH': 'E12000006',
    'TLI': 'E12000007',  'TLJ': 'E12000008',  'TLK': 'E12000009',
    'TLL': 'W92000004',  'TLM': 'S92000003',  'TLN': 'N92000002',
}

# COMPLETE LAD CONCORDANCE (same as employment/GVA)
LAD_CONCORDANCE = {
    # Cumbria (Apr 2023): 6 → 2
    'E07000026': 'E06000063', 'E07000028': 'E06000063', 'E07000029': 'E06000063',
    'E07000027': 'E06000064', 'E07000030': 'E06000064', 'E07000031': 'E06000064',
    
    # Northamptonshire (Apr 2021): 7 → 2
    'E07000150': 'E06000061', 'E07000152': 'E06000061', 'E07000153': 'E06000061', 'E07000156': 'E06000061',
    'E07000151': 'E06000062', 'E07000154': 'E06000062', 'E07000155': 'E06000062',
    
    # Buckinghamshire (Apr 2020): 4 → 1
    'E07000004': 'E06000060', 'E07000005': 'E06000060', 'E07000006': 'E06000060', 'E07000007': 'E06000060',
    
    # North Yorkshire (Apr 2023): 7 → 1
    'E07000163': 'E06000065', 'E07000164': 'E06000065', 'E07000165': 'E06000065', 'E07000166': 'E06000065',
    'E07000167': 'E06000065', 'E07000168': 'E06000065', 'E07000169': 'E06000065',
    
    # Somerset (Apr 2023): 4 → 1
    'E07000187': 'E06000066', 'E07000188': 'E06000066', 'E07000189': 'E06000066', 'E07000246': 'E06000066',
}

# -----------------------------
# Helper Functions
# -----------------------------

def _write_duck(table_fullname: str, df: pd.DataFrame):
    """Write dataframe to DuckDB with schema support"""
    if not HAVE_DUCKDB:
        log.warning("duckdb not installed; skipping DuckDB writes.")
        return
    
    schema, table = table_fullname.split(".", 1) if "." in table_fullname else ("main", table_fullname)
    
    con = duckdb.connect(str(DUCK_PATH))
    try:
        if schema != "main":
            con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        con.register("df_tmp", df)
        con.execute(f"CREATE OR REPLACE TABLE {schema}.{table} AS SELECT * FROM df_tmp")
        log.info(f"✓ Wrote {len(df)} rows to {schema}.{table}")
    finally:
        con.close()


def load_lookup() -> pd.DataFrame:
    """Load and process ONS LAD->ITL lookup"""
    if not LOOKUP_PATH.exists():
        log.error(f"Lookup file not found: {LOOKUP_PATH}")
        raise FileNotFoundError(f"Required lookup file missing: {LOOKUP_PATH}")
    
    lookup = pd.read_csv(LOOKUP_PATH)
    lookup.columns = [col.replace('\ufeff', '') for col in lookup.columns]
    
    cols_needed = ['LAD25CD', 'LAD25NM', 'ITL325CD', 'ITL325NM', 
                   'ITL225CD', 'ITL225NM', 'ITL125CD', 'ITL125NM']
    
    missing = [c for c in cols_needed if c not in lookup.columns]
    if missing:
        log.error(f"Lookup missing columns: {missing}")
        raise ValueError(f"Lookup file incomplete: missing {missing}")
    
    lookup = lookup[cols_needed].copy()
    lookup['ITL1_ONS_CD'] = lookup['ITL125CD'].map(TLC_TO_ONS)
    lookup = lookup.rename(columns={'ITL125NM': 'ITL1_NAME_TLC'})
    lookup = lookup.drop_duplicates(subset=['LAD25CD'])
    
    log.info(f"Loaded lookup: {len(lookup)} LADs mapped to parent geographies")
    
    return lookup


def fetch_nomis_csv(url: str) -> pd.DataFrame:
    """Fetch CSV from NOMIS API with error handling"""
    log.info(f"NOMIS → Fetching LAD GDHI data (total £m only)...")
    
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        
        content = response.text
        if len(content) < 100:
            log.error(f"Response too short ({len(content)} chars)")
            raise ValueError("Empty or invalid response from NOMIS")
        
        df = pd.read_csv(io.StringIO(content), low_memory=False)
        
        if df.empty:
            log.error("Dataframe is empty after parsing")
            raise ValueError("NOMIS returned empty dataframe")
        
        log.info(f"Fetched {len(df)} rows, {len(df.columns)} columns")
        return df
        
    except Exception as e:
        log.error(f"NOMIS fetch failed: {e}")
        raise


def apply_boundary_concordance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply complete LAD boundary concordance.
    Maps old district codes to 2023 unitary authority codes.
    """
    old_codes = df['region_code'].isin(LAD_CONCORDANCE.keys())
    
    if not old_codes.any():
        log.info("No old LAD codes found (data already uses 2023 boundaries)")
        return df
    
    # Group by reorganization for logging
    reorganizations = {
        'Cumbria': ['E07000026', 'E07000027', 'E07000028', 'E07000029', 'E07000030', 'E07000031'],
        'Northamptonshire': ['E07000150', 'E07000151', 'E07000152', 'E07000153', 'E07000154', 'E07000155', 'E07000156'],
        'Buckinghamshire': ['E07000004', 'E07000005', 'E07000006', 'E07000007'],
        'North Yorkshire': ['E07000163', 'E07000164', 'E07000165', 'E07000166', 'E07000167', 'E07000168', 'E07000169'],
        'Somerset': ['E07000187', 'E07000188', 'E07000189', 'E07000246']
    }
    
    log.info("Applying boundary concordance...")
    
    for area, codes in reorganizations.items():
        area_codes = df['region_code'].isin(codes)
        if area_codes.any():
            affected_obs = area_codes.sum()
            unique_lads = df.loc[area_codes, 'region_code'].nunique()
            log.info(f"  {area}: {affected_obs} observations from {unique_lads} old districts")
    
    # Apply mapping
    df['region_code'] = df['region_code'].map(lambda x: LAD_CONCORDANCE.get(x, x))
    
    total_mapped = old_codes.sum()
    log.info(f"✓ Mapped {total_mapped} observations to 2023 boundaries")
    
    return df


def transform_to_tidy(df_raw: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    """
    Transform NOMIS LAD GDHI data to tidy schema.
    Only processes measure=1 (total £m), NOT per head.
    Applies concordance, standardizes to 2023 boundaries, enriches with parent geographies.
    """
    # Verify we only have measure=1 (total £m)
    if 'MEASURE' in df_raw.columns:
        measures = df_raw['MEASURE'].unique()
        if len(measures) > 1 or (len(measures) == 1 and measures[0] != 1):
            log.warning(f"Expected only measure=1, found: {measures}")
            log.warning("Filtering to measure=1 (GDHI total £m)")
            df_raw = df_raw[df_raw['MEASURE'] == 1].copy()
    
    # Find value column
    value_col = None
    for col in ["OBS_VALUE", "obs_value", "VALUE", "value"]:
        if col in df_raw.columns:
            value_col = col
            break
    
    if value_col is None:
        raise KeyError(f"Could not find value column. Columns: {df_raw.columns.tolist()}")
    
    # Find geography columns
    geo_code_col = "GEOGRAPHY_CODE" if "GEOGRAPHY_CODE" in df_raw.columns else "GEOGRAPHY"
    geo_name_col = "GEOGRAPHY_NAME" if "GEOGRAPHY_NAME" in df_raw.columns else None
    
    # Find date column
    date_col = None
    for col in ["DATE", "date", "Date", "TIME", "time", "Time"]:
        if col in df_raw.columns:
            date_col = col
            break
    
    if date_col is None:
        raise KeyError(f"Could not find date column. Columns: {df_raw.columns.tolist()}")
    
    # Ensure numeric types
    df_raw[value_col] = pd.to_numeric(df_raw[value_col], errors="coerce")
    df_raw[date_col] = pd.to_numeric(df_raw[date_col], errors="coerce")
    
    # Build base tidy frame
    tidy = pd.DataFrame({
        "region_code": df_raw[geo_code_col],
        "region_name": df_raw[geo_name_col] if geo_name_col else None,
        "region_level": "LAD",
        "metric_id": "gdhi_total_mn_gbp",
        "period": df_raw[date_col].astype("Int64"),
        "value": df_raw[value_col],
        "unit": "GBP_m",
        "freq": "A",
        "source": "NOMIS",
        "vintage": VINTAGE,
    })
    
    tidy["geo_hierarchy"] = "LAD>ITL3>ITL2>ITL1"
    
    # Drop nulls and enforce non-negativity
    tidy = tidy.dropna(subset=["period", "value"]).reset_index(drop=True)
    tidy = tidy[tidy['value'] >= 0]
    
    initial_lad_count = tidy['region_code'].nunique()
    initial_obs = len(tidy)
    
    log.info(f"Initial: {initial_lad_count} unique LADs, {initial_obs} observations")
    
    # DEDUPLICATE (in case NOMIS has same bug as GVA)
    before_dedup = len(tidy)
    tidy = tidy.drop_duplicates(subset=['region_code', 'period'], keep='first')
    after_dedup = len(tidy)
    
    if before_dedup > after_dedup:
        dropped = before_dedup - after_dedup
        log.warning(f"⚠️  Dropped {dropped} duplicate LAD-period records (NOMIS API issue)")
    
    # APPLY CONCORDANCE
    tidy = apply_boundary_concordance(tidy)
    
    # Aggregate by LAD-period (multiple old districts → 1 new unitary)
    tidy = tidy.groupby(['region_code', 'region_level', 'metric_id', 'period', 
                         'unit', 'freq', 'vintage', 'geo_hierarchy'], as_index=False).agg({
        'value': 'sum',
        'region_name': 'first',
        'source': 'first'
    })
    
    post_concordance_lads = tidy['region_code'].nunique()
    post_concordance_obs = len(tidy)
    
    if post_concordance_lads != initial_lad_count or post_concordance_obs < after_dedup:
        log.info(f"After concordance: {post_concordance_lads} LADs, {post_concordance_obs} observations")
    
    # Final duplicate check
    dups = tidy.duplicated(subset=['region_code', 'period'], keep=False)
    if dups.any():
        log.error(f"Found {dups.sum()} duplicate LAD-period combinations after transform")
        raise ValueError("Duplicate LAD-period combinations in final output")
    
    # Check match rate BEFORE merge
    match_rate = tidy['region_code'].isin(lookup['LAD25CD']).mean()
    log.info(f"Lookup match rate (post-concordance): {match_rate:.2%}")
    
    if match_rate < 0.98:
        unmatched = tidy[~tidy['region_code'].isin(lookup['LAD25CD'])]['region_code'].unique()
        log.warning(f"Only {match_rate:.2%} matched after concordance")
        log.warning(f"Unmatched codes (first 10): {unmatched[:10].tolist()}")
    
    # Filter to only LADs in 2023 lookup
    before_filter = len(tidy)
    tidy = tidy[tidy['region_code'].isin(lookup['LAD25CD'])]
    after_filter = len(tidy)
    
    if before_filter > after_filter:
        filtered = before_filter - after_filter
        log.info(f"Filtered {filtered} observations not in 2023 lookup")
    
    # Enrich with parent geographies
    tidy = tidy.merge(
        lookup[['LAD25CD', 'ITL325CD', 'ITL325NM', 'ITL225CD', 'ITL225NM', 
                'ITL1_ONS_CD', 'ITL1_NAME_TLC']],
        left_on='region_code',
        right_on='LAD25CD',
        how='left'
    )
    
    tidy = tidy.rename(columns={
        'ITL325CD': 'itl3_code',
        'ITL325NM': 'itl3_name',
        'ITL225CD': 'itl2_code',
        'ITL225NM': 'itl2_name',
        'ITL1_ONS_CD': 'itl1_code',
        'ITL1_NAME_TLC': 'itl1_name'
    })
    
    tidy = tidy.drop(columns=['LAD25CD'])
    
    # Validate parent mappings
    missing_itl1 = tidy['itl1_code'].isna().sum()
    if missing_itl1 > 0:
        missing_lads = tidy[tidy['itl1_code'].isna()]['region_code'].unique()
        log.error(f"{missing_itl1} observations missing ITL1 mapping")
        log.error(f"Affected LADs: {missing_lads.tolist()}")
        raise ValueError("Parent geography lookup incomplete")
    
    final_lad_count = tidy['region_code'].nunique()
    log.info(f"Final: {final_lad_count} LADs, {len(tidy)} observations")
    log.info(f"Year range: {tidy['period'].min()}-{tidy['period'].max()}")
    
    return tidy


# -----------------------------
# Main Pipeline
# -----------------------------

def main():
    log.info("="*70)
    log.info("LAD GDHI INGEST v1.0 (TOTAL £M ONLY)")
    log.info("="*70)
    log.info(f"Vintage: {VINTAGE}")
    log.info(f"Dataset: NM_185_1 (Regional GDHI)")
    log.info(f"Metric: GDHI total (£ million) - measure=1 only")
    log.info(f"Note: GDHI per head will be derived later from total/population")
    log.info(f"\nBoundary changes handled:")
    log.info(f"  - Cumbria (2023): 6 → 2")
    log.info(f"  - Northamptonshire (2021): 7 → 2")
    log.info(f"  - Buckinghamshire (2020): 4 → 1")
    log.info(f"  - North Yorkshire (2023): 7 → 1")
    log.info(f"  - Somerset (2023): 4 → 1")
    
    # Load lookup
    lookup = load_lookup()
    
    # Fetch GDHI data
    log.info("\n--- Fetching from NOMIS ---")
    df_raw = fetch_nomis_csv(NOMIS_URL)
    
    # Save raw
    raw_path = RAW_DIR / "lad_gdhi_nomis.csv"
    df_raw.to_csv(raw_path, index=False)
    log.info(f"✓ Saved raw → {raw_path}")
    
    _write_duck("bronze.gdhi_lad_raw", df_raw)
    
    # Transform to tidy
    log.info("\n--- Transforming to tidy schema ---")
    tidy = transform_to_tidy(df_raw, lookup)
    
    # Sort and validate
    tidy = tidy.sort_values(['region_code', 'period']).reset_index(drop=True)
    
    # Final duplicate check
    dups = tidy.duplicated(subset=['region_code', 'period'], keep=False)
    if dups.any():
        log.error(f"Found {dups.sum()} duplicate LAD-period combinations in final output")
        raise ValueError("Duplicate LAD-period combinations detected")
    
    # Save silver CSV
    tidy.to_csv(SILVER_CSV, index=False)
    log.info(f"\n✓ Saved silver CSV → {SILVER_CSV} ({len(tidy)} rows)")
    
    # Write to DuckDB silver
    if HAVE_DUCKDB:
        _write_duck("silver.lad_gdhi_history", tidy)
    
    # Summary report
    log.info("\n" + "="*70)
    log.info("INGEST SUMMARY")
    log.info("="*70)
    
    log.info(f"\nLAD count: {tidy['region_code'].nunique()}")
    log.info(f"Years: {tidy['period'].min()} - {tidy['period'].max()}")
    log.info(f"Total observations: {len(tidy)}")
    
    # Year coverage
    year_coverage = tidy.groupby('period')['region_code'].nunique()
    log.info(f"\nTemporal coverage:")
    log.info(f"  Average LADs per year: {year_coverage.mean():.0f}")
    log.info(f"  Min LADs in a year: {year_coverage.min()}")
    log.info(f"  Max LADs in a year: {year_coverage.max()}")
    
    log.info(f"\nParent Geography Coverage:")
    log.info(f"  ITL3 regions: {tidy['itl3_code'].nunique()}")
    log.info(f"  ITL2 regions: {tidy['itl2_code'].nunique()}")
    log.info(f"  ITL1 regions: {tidy['itl1_code'].nunique()}")
    
    log.info(f"\nLADs by ITL1:")
    itl1_coverage = tidy.groupby('itl1_name')['region_code'].nunique().sort_values(ascending=False)
    for itl1, count in itl1_coverage.items():
        log.info(f"  {itl1}: {count} LADs")
    
    # GDHI value summary
    log.info(f"\nGDHI value distribution (£m):")
    log.info(f"  Min:    {tidy['value'].min():,.0f}")
    log.info(f"  Median: {tidy['value'].median():,.0f}")
    log.info(f"  Max:    {tidy['value'].max():,.0f}")
    
    if tidy['period'].max() in tidy['period'].values:
        latest_year_total = tidy[tidy['period'] == tidy['period'].max()]['value'].sum()
        log.info(f"  Total ({tidy['period'].max()}): £{latest_year_total:,.0f}m")
    
    log.info(f"\n✓ DATA QUALITY CHECKS:")
    log.info(f"  ✓ No duplicate LAD-period combinations")
    log.info(f"  ✓ Only measure=1 (total £m) ingested")
    log.info(f"  ✓ All boundary changes mapped to 2023 codes")
    log.info(f"  ✓ All LADs have ITL1 parent mappings")
    log.info(f"  ✓ No negative GDHI values")
    
    log.info("\n📝 NOTE: GDHI per head NOT ingested")
    log.info("   Will be derived later as: gdhi_total_mn_gbp / population_total")
    log.info("   This ensures consistency with reconciled population forecasts")
    
    log.info("\n✅ LAD GDHI ingest complete!")
    log.info("="*70)


if __name__ == "__main__":
    main()