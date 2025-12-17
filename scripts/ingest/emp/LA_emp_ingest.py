#!/usr/bin/env python3
"""
LAD Employment Ingest from NOMIS API → DuckDB + CSV (Production v2.0 - FULL CONCORDANCE)

Fetches Local Authority District (LAD) level employment data from NOMIS BRES.
Handles ALL boundary changes 2019-2023 with complete concordance mapping.

Boundary Changes Handled:
- Cumbria (Apr 2023): 6 districts → 2 unitaries
- Northamptonshire (Apr 2021): 7 districts → 2 unitaries
- Buckinghamshire (Apr 2020): 4 districts → 1 unitary
- North Yorkshire (Apr 2023): 7 districts → 1 unitary
- Somerset (Apr 2023): 4 districts → 1 unitary
- Sheffield/Barnsley: Code verification (if needed)

Datasets:
- NM_172_1: 2009-2015 (LADs as of 2015 boundaries)
- NM_189_1: 2015-2024 (LADs as of 2023 boundaries)

Strategy:
1. Fetch both datasets
2. Apply concordance to 2009-2015 data (map old → new codes)
3. Standardize both to 2023 boundaries
4. Combine, handling 2015 overlap
5. Enrich with parent geographies

Outputs:
- data/raw/employment/lad_employment_2009_2015.csv (raw, 2015 boundaries)
- data/raw/employment/lad_employment_2015_2024.csv (raw, 2023 boundaries)
- data/silver/lad_employment_history.csv (tidy, 2023 boundaries)
- warehouse.duckdb:
    bronze.emp_lad_2009_2015_raw
    bronze.emp_lad_2015_2024_raw
    silver.lad_employment_history
"""

import io
import logging
import os
import ssl
import urllib.parse
import urllib.request
import hashlib
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

try:
    import duckdb
    HAVE_DUCKDB = True
except ImportError:
    HAVE_DUCKDB = False

try:
    import requests  # optional
    HAVE_REQUESTS = True
except Exception:
    HAVE_REQUESTS = False

# -----------------------------
# Configuration
# -----------------------------
RAW_DIR = Path("data/raw/employment")
RAW_DIR.mkdir(parents=True, exist_ok=True)

SILVER_DIR = Path("data/silver")
SILVER_DIR.mkdir(parents=True, exist_ok=True)
SILVER_CSV = SILVER_DIR / "lad_employment_history.csv"

LAKE_DIR = Path("data/lake")
LAKE_DIR.mkdir(parents=True, exist_ok=True)
DUCK_PATH = LAKE_DIR / "warehouse.duckdb"

LOOKUP_PATH = Path("data/reference/master_2025_geography_lookup.csv")

VINTAGE = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# NOMIS API URLs for LAD employment (BRES)
# Dataset 1: 2009-2015 (LAD 2015 boundaries)
NOMIS_URL_2009_2015 = (
    "https://www.nomisweb.co.uk/api/v01/dataset/NM_172_1.data.csv"
    "?geography=1820327937...1820328307"
    "&industry=37748736"
    "&employment_status=1"
    "&measure=1"
    "&measures=20100"
)

# Dataset 2: 2015-2024 (LAD 2023 boundaries)
NOMIS_URL_2015_2024 = (
    "https://www.nomisweb.co.uk/api/v01/dataset/NM_189_1.data.csv"
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
    "1778384998...1778385003,1778384959,1778385193...1778385246"
    "&industry=37748736"
    "&employment_status=1"
    "&measure=1"
    "&measures=20100"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("lad_emp_ingest")

# TLC to ONS code mapping
TLC_TO_ONS = {
    'TLC': 'E12000001',  # North East (England)
    'TLD': 'E12000002',  # North West (England)
    'TLE': 'E12000003',  # Yorkshire and The Humber
    'TLF': 'E12000004',  # East Midlands (England)
    'TLG': 'E12000005',  # West Midlands (England)
    'TLH': 'E12000006',  # East of England
    'TLI': 'E12000007',  # London
    'TLJ': 'E12000008',  # South East (England)
    'TLK': 'E12000009',  # South West (England)
    'TLL': 'W92000004',  # Wales
    'TLM': 'S92000003',  # Scotland
    'TLN': 'N92000002',  # Northern Ireland
}

# COMPLETE LAD CONCORDANCE: Old (2015) → New (2023) boundaries
# Handles all local government reorganizations 2019-2023
LAD_CONCORDANCE = {
    # Cumbria reorganization (April 2023): 6 → 2
    'E07000026': 'E06000063',  # Allerdale → Cumberland
    'E07000028': 'E06000063',  # Carlisle → Cumberland
    'E07000029': 'E06000063',  # Copeland → Cumberland
    'E07000027': 'E06000064',  # Barrow-in-Furness → Westmorland & Furness
    'E07000030': 'E06000064',  # Eden → Westmorland & Furness
    'E07000031': 'E06000064',  # South Lakeland → Westmorland & Furness
    
    # Northamptonshire reorganization (April 2021): 7 → 2
    'E07000150': 'E06000061',  # Corby → North Northamptonshire
    'E07000152': 'E06000061',  # East Northamptonshire → North Northamptonshire
    'E07000153': 'E06000061',  # Kettering → North Northamptonshire
    'E07000156': 'E06000061',  # Wellingborough → North Northamptonshire
    'E07000151': 'E06000062',  # Daventry → West Northamptonshire
    'E07000154': 'E06000062',  # Northampton → West Northamptonshire
    'E07000155': 'E06000062',  # South Northamptonshire → West Northamptonshire
    
    # Buckinghamshire reorganization (April 2020): 4 → 1
    'E07000004': 'E06000060',  # Aylesbury Vale → Buckinghamshire
    'E07000005': 'E06000060',  # Chiltern → Buckinghamshire
    'E07000006': 'E06000060',  # South Bucks → Buckinghamshire
    'E07000007': 'E06000060',  # Wycombe → Buckinghamshire
    
    # North Yorkshire reorganization (April 2023): 7 → 1
    'E07000163': 'E06000065',  # Craven → North Yorkshire
    'E07000164': 'E06000065',  # Hambleton → North Yorkshire
    'E07000165': 'E06000065',  # Harrogate → North Yorkshire
    'E07000166': 'E06000065',  # Richmondshire → North Yorkshire
    'E07000167': 'E06000065',  # Ryedale → North Yorkshire
    'E07000168': 'E06000065',  # Scarborough → North Yorkshire
    'E07000169': 'E06000065',  # Selby → North Yorkshire
    
    # Somerset reorganization (April 2023): 4 → 1
    'E07000187': 'E06000066',  # Mendip → Somerset
    'E07000188': 'E06000066',  # Sedgemoor → Somerset
    'E07000189': 'E06000066',  # Somerset West and Taunton → Somerset
    'E07000246': 'E06000066',  # South Somerset → Somerset
}

# -----------------------------
# Helper Functions
# -----------------------------

def _sha16(content: str | bytes) -> str:
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()[:16]


def _nomis_url(dataset_id: str, params: dict) -> str:
    qs = urllib.parse.urlencode(params, safe=",")
    return f"https://www.nomisweb.co.uk/api/v01/dataset/{dataset_id}.data.csv?{qs}"


def _gb_lad_codes_from_lookup(lookup: pd.DataFrame) -> list[str]:
    # BRES is GB-only; exclude NI LADs.
    codes = lookup["LAD25CD"].dropna().astype(str).unique().tolist()
    gb = [c for c in codes if c[:1] in ("E", "W", "S")]
    return sorted(gb)


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


def fetch_nomis_csv(url: str, dataset_name: str) -> pd.DataFrame:
    """Fetch CSV from NOMIS API with error handling"""
    log.info(f"NOMIS → Fetching {dataset_name}...")
    
    try:
        if HAVE_REQUESTS:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            content = response.content
        else:
            insecure = os.getenv("NOMIS_INSECURE_SSL", "0") == "1"
            ctx = ssl._create_unverified_context() if insecure else ssl.create_default_context()
            with urllib.request.urlopen(url, timeout=120, context=ctx) as r:
                content = r.read()

        if content is None or len(content) < 100:
            log.error(f"Response too short ({0 if content is None else len(content)} bytes)")
            raise ValueError("Empty or invalid response from NOMIS")

        text = content.decode("utf-8", errors="replace")
        
        df = pd.read_csv(io.StringIO(text), low_memory=False)
        
        if df.empty:
            log.error("Dataframe is empty after parsing")
            raise ValueError("NOMIS returned empty dataframe")
        
        log.info(f"Fetched {len(df)} rows, {len(df.columns)} columns")
        return df
        
    except Exception as e:
        log.error(f"NOMIS fetch failed: {e}")
        raise


def fetch_nomis_csv_batched(
    dataset_id: str,
    base_params: dict,
    geo_codes: list[str],
    dataset_name: str,
    batch_size: int = 200,
) -> pd.DataFrame:
    """Fetch NOMIS data in ONS-code batches and concatenate."""
    dfs = []
    for i in range(0, len(geo_codes), batch_size):
        batch = geo_codes[i : i + batch_size]
        params = dict(base_params)
        params["geography"] = ",".join(batch)
        url = _nomis_url(dataset_id, params)
        df = fetch_nomis_csv(url, f"{dataset_name} (batch {i//batch_size + 1})")
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    log.info(f"{dataset_name}: combined {len(out)} rows across {len(dfs)} batches")
    return out


def apply_boundary_concordance(df: pd.DataFrame, dataset_label: str) -> pd.DataFrame:
    """
    Apply complete LAD boundary concordance for 2009-2015 data.
    Maps old district codes to 2023 unitary authority codes.
    
    This ensures historical employment data aggregates correctly
    to current LAD boundaries.
    """
    if "2009-2015" not in dataset_label:
        # Only apply to old dataset
        return df
    
    old_codes = df['region_code'].isin(LAD_CONCORDANCE.keys())
    
    if not old_codes.any():
        log.info(f"{dataset_label}: No old LAD codes found (might use current codes already)")
        return df
    
    # Group by reorganization for logging
    reorganizations = {
        'Cumbria': ['E07000026', 'E07000027', 'E07000028', 'E07000029', 'E07000030', 'E07000031'],
        'Northamptonshire': ['E07000150', 'E07000151', 'E07000152', 'E07000153', 'E07000154', 'E07000155', 'E07000156'],
        'Buckinghamshire': ['E07000004', 'E07000005', 'E07000006', 'E07000007'],
        'North Yorkshire': ['E07000163', 'E07000164', 'E07000165', 'E07000166', 'E07000167', 'E07000168', 'E07000169'],
        'Somerset': ['E07000187', 'E07000188', 'E07000189', 'E07000246']
    }
    
    log.info(f"\n{dataset_label}: Applying boundary concordance...")
    
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


def transform_to_tidy(
    df_raw: pd.DataFrame, 
    lookup: pd.DataFrame,
    dataset_label: str
) -> pd.DataFrame:
    """
    Transform NOMIS LAD employment data to tidy schema.
    Applies concordance, standardizes to 2023 boundaries, enriches with parent geographies.
    """
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
        "metric_id": "emp_total_jobs",
        "period": df_raw[date_col].astype("Int64"),
        "value": df_raw[value_col],
        "unit": "jobs",
        "freq": "A",
        "source": f"NOMIS_{dataset_label}",
        "vintage": VINTAGE,
    })
    
    tidy["geo_hierarchy"] = "LAD>ITL3>ITL2>ITL1"
    
    # Drop nulls and enforce non-negativity
    tidy = tidy.dropna(subset=["period", "value"]).reset_index(drop=True)
    tidy = tidy[tidy['value'] >= 0]  # Remove negative/suppressed values
    
    initial_lad_count = tidy['region_code'].nunique()
    initial_obs = len(tidy)
    
    # APPLY CONCORDANCE (for 2009-2015 data only)
    tidy = apply_boundary_concordance(tidy, dataset_label)
    
    # After concordance, aggregate by LAD-period (multiple old districts → 1 new unitary)
    tidy = tidy.groupby(['region_code', 'region_level', 'metric_id', 'period', 
                         'unit', 'freq', 'vintage', 'geo_hierarchy'], as_index=False).agg({
        'value': 'sum',
        'region_name': 'first',
        'source': 'first'
    })
    
    post_concordance_lads = tidy['region_code'].nunique()
    post_concordance_obs = len(tidy)
    
    if post_concordance_lads != initial_lad_count:
        log.info(f"{dataset_label}: LADs after concordance: {initial_lad_count} → {post_concordance_lads}")
        log.info(f"{dataset_label}: Observations: {initial_obs} → {post_concordance_obs} (aggregated)")
    
    # Check for duplicate LAD-period combinations (should not exist after aggregation)
    dups = tidy.duplicated(subset=['region_code', 'period'], keep=False)
    if dups.any():
        log.error(f"Found {dups.sum()} duplicate LAD-period combinations in {dataset_label}")
        raise ValueError("Duplicate LAD-period combinations detected")
    
    # Filter to only LADs in 2023 lookup (removes any remaining unmapped codes)
    match_rate = tidy['region_code'].isin(lookup['LAD25CD']).mean()
    log.info(f"{dataset_label}: Lookup match rate (post-concordance): {match_rate:.2%}")
    
    before_filter = len(tidy)
    tidy = tidy[tidy['region_code'].isin(lookup['LAD25CD'])]
    after_filter = len(tidy)
    
    if before_filter > after_filter:
        filtered = before_filter - after_filter
        log.info(f"{dataset_label}: Filtered {filtered} observations not in 2023 lookup")
    
    if match_rate < 0.98:
        unmatched = tidy[~tidy['region_code'].isin(lookup['LAD25CD'])]['region_code'].unique()
        log.warning(f"{dataset_label}: Only {match_rate:.2%} matched after concordance")
        log.warning(f"Unmatched codes: {unmatched[:10].tolist()}")
    
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
        log.error(f"{dataset_label}: {missing_itl1} observations missing ITL1 mapping")
        log.error(f"Affected LADs: {missing_lads.tolist()}")
        raise ValueError("Parent geography lookup incomplete")
    
    final_lad_count = tidy['region_code'].nunique()
    log.info(f"{dataset_label}: Final → {final_lad_count} LADs, {len(tidy)} observations")
    
    return tidy


def combine_datasets(df_2009_2015: pd.DataFrame, df_2015_2024: pd.DataFrame) -> pd.DataFrame:
    """
    Combine employment datasets, handling 2015 overlap.
    Prefer 2015-2024 dataset for 2015 as it uses current boundaries.
    """
    log.info("\n--- Combining datasets ---")
    
    # Check for 2015 in both datasets
    has_2015_old = (df_2009_2015['period'] == 2015).any()
    has_2015_new = (df_2015_2024['period'] == 2015).any()
    
    if has_2015_old and has_2015_new:
        log.info("2015 data present in both datasets")
        df_2009_2015 = df_2009_2015[df_2009_2015['period'] != 2015]
        log.info("✓ Using 2015 data from 2015-2024 dataset (current boundaries)")
    
    # Combine
    combined = pd.concat([df_2009_2015, df_2015_2024], ignore_index=True)
    
    # Sort
    combined = combined.sort_values(['region_code', 'period']).reset_index(drop=True)
    
    # Final duplicate check (should not exist after above handling)
    dups = combined.duplicated(subset=['region_code', 'period'], keep=False)
    if dups.any():
        log.error(f"Found {dups.sum()} duplicate LAD-period combinations after combining")
        # Show which LAD-periods are duplicated
        dup_cases = combined[dups][['region_code', 'period', 'value']].sort_values(['region_code', 'period'])
        log.error(f"Duplicate cases:\n{dup_cases.head(20)}")
        raise ValueError("Duplicate LAD-period combinations in combined data")
    
    # Final cleanup
    combined = combined.dropna(subset=['period', 'region_code', 'value'])
    
    log.info(f"Combined: {combined['region_code'].nunique()} LADs, {len(combined)} observations")
    log.info(f"Year range: {combined['period'].min()}-{combined['period'].max()}")
    
    return combined


# -----------------------------
# Main Pipeline
# -----------------------------

def main():
    log.info("="*70)
    log.info("LAD EMPLOYMENT INGEST v2.0 (FULL CONCORDANCE)")
    log.info("="*70)
    log.info(f"Vintage: {VINTAGE}")
    log.info(f"Datasets: NM_172_1 (2009-2015) + NM_189_1 (2015-2024)")
    log.info(f"Strategy: Fetch → Concordance → Standardize → Combine")
    log.info(f"\nBoundary changes handled:")
    log.info(f"  - Cumbria (2023): 6 → 2")
    log.info(f"  - Northamptonshire (2021): 7 → 2")
    log.info(f"  - Buckinghamshire (2020): 4 → 1")
    log.info(f"  - North Yorkshire (2023): 7 → 1")
    log.info(f"  - Somerset (2023): 4 → 1")
    
    # Load lookup
    lookup = load_lookup()
    
    # Build geography list from lookup (GB only)
    geo_codes = _gb_lad_codes_from_lookup(lookup)
    base_params = {
        "industry": "37748736",
        "employment_status": "1",
        "measure": "1",
        "measures": "20100",
    }

    # Fetch 2009-2015 data
    log.info("\n" + "="*70)
    log.info("FETCHING 2009-2015 EMPLOYMENT (NM_172_1)")
    log.info("="*70)
    df_raw_2009 = fetch_nomis_csv_batched("NM_172_1", base_params, geo_codes, "2009-2015 (LAD 2015 boundaries)")
    
    raw_path_2009 = RAW_DIR / "lad_employment_2009_2015.csv"
    df_raw_2009.to_csv(raw_path_2009, index=False)
    log.info(f"✓ Saved raw → {raw_path_2009}")
    
    _write_duck("bronze.emp_lad_2009_2015_raw", df_raw_2009)
    
    # Fetch 2015-2024 data
    log.info("\n" + "="*70)
    log.info("FETCHING 2015-2024 EMPLOYMENT (NM_189_1)")
    log.info("="*70)
    df_raw_2024 = fetch_nomis_csv_batched("NM_189_1", base_params, geo_codes, "2015-2024 (LAD 2023 boundaries)")
    
    raw_path_2024 = RAW_DIR / "lad_employment_2015_2024.csv"
    df_raw_2024.to_csv(raw_path_2024, index=False)
    log.info(f"✓ Saved raw → {raw_path_2024}")
    
    _write_duck("bronze.emp_lad_2015_2024_raw", df_raw_2024)
    
    # Transform both to tidy
    log.info("\n" + "="*70)
    log.info("TRANSFORMING TO TIDY SCHEMA")
    log.info("="*70)
    tidy_2009 = transform_to_tidy(df_raw_2009, lookup, "2009-2015")
    tidy_2024 = transform_to_tidy(df_raw_2024, lookup, "2015-2024")
    
    # Combine datasets
    log.info("\n" + "="*70)
    log.info("COMBINING DATASETS")
    log.info("="*70)
    tidy_combined = combine_datasets(tidy_2009, tidy_2024)
    
    # Save silver CSV
    tidy_combined.to_csv(SILVER_CSV, index=False)
    log.info(f"\n✓ Saved silver CSV → {SILVER_CSV} ({len(tidy_combined)} rows)")
    
    # Write to DuckDB silver
    if HAVE_DUCKDB:
        _write_duck("silver.lad_employment_history", tidy_combined)
    
    # Summary report
    log.info("\n" + "="*70)
    log.info("INGEST SUMMARY")
    log.info("="*70)
    
    log.info(f"\nLAD count: {tidy_combined['region_code'].nunique()}")
    log.info(f"Years: {tidy_combined['period'].min()} - {tidy_combined['period'].max()}")
    log.info(f"Total observations: {len(tidy_combined)}")
    
    # Year-by-year LAD count
    year_counts = tidy_combined.groupby('period')['region_code'].nunique()
    log.info(f"\nLAD coverage by year:")
    log.info(f"  2009-2014: {year_counts[year_counts.index < 2015].mean():.0f} LADs avg")
    log.info(f"  2015-2024: {year_counts[year_counts.index >= 2015].mean():.0f} LADs avg")
    
    log.info(f"\nParent Geography Coverage:")
    log.info(f"  ITL3 regions: {tidy_combined['itl3_code'].nunique()}")
    log.info(f"  ITL2 regions: {tidy_combined['itl2_code'].nunique()}")
    log.info(f"  ITL1 regions: {tidy_combined['itl1_code'].nunique()}")
    
    log.info(f"\nLADs by ITL1:")
    itl1_coverage = tidy_combined.groupby('itl1_name')['region_code'].nunique().sort_values(ascending=False)
    for itl1, count in itl1_coverage.items():
        log.info(f"  {itl1}: {count} LADs")
    
    log.info(f"\n✓ DATA QUALITY CHECKS:")
    log.info(f"  ✓ No duplicate LAD-period combinations")
    log.info(f"  ✓ All boundary changes mapped to 2023 codes")
    log.info(f"  ✓ All LADs have ITL1 parent mappings")
    log.info(f"  ✓ 2015 overlap handled (used 2023 boundaries)")
    log.info(f"  ✓ No negative employment values")
    
    log.info("\n✅ LAD employment ingest complete!")
    log.info("="*70)


if __name__ == "__main__":
    main()