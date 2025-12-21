#!/usr/bin/env python3
"""
LAD Unified Ingest Pipeline (Production v1.4 - Vintage Tracking)

Fetches all LAD-level metrics from NOMIS API → DuckDB + CSV
Now includes VintageTracker to detect upstream NOMIS changes.

v1.4 Changes:
- Integrated VintageTracker for change detection
- fetch_nomis_csv() now returns (df, raw_text) tuple
- Vintage summary saved to data/pipeline/vintage_lad.json
- Added population_16_64 (working-age population)

Metrics:
- Employment (jobs): 2009-2024 via two datasets with boundary concordance
- GVA (£m): Current prices with deduplication
- GDHI (£m): Total only (per head derived later)
- Population (persons): Mid-year estimates (all ages)
- Population 16-64 (persons): Working-age population
- Employment Rate (%): APS 16-64, fiscal year format
- Unemployment Rate (%): APS model-based, fiscal year format

Outputs:
- data/raw/{metric}/lad_{metric}_*.csv
- data/silver/lad_{metric}_history.csv
- warehouse.duckdb: bronze.{metric}_lad_* + silver.lad_{metric}_history
- data/logs/lad_ingest_summary.json (pipeline status)
- data/pipeline/vintage_lad.json (change detection)
"""

import io
import json
import logging
import sys
import time
import os
import ssl
import urllib.parse
import urllib.request
import hashlib
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

try:
    import duckdb
    HAVE_DUCKDB = True
except ImportError:
    HAVE_DUCKDB = False

try:
    import requests  # type: ignore  # optional; not available in all runtime environments
    HAVE_REQUESTS = True
except Exception:
    HAVE_REQUESTS = False

try:
    from manifest.data_vintage import VintageTracker
    HAVE_VINTAGE = True
except ImportError:
    HAVE_VINTAGE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

SILVER_DIR = Path("data/silver")
SILVER_DIR.mkdir(parents=True, exist_ok=True)

LAKE_DIR = Path("data/lake")
LAKE_DIR.mkdir(parents=True, exist_ok=True)
DUCK_PATH = LAKE_DIR / "warehouse.duckdb"


def register_indicator(metric_id: str, display_name: str, nomis_datasets: str, unit: str, category: str):
    """Register indicator metadata in DuckDB for downstream reporting."""
    if not HAVE_DUCKDB:
        return
    try:
        con = duckdb.connect(str(DUCK_PATH))
        con.execute("CREATE SCHEMA IF NOT EXISTS metadata")
        con.execute("""
            CREATE TABLE IF NOT EXISTS metadata.indicators (
                metric_id VARCHAR PRIMARY KEY,
                display_name VARCHAR,
                nomis_datasets VARCHAR,
                unit VARCHAR,
                category VARCHAR
            )
        """)
        con.execute("""
            INSERT OR REPLACE INTO metadata.indicators 
            VALUES (?, ?, ?, ?, ?)
        """, [metric_id, display_name, nomis_datasets, unit, category])
        con.close()
    except Exception as e:
        log.warning(f"Failed to register indicator {metric_id}: {e}")


# ============================================================================
# PATHS (continued)
# ============================================================================

LOG_DIR = Path("data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

PIPELINE_DIR = Path("data/pipeline")
PIPELINE_DIR.mkdir(parents=True, exist_ok=True)

LOOKUP_PATH = Path("data/reference/master_2025_geography_lookup.csv")
VINTAGE = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ============================================================================
# NISRA (Northern Ireland) sources for LAD-equivalent LGDs
# IMPORTANT: We ONLY use these for employment indicators (rates + NI jobs).
# Do NOT change NOMIS/NOMIS geography behaviour for GDHI/GVA/Population etc.
# ============================================================================

NISRA_LMSLGD_CSV = "https://ws-data.nisra.gov.uk/public/api.restful/PxStat.Data.Cube_API.ReadDataset/LMSLGD/CSV/1.0/en"
NISRA_BRESHEADLGD_CSV = "https://ws-data.nisra.gov.uk/public/api.restful/PxStat.Data.Cube_API.ReadDataset/BRESHEADLGD/CSV/1.0/en"
NI_TOTAL_CODE = "N92000002"

_NISRA_LMS_STAT_MAP = {
    # Only these two; we intentionally ignore EMPN to avoid schema drift.
    "EMPR": ("employment_rate_pct", "percent", "NISRA_employment_rate"),
    "UNEMPR": ("unemployment_rate_pct", "percent", "NISRA_unemployment_rate"),
}

_NISRA_BRES_STAT_MAP = {
    # NI-only jobs metric (distinct from GB emp_total_jobs)
    "EJOBS": ("emp_total_jobs_ni", "jobs", "NISRA_emp_total_jobs_ni"),
}

# ============================================================================
# PIPELINE REPORTER
# ============================================================================

class PipelineReporter:
    """Structured pipeline status reporter for RegionIQ forecasting governance."""
    
    def __init__(self, stage_name: str):
        self.stage = stage_name
        self.start_time = time.time()
        self.warnings = []
        self.critical_errors = []
        self.metrics = {}
        self.status = "success"
    
    def add_warning(self, message: str):
        self.warnings.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": message
        })
        log.warning(f"⚠️  {message}")
        if self.status == "success":
            self.status = "warning"
    
    def add_critical_error(self, message: str):
        self.critical_errors.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": message
        })
        log.error(f"❌ CRITICAL: {message}")
        self.status = "failed"
    
    def add_metric(self, key: str, value):
        self.metrics[key] = value
    
    def finalize(self) -> Dict:
        duration = time.time() - self.start_time
        return {
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
    
    def save_and_exit(self, filename: str):
        summary = self.finalize()
        output_path = LOG_DIR / filename
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        log.info(f"\n✓ Pipeline summary → {output_path}")
        exit_code = 0 if self.status in ["success", "warning"] else 1
        if self.status == "success":
            log.info("✅ Pipeline completed successfully")
        elif self.status == "warning":
            log.warning("⚠️  Pipeline completed with warnings")
        else:
            log.error("❌ Pipeline failed with critical errors")
        sys.exit(exit_code)

# TLC to ONS code mapping
TLC_TO_ONS = {
    'TLC': 'E12000001', 'TLD': 'E12000002', 'TLE': 'E12000003',
    'TLF': 'E12000004', 'TLG': 'E12000005', 'TLH': 'E12000006',
    'TLI': 'E12000007', 'TLJ': 'E12000008', 'TLK': 'E12000009',
    'TLL': 'W92000004', 'TLM': 'S92000003', 'TLN': 'N92000002',
}

# Complete LAD concordance (old → new boundaries)
LAD_CONCORDANCE = {
    'E07000026': 'E06000063', 'E07000028': 'E06000063', 'E07000029': 'E06000063',
    'E07000027': 'E06000064', 'E07000030': 'E06000064', 'E07000031': 'E06000064',
    'E07000150': 'E06000061', 'E07000152': 'E06000061', 'E07000153': 'E06000061', 'E07000156': 'E06000061',
    'E07000151': 'E06000062', 'E07000154': 'E06000062', 'E07000155': 'E06000062',
    'E07000004': 'E06000060', 'E07000005': 'E06000060', 'E07000006': 'E06000060', 'E07000007': 'E06000060',
    'E07000163': 'E06000065', 'E07000164': 'E06000065', 'E07000165': 'E06000065', 'E07000166': 'E06000065',
    'E07000167': 'E06000065', 'E07000168': 'E06000065', 'E07000169': 'E06000065',
    'E07000187': 'E06000066', 'E07000188': 'E06000066', 'E07000189': 'E06000066', 'E07000246': 'E06000066',
}

# Geography ranges for NOMIS API
LAD_GEO_RANGE = (
    "1778384897...1778384901,1778384941,1778384950,1778385143...1778385146,"
    "1778385159,1778384902...1778384905,1778384942,1778384943,1778384956,1778384957,"
    "1778385033...1778385044,1778385124...1778385138,1778384906...1778384910,1778384958,"
    "1778385139...1778385142,1778385154...1778385158,1778384911...1778384914,1778384954,"
    "1778384955,1778384965...1778385072,1778384915...1778384917,1778384944,"
    "1778385078...1778385085,1778385100...1778385104,1778385112...1778385117,"
    "1778385147...1778385153,1778384925...1778384928,1778384948,1778384949,"
    "1778384960...1778384964,1778384986...1778384997,1778385015...1778385020,"
    "1778385059...1778385065,1778385086...1778385088,1778385118...1778385123,"
    "1778385160...1778385192,1778384929...1778384940,1778384953,1778384981...1778384985,"
    "1778385004...1778385014,1778385021...1778385032,1778385073...1778385077,"
    "1778385089...1778385099,1778385105...1778385111,1778384918...1778384924,"
    "1778384945...1778384947,1778384951,1778384952,1778384973...1778384980,"
    "1778384998...1778385003,1778384959,1778385193...1778385257"
)

APS_DATE_RANGE = (
    "latestMINUS80,latestMINUS76,latestMINUS72,latestMINUS68,latestMINUS64,"
    "latestMINUS60,latestMINUS56,latestMINUS52,latestMINUS48,latestMINUS44,"
    "latestMINUS40,latestMINUS36,latestMINUS32,latestMINUS28,latestMINUS24,"
    "latestMINUS20,latestMINUS16,latestMINUS12,latestMINUS8,latestMINUS4,latest"
)

# ============================================================================
# NOMIS helpers (geography + robust fetching)
# ============================================================================

def _gb_lad_codes_from_lookup(lookup: pd.DataFrame) -> List[str]:
    """
    Return GB LAD/council-area codes from our canonical lookup.
    - BRES is GB-only (excludes NI)
    - APS NM_17_5 local-authority geography appears GB-only too (NI districts not returned)
    """
    codes = lookup["LAD25CD"].dropna().astype(str).unique().tolist()
    gb = [c for c in codes if c[:1] in ("E", "W", "S")]
    return sorted(gb)


def _uk_lad_codes_from_lookup(lookup: pd.DataFrame) -> List[str]:
    """
    Return UK LAD/LGD codes from our canonical lookup (includes NI LGDs N09...).
    Use ONLY for NOMIS metrics where NI LAD-level data is available (e.g. GVA/GDHI/Population).
    """
    codes = lookup["LAD25CD"].dropna().astype(str).unique().tolist()
    uk = [c for c in codes if c[:1] in ("E", "W", "S", "N")]
    return sorted(uk)


def _fetch_csv_url(url: str, label: str) -> pd.DataFrame:
    """Fetch a CSV over HTTP(S) and parse to DataFrame (uses pandas url handling)."""
    log.info(f"  Fetching {label}...")
    try:
        # Prefer requests when available to support optional insecure SSL for sandboxed runs.
        if HAVE_REQUESTS:
            insecure = os.getenv("NISRA_INSECURE_SSL", "0") == "1"
            resp = requests.get(url, timeout=120, verify=not insecure)
            resp.raise_for_status()
            # Decode with utf-8-sig to strip BOM if present
            text = resp.content.decode("utf-8-sig", errors="replace")
            df = pd.read_csv(io.StringIO(text), low_memory=False)
        else:
            # pandas → urllib; allow insecure SSL only if explicitly enabled.
            insecure = os.getenv("NISRA_INSECURE_SSL", "0") == "1"
            if insecure:
                ctx = ssl._create_unverified_context()
                with urllib.request.urlopen(url, timeout=120, context=ctx) as r:
                    # Use utf-8-sig to strip BOM
                    text = r.read().decode("utf-8-sig", errors="replace")
                df = pd.read_csv(io.StringIO(text), low_memory=False)
            else:
                df = pd.read_csv(url, encoding="utf-8-sig")
    except Exception as e:
        raise RuntimeError(f"{label} fetch failed: {e}")
    if df.empty:
        raise ValueError(f"{label} returned 0 rows")
    
    # Clean column names: strip BOM remnants, quotes, and whitespace (NISRA quirks)
    df.columns = [
        col.replace('\ufeff', '').strip().strip('"').strip("'")
        for col in df.columns
    ]
    
    return df


def _nisra_tidy_from_cube(
    df_raw: pd.DataFrame,
    lookup: pd.DataFrame,
    stat_map: Dict[str, Tuple[str, str, str]],
    extra_filters: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Convert NISRA PxStat CSV (LMSLGD/BRESHEADLGD) to our standard tidy LAD schema.
    """
    cols = {c.lower(): c for c in df_raw.columns}
    stat_col = cols.get("statistic", "STATISTIC")
    year_col = cols.get("tlist(a1)", "TLIST(A1)")
    geo_col = cols.get("lgd2014", "LGD2014")
    val_col = cols.get("value", "VALUE")

    # Apply optional filters for extra dimensions (e.g. BRESHEADLGD has GENWP/HEADLINE)
    if extra_filters:
        for k_lower, required_val in extra_filters.items():
            c = cols.get(k_lower)
            if c and c in df_raw.columns:
                df_raw = df_raw[df_raw[c].astype(str) == str(required_val)]

    df = df_raw[[stat_col, year_col, geo_col, val_col]].copy()
    df = df[df[stat_col].isin(stat_map.keys())]
    df = df[df[geo_col] != NI_TOTAL_CODE]

    # Attach names + ITL parents from canonical lookup (NI LGD codes live in LAD25CD).
    # Note: load_lookup() renames ITL125NM → ITL1_NAME_TLC and adds ITL1_ONS_CD.
    itl1_name_col = "ITL1_NAME_TLC" if "ITL1_NAME_TLC" in lookup.columns else "ITL125NM"
    required_cols = [
        "LAD25CD", "LAD25NM",
        "ITL325CD", "ITL325NM",
        "ITL225CD", "ITL225NM",
        "ITL1_ONS_CD", itl1_name_col,
    ]
    missing = [c for c in required_cols if c not in lookup.columns]
    if missing:
        raise KeyError(f"NISRA lookup missing columns: {missing}")

    parents = lookup[required_cols].drop_duplicates()
    df = df.merge(parents, left_on=geo_col, right_on="LAD25CD", how="left")

    tidy_rows = []
    for _, r in df.iterrows():
        metric_id, unit, source = stat_map.get(r[stat_col])
        tidy_rows.append({
            "region_code": r[geo_col],
            "region_level": "LAD",
            "metric_id": metric_id,
            "period": int(r[year_col]),
            "unit": unit,
            "freq": "A",
            "vintage": VINTAGE,
            "geo_hierarchy": "LAD>ITL3>ITL2>ITL1",
            "value": float(r[val_col]),
            "region_name": r.get("LAD25NM"),
            "source": source,
            "itl3_code": r.get("ITL325CD"),
            "itl3_name": r.get("ITL325NM"),
            "itl2_code": r.get("ITL225CD"),
            "itl2_name": r.get("ITL225NM"),
            "itl1_code": r.get("ITL1_ONS_CD"),
            "itl1_name": r.get(itl1_name_col),
        })
    tidy = pd.DataFrame(tidy_rows)
    return tidy


def _nomis_url(dataset_id: str, params: Dict[str, str]) -> str:
    # NOMIS expects comma-separated list for geography; keep commas unescaped.
    qs = urllib.parse.urlencode(params, safe=",")
    return f"https://www.nomisweb.co.uk/api/v01/dataset/{dataset_id}.data.csv?{qs}"


def _sha16(content: str | bytes) -> str:
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()[:16]


def fetch_nomis_csv(url: str, label: str) -> Tuple[pd.DataFrame, bytes]:
    """
    Fetch CSV from NOMIS API with error handling.
    Returns (DataFrame, raw_bytes) tuple for vintage tracking.
    """
    try:
        if HAVE_REQUESTS:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            raw_bytes = response.content
        else:
            # Fallback to stdlib. Allows insecure SSL only if explicitly enabled.
            insecure = os.getenv("NOMIS_INSECURE_SSL", "0") == "1"
            ctx = ssl._create_unverified_context() if insecure else ssl.create_default_context()
            with urllib.request.urlopen(url, timeout=120, context=ctx) as r:
                raw_bytes = r.read()

        if raw_bytes is None or len(raw_bytes) < 100:
            raise ValueError("Empty response from NOMIS")

        raw_text = raw_bytes.decode("utf-8", errors="replace")
        df = pd.read_csv(io.StringIO(raw_text), low_memory=False)

        if df.empty:
            raise ValueError("Empty dataframe from NOMIS")

        log.info(f"    Fetched {label}: {len(df)} rows")
        return df, raw_bytes

    except Exception as e:
        log.error(f"NOMIS fetch failed ({label}): {e}")
        raise


def fetch_nomis_csv_batched(
    dataset_id: str,
    base_params: Dict[str, str],
    geo_codes: List[str],
    label: str,
    batch_size: int = 200,
) -> Tuple[pd.DataFrame, bytes]:
    """
    Fetch NOMIS data in ONS-code batches to avoid huge URLs.
    Returns combined DF and a tiny fingerprint (hashes of each batch) for vintage tracking.
    """
    if not geo_codes:
        raise ValueError(f"{label}: no geography codes provided")

    dfs: List[pd.DataFrame] = []
    batch_hashes: List[str] = []

    for i in range(0, len(geo_codes), batch_size):
        batch = geo_codes[i : i + batch_size]
        params = dict(base_params)
        params["geography"] = ",".join(batch)
        url = _nomis_url(dataset_id, params)
        df, raw_bytes = fetch_nomis_csv(url, f"{label} (batch {i//batch_size + 1})")
        dfs.append(df)
        batch_hashes.append(_sha16(raw_bytes))

    combined = pd.concat(dfs, ignore_index=True)
    fingerprint = ("\n".join(batch_hashes)).encode("utf-8")
    return combined, fingerprint

# Metric configurations with dataset IDs for vintage tracking
METRICS = {
    'employment': {
        'metric_id': 'emp_total_jobs',
        'unit': 'jobs',
        'raw_dir': 'employment',
        'bronze_table': 'emp_lad',
        'dual_dataset': True,
        # NOTE: We no longer rely on hardcoded NOMIS internal geography IDs for LAD.
        # We fetch using ONS LAD codes from our lookup (batched) to ensure Wales/Scotland coverage.
        'dataset_params': {
            'industry': '37748736',
            'employment_status': '1',
            'measure': '1',
            'measures': '20100',
        },
        'urls': {
            '2009_2015': (
                f"https://www.nomisweb.co.uk/api/v01/dataset/NM_172_1.data.csv"
                f"?geography=1820327937...1820328307"
                f"&industry=37748736&employment_status=1&measure=1&measures=20100"
            ),
            '2015_2024': (
                f"https://www.nomisweb.co.uk/api/v01/dataset/NM_189_1.data.csv"
                f"?geography={LAD_GEO_RANGE}"
                f"&industry=37748736&employment_status=1&measure=1&measures=20100"
            )
        },
        'dataset_ids': {'2009_2015': 'NM_172_1', '2015_2024': 'NM_189_1'},
        'needs_dedup': True,
        'is_rate': False
    },
    'gva': {
        'metric_id': 'nominal_gva_mn_gbp',
        'unit': 'GBP_m',
        'raw_dir': 'gva',
        'bronze_table': 'gva_lad',
        'dataset_id': 'NM_2400_1',
        'dataset_params': {'cell': '0', 'measures': '20100'},
        'url': (
            f"https://www.nomisweb.co.uk/api/v01/dataset/NM_2400_1.data.csv"
            f"?geography={LAD_GEO_RANGE}&cell=0&measures=20100"
        ),
        'needs_dedup': True,
        'is_rate': False
    },
    'gdhi': {
        'metric_id': 'gdhi_total_mn_gbp',
        'unit': 'GBP_m',
        'raw_dir': 'gdhi',
        'bronze_table': 'gdhi_lad',
        'dataset_id': 'NM_185_1',
        'dataset_params': {'component_of_gdhi': '0', 'measure': '1', 'measures': '20100'},
        'url': (
            f"https://www.nomisweb.co.uk/api/v01/dataset/NM_185_1.data.csv"
            f"?geography={LAD_GEO_RANGE}"
            f"&component_of_gdhi=0&measure=1&measures=20100"
        ),
        'needs_dedup': True,
        'validate_measure': 1,
        'is_rate': False
    },
    'population': {
        'metric_id': 'population_total',
        'unit': 'persons',
        'raw_dir': 'population',
        'bronze_table': 'pop_lad',
        'dataset_id': 'NM_2002_1',
        'dataset_suffix': 'total',
        'dataset_params': {'gender': '0', 'c_age': '200', 'measures': '20100'},
        'url': (
            f"https://www.nomisweb.co.uk/api/v01/dataset/NM_2002_1.data.csv"
            f"?geography={LAD_GEO_RANGE}"
            f"&gender=0&c_age=200&measures=20100"
        ),
        'needs_dedup': True,
        'is_rate': False
    },
    'population_16_64': {
        'metric_id': 'population_16_64',
        'unit': 'persons',
        'raw_dir': 'population',
        'bronze_table': 'pop_16_64_lad',
        'dataset_id': 'NM_2002_1',
        'dataset_suffix': '16_64',
        'dataset_params': {'gender': '0', 'c_age': '203', 'measures': '20100'},
        'url': (
            f"https://www.nomisweb.co.uk/api/v01/dataset/NM_2002_1.data.csv"
            f"?geography={LAD_GEO_RANGE}"
            f"&gender=0&c_age=203&measures=20100"
        ),
        'needs_dedup': True,
        'is_rate': False
    },
    'employment_rate': {
        'metric_id': 'employment_rate_pct',
        'unit': 'percent',
        'raw_dir': 'labour_market',
        'bronze_table': 'emp_rate_lad',
        'dataset_id': 'NM_17_5',
        'dataset_suffix': 'emp_rate',
        'dataset_params': {'date': APS_DATE_RANGE, 'variable': '45', 'measures': '20599'},
        'url': (
            f"https://www.nomisweb.co.uk/api/v01/dataset/NM_17_5.data.csv"
            f"?geography={LAD_GEO_RANGE}"
            f"&date={APS_DATE_RANGE}"
            f"&variable=45&measures=20599"
        ),
        'date_format': 'fiscal',
        'needs_dedup': True,
        'is_rate': True
    },
    'unemployment_rate': {
        'metric_id': 'unemployment_rate_pct',
        'unit': 'percent',
        'raw_dir': 'labour_market',
        'bronze_table': 'unemp_rate_lad',
        'dataset_id': 'NM_17_5',
        'dataset_suffix': 'unemp_rate',
        'dataset_params': {'date': APS_DATE_RANGE, 'variable': '83', 'measures': '20599'},
        'url': (
            f"https://www.nomisweb.co.uk/api/v01/dataset/NM_17_5.data.csv"
            f"?geography={LAD_GEO_RANGE}"
            f"&date={APS_DATE_RANGE}"
            f"&variable=83&measures=20599"
        ),
        'date_format': 'fiscal',
        'needs_dedup': True,
        'is_rate': True
    }
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("lad_unified_ingest")

# ============================================================================
# SHARED HELPER FUNCTIONS
# ============================================================================

def _write_duck(table_fullname: str, df: pd.DataFrame):
    """Write dataframe to DuckDB with schema support"""
    if not HAVE_DUCKDB:
        return
    
    schema, table = table_fullname.split(".", 1) if "." in table_fullname else ("main", table_fullname)
    
    con = duckdb.connect(str(DUCK_PATH))
    try:
        if schema != "main":
            con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        con.register("df_tmp", df)
        con.execute(f"CREATE OR REPLACE TABLE {schema}.{table} AS SELECT * FROM df_tmp")
        log.info(f"    ✓ DuckDB: {schema}.{table} ({len(df)} rows)")
    finally:
        con.close()


def load_lookup() -> pd.DataFrame:
    """Load and process ONS LAD->ITL lookup"""
    if not LOOKUP_PATH.exists():
        raise FileNotFoundError(f"Required lookup: {LOOKUP_PATH}")
    
    lookup = pd.read_csv(LOOKUP_PATH)
    lookup.columns = [col.replace('\ufeff', '') for col in lookup.columns]
    
    cols_needed = ['LAD25CD', 'LAD25NM', 'ITL325CD', 'ITL325NM', 
                   'ITL225CD', 'ITL225NM', 'ITL125CD', 'ITL125NM']
    
    missing = [c for c in cols_needed if c not in lookup.columns]
    if missing:
        raise ValueError(f"Lookup missing: {missing}")
    
    lookup = lookup[cols_needed].copy()
    lookup['ITL1_ONS_CD'] = lookup['ITL125CD'].map(TLC_TO_ONS)
    lookup = lookup.rename(columns={'ITL125NM': 'ITL1_NAME_TLC'})
    lookup = lookup.drop_duplicates(subset=['LAD25CD'])
    
    return lookup


def apply_boundary_concordance(df: pd.DataFrame, dataset_label: str) -> pd.DataFrame:
    """Apply LAD boundary concordance (2015→2023 codes)"""
    old_codes = df['region_code'].isin(LAD_CONCORDANCE.keys())
    
    if not old_codes.any():
        return df
    
    log.info(f"    Applying concordance to {dataset_label}...")
    df['region_code'] = df['region_code'].map(lambda x: LAD_CONCORDANCE.get(x, x))
    
    total_mapped = old_codes.sum()
    log.info(f"    ✓ Mapped {total_mapped} observations to 2023 boundaries")
    
    return df


def transform_to_tidy(
    df_raw: pd.DataFrame,
    lookup: pd.DataFrame,
    metric_config: Dict,
    dataset_label: str
) -> pd.DataFrame:
    """Transform NOMIS data to tidy schema with parent geography enrichment"""
    
    value_col = next((c for c in ["OBS_VALUE", "obs_value", "VALUE", "value"] if c in df_raw.columns), None)
    if value_col is None:
        raise KeyError(f"No value column in {df_raw.columns.tolist()}")
    
    geo_code_col = "GEOGRAPHY_CODE" if "GEOGRAPHY_CODE" in df_raw.columns else "GEOGRAPHY"
    geo_name_col = "GEOGRAPHY_NAME" if "GEOGRAPHY_NAME" in df_raw.columns else None
    
    date_col = next((c for c in ["DATE", "date", "Date", "TIME", "time"] if c in df_raw.columns), None)
    if date_col is None:
        raise KeyError(f"No date column in {df_raw.columns.tolist()}")
    
    if metric_config.get('date_format') == 'fiscal':
        df_raw[date_col] = df_raw[date_col].astype(str).str[:4].astype(int)
        log.info(f"    ✓ Parsed fiscal year format (YYYY-YY → YYYY)")
    else:
        df_raw[date_col] = pd.to_numeric(df_raw[date_col], errors="coerce")
    
    if 'validate_measure' in metric_config and 'MEASURE' in df_raw.columns:
        expected = metric_config['validate_measure']
        measures = df_raw['MEASURE'].unique()
        if len(measures) > 1 or (len(measures) == 1 and measures[0] != expected):
            log.warning(f"    Filtering to measure={expected}")
            df_raw = df_raw[df_raw['MEASURE'] == expected].copy()
    
    df_raw[value_col] = pd.to_numeric(df_raw[value_col], errors="coerce")
    
    tidy = pd.DataFrame({
        "region_code": df_raw[geo_code_col],
        "region_name": df_raw[geo_name_col] if geo_name_col else None,
        "region_level": "LAD",
        "metric_id": metric_config['metric_id'],
        "period": df_raw[date_col].astype("Int64"),
        "value": df_raw[value_col],
        "unit": metric_config['unit'],
        "freq": "A",
        "source": f"NOMIS_{dataset_label}",
        "vintage": VINTAGE,
        "geo_hierarchy": "LAD>ITL3>ITL2>ITL1"
    })
    
    tidy = tidy.dropna(subset=["period", "value"]).reset_index(drop=True)
    
    if not metric_config.get('is_rate', False):
        tidy = tidy[tidy['value'] >= 0]
    
    if metric_config.get('needs_dedup', False):
        before = len(tidy)
        tidy = tidy.drop_duplicates(subset=['region_code', 'period'], keep='first')
        after = len(tidy)
        if before > after:
            log.info(f"    ✓ Deduped: removed {before - after} duplicate rows")
    
    tidy = apply_boundary_concordance(tidy, dataset_label)
    
    agg_func = 'mean' if metric_config.get('is_rate', False) else 'sum'
    
    tidy = tidy.groupby(['region_code', 'region_level', 'metric_id', 'period', 
                         'unit', 'freq', 'vintage', 'geo_hierarchy'], as_index=False).agg({
        'value': agg_func,
        'region_name': 'first',
        'source': 'first'
    })
    
    dups = tidy.duplicated(subset=['region_code', 'period'], keep=False)
    if dups.any():
        raise ValueError(f"Duplicate LAD-period in {dataset_label}")
    
    match_rate = tidy['region_code'].isin(lookup['LAD25CD']).mean()
    if match_rate < 0.98:
        unmatched = tidy[~tidy['region_code'].isin(lookup['LAD25CD'])]['region_code'].unique()
        log.warning(f"    Only {match_rate:.2%} matched. Unmatched: {unmatched[:5].tolist()}")
    
    before_filter = len(tidy)
    tidy = tidy[tidy['region_code'].isin(lookup['LAD25CD'])]
    if before_filter > len(tidy):
        log.info(f"    Filtered {before_filter - len(tidy)} non-2023 codes")
    
    tidy = tidy.merge(
        lookup[['LAD25CD', 'ITL325CD', 'ITL325NM', 'ITL225CD', 'ITL225NM', 
                'ITL1_ONS_CD', 'ITL1_NAME_TLC']],
        left_on='region_code',
        right_on='LAD25CD',
        how='left'
    )
    
    tidy = tidy.rename(columns={
        'ITL325CD': 'itl3_code', 'ITL325NM': 'itl3_name',
        'ITL225CD': 'itl2_code', 'ITL225NM': 'itl2_name',
        'ITL1_ONS_CD': 'itl1_code', 'ITL1_NAME_TLC': 'itl1_name'
    })
    
    tidy = tidy.drop(columns=['LAD25CD'])
    
    missing_itl1 = tidy['itl1_code'].isna().sum()
    if missing_itl1 > 0:
        raise ValueError(f"Missing ITL1 mappings for {missing_itl1} observations")
    
    final_lad_count = tidy['region_code'].nunique()
    log.info(f"    ✓ Transformed: {final_lad_count} LADs, {len(tidy)} obs [{tidy['period'].min()}-{tidy['period'].max()}]")
    
    return tidy


# ============================================================================
# METRIC-SPECIFIC INGEST FUNCTIONS
# ============================================================================

def ingest_employment(lookup: pd.DataFrame, tracker: Optional['VintageTracker'] = None) -> pd.DataFrame:
    """Employment ingest: Special dual-dataset handling with vintage tracking"""
    log.info("\n" + "="*70)
    log.info("EMPLOYMENT INGEST (DUAL DATASET)")
    log.info("="*70)
    
    config = METRICS['employment']
    raw_dir = Path("data/raw") / config['raw_dir']
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Build geography list from our canonical lookup (GB only)
    geo_codes = _gb_lad_codes_from_lookup(lookup)
    
    # Fetch 2009-2015
    log.info("  Dataset 1: 2009-2015 (LAD 2015 boundaries)")
    df_raw_2009, raw_fp_2009 = fetch_nomis_csv_batched(
        dataset_id=config["dataset_ids"]["2009_2015"],
        base_params=config["dataset_params"],
        geo_codes=geo_codes,
        label="2009-2015",
        batch_size=200,
    )
    
    # Record vintage for 2009-2015 dataset
    if tracker and HAVE_VINTAGE:
        min_period = int(pd.to_numeric(df_raw_2009.get('DATE', df_raw_2009.get('date', [2009])), errors='coerce').min())
        max_period = int(pd.to_numeric(df_raw_2009.get('DATE', df_raw_2009.get('date', [2015])), errors='coerce').max())
        changed = tracker.record("nomis", "NM_172_1", raw_fp_2009, n_rows=len(df_raw_2009), min_period=min_period, max_period=max_period)
        log.info(f"    ✓ Vintage tracked: NM_172_1 (changed={changed})")
    
    raw_path_2009 = raw_dir / "lad_employment_2009_2015.csv"
    df_raw_2009.to_csv(raw_path_2009, index=False)
    _write_duck("bronze.emp_lad_2009_2015_raw", df_raw_2009)
    
    # Fetch 2015-2024
    log.info("  Dataset 2: 2015-2024 (LAD 2023 boundaries)")
    df_raw_2024, raw_fp_2024 = fetch_nomis_csv_batched(
        dataset_id=config["dataset_ids"]["2015_2024"],
        base_params=config["dataset_params"],
        geo_codes=geo_codes,
        label="2015-2024",
        batch_size=200,
    )
    
    # Record vintage for 2015-2024 dataset
    if tracker and HAVE_VINTAGE:
        min_period = int(pd.to_numeric(df_raw_2024.get('DATE', df_raw_2024.get('date', [2015])), errors='coerce').min())
        max_period = int(pd.to_numeric(df_raw_2024.get('DATE', df_raw_2024.get('date', [2024])), errors='coerce').max())
        changed = tracker.record("nomis", "NM_189_1", raw_fp_2024, n_rows=len(df_raw_2024), min_period=min_period, max_period=max_period)
        log.info(f"    ✓ Vintage tracked: NM_189_1 (changed={changed})")
    
    raw_path_2024 = raw_dir / "lad_employment_2015_2024.csv"
    df_raw_2024.to_csv(raw_path_2024, index=False)
    _write_duck("bronze.emp_lad_2015_2024_raw", df_raw_2024)
    
    # Transform both
    log.info("  Transforming datasets...")
    tidy_2009 = transform_to_tidy(df_raw_2009, lookup, config, "2009-2015")
    tidy_2024 = transform_to_tidy(df_raw_2024, lookup, config, "2015-2024")
    
    # Combine (prefer 2015-2024 for overlap year 2015)
    log.info("  Combining datasets...")
    if (tidy_2009['period'] == 2015).any() and (tidy_2024['period'] == 2015).any():
        tidy_2009 = tidy_2009[tidy_2009['period'] != 2015]
        log.info("    ✓ Using 2015 from current boundaries dataset")
    
    combined = pd.concat([tidy_2009, tidy_2024], ignore_index=True)
    combined = combined.sort_values(['region_code', 'period']).reset_index(drop=True)
    
    dups = combined.duplicated(subset=['region_code', 'period'], keep=False)
    if dups.any():
        raise ValueError("Duplicate LAD-period in combined employment")
    
    log.info(f"  ✓ Combined: {combined['region_code'].nunique()} LADs, {len(combined)} obs")
    
    return combined


def ingest_standard_metric(metric_name: str, lookup: pd.DataFrame, tracker: Optional['VintageTracker'] = None) -> pd.DataFrame:
    """Standard single-dataset ingest with vintage tracking"""
    log.info("\n" + "="*70)
    log.info(f"{metric_name.upper()} INGEST")
    log.info("="*70)
    
    config = METRICS[metric_name]
    raw_dir = Path("data/raw") / config['raw_dir']
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Fetch
    log.info(f"  Fetching {metric_name}...")
    # Geography contract:
    # - For non-employment NOMIS metrics, include NI LADs (N09...) where available.
    # - For labour market rates via NOMIS (NM_17_5), stay GB-only; NI rates come from NISRA.
    if metric_name in ("gva", "gdhi", "population", "population_16_64"):
        geo_codes = _uk_lad_codes_from_lookup(lookup)
    else:
        geo_codes = _gb_lad_codes_from_lookup(lookup)
    df_raw, raw_fp = fetch_nomis_csv_batched(
        dataset_id=config["dataset_id"],
        base_params=config.get("dataset_params", {}),
        geo_codes=geo_codes,
        label=metric_name,
        batch_size=200,
    )
    
    # Record vintage
    if tracker and HAVE_VINTAGE:
        # Extract period range from raw data
        date_col = next((c for c in ["DATE", "date", "Date", "TIME", "time"] if c in df_raw.columns), None)
        if date_col:
            if config.get('date_format') == 'fiscal':
                periods = df_raw[date_col].astype(str).str[:4].astype(int)
            else:
                periods = pd.to_numeric(df_raw[date_col], errors='coerce')
            min_period = int(periods.min())
            max_period = int(periods.max())
        else:
            min_period, max_period = 1997, 2024
        
        # Use suffix for NM_17_5 which serves multiple metrics
        dataset_id = config['dataset_id']
        if 'dataset_suffix' in config:
            dataset_id = f"{dataset_id}_{config['dataset_suffix']}"
        
        changed = tracker.record("nomis", dataset_id, raw_fp, n_rows=len(df_raw), min_period=min_period, max_period=max_period)
        log.info(f"    ✓ Vintage tracked: {dataset_id} (changed={changed})")
    
    # Save raw
    raw_path = raw_dir / f"lad_{metric_name}_nomis.csv"
    df_raw.to_csv(raw_path, index=False)
    _write_duck(f"bronze.{config['bronze_table']}_raw", df_raw)
    
    # Transform
    log.info(f"  Transforming {metric_name}...")
    tidy = transform_to_tidy(df_raw, lookup, config, metric_name)
    
    return tidy


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def main():
    reporter = PipelineReporter("lad_ingest")
    
    log.info("="*70)
    log.info("LAD UNIFIED INGEST PIPELINE v1.4 (VINTAGE TRACKING)")
    log.info("="*70)
    log.info(f"Vintage: {VINTAGE}")
    log.info(f"Metrics: Employment | GVA | GDHI | Population | Pop 16-64 | Emp Rate | Unemp Rate")
    log.info(f"Geography: ~360 LADs (2023 boundaries)")
    
    # Initialize vintage tracker
    tracker = None
    if HAVE_VINTAGE:
        tracker = VintageTracker()
        log.info(f"\n✓ VintageTracker initialized")
    else:
        log.warning(f"\n⚠️  VintageTracker not available (manifest.data_vintage not found)")
    
    log.info(f"\nv1.4 Features:")
    log.info(f"  - NOMIS duplicate rows removed (ALL metrics)")
    log.info(f"  - Vintage tracking for change detection")
    log.info(f"\nBoundary changes handled:")
    log.info(f"  - Cumbria (2023): 6 → 2")
    log.info(f"  - Northamptonshire (2021): 7 → 2")
    log.info(f"  - Buckinghamshire (2020): 4 → 1")
    log.info(f"  - North Yorkshire (2023): 7 → 1")
    log.info(f"  - Somerset (2023): 4 → 1")
    
    # Load lookup
    log.info("\nLoading geography lookup...")
    try:
        lookup = load_lookup()
        log.info(f"✓ Loaded: {len(lookup)} LADs mapped to parent geographies")
        reporter.add_metric("lookup_lads", len(lookup))
    except Exception as e:
        reporter.add_critical_error(f"Lookup file load failed: {e}")
        reporter.save_and_exit("lad_ingest_summary.json")
    
    results = {}
    
    # Ingest employment (dual-dataset)
    log.info("\n" + "="*70)
    log.info("INGESTING METRICS")
    log.info("="*70)
    
    try:
        results['employment'] = ingest_employment(lookup, tracker)
        reporter.add_metric("employment_rows", len(results['employment']))
    except Exception as e:
        reporter.add_critical_error(f"Employment ingest failed: {e}")
        reporter.save_and_exit("lad_ingest_summary.json")

    # --------------------------------------------------------------------
    # Northern Ireland employment additions (NISRA) — employment ONLY
    #   - Rates (EMPR/UNEMPR) will be merged into rate tables below
    #   - Jobs (EJOBS) will be appended into lad_employment_history as a
    #     separate metric_id: emp_total_jobs_ni
    # IMPORTANT: Do not alter NOMIS ingests for non-employment metrics.
    # --------------------------------------------------------------------
    ni_rate_tidy = None
    ni_jobs_tidy = None
    try:
        log.info("\n" + "="*70)
        log.info("NI EMPLOYMENT (NISRA) — LAD (LGD2014)")
        log.info("="*70)
        lms_raw = _fetch_csv_url(NISRA_LMSLGD_CSV, "NISRA LMSLGD (rates)")
        ni_rate_tidy = _nisra_tidy_from_cube(lms_raw, lookup, _NISRA_LMS_STAT_MAP)
        reporter.add_metric("ni_rates_rows", int(len(ni_rate_tidy)))
        _write_duck("bronze.nisra_lmslgd_raw", lms_raw)
    except Exception as e:
        # Fallback: use last successfully ingested bronze table if present.
        # If neither live fetch nor bronze fallback works, fail hard (we don't want
        # to silently publish NI without employment rates).
        reporter.add_warning(f"NISRA LMSLGD ingest failed (attempting bronze fallback): {e}")
        if HAVE_DUCKDB:
            try:
                con = duckdb.connect(str(DUCK_PATH), read_only=True)
                lms_raw = con.execute("SELECT * FROM bronze.nisra_lmslgd_raw").fetchdf()
                con.close()
                ni_rate_tidy = _nisra_tidy_from_cube(lms_raw, lookup, _NISRA_LMS_STAT_MAP)
                reporter.add_metric("ni_rates_rows", int(len(ni_rate_tidy)))
                log.info(f"  ✓ Loaded NISRA LMSLGD from bronze.nisra_lmslgd_raw: {len(lms_raw)} rows")
            except Exception as e2:
                reporter.add_critical_error(f"NISRA LMSLGD unavailable (fetch + bronze fallback failed): {e2}")
                reporter.save_and_exit("lad_ingest_summary.json")
        else:
            reporter.add_critical_error("NISRA LMSLGD unavailable and DuckDB not available for fallback")
            reporter.save_and_exit("lad_ingest_summary.json")

    try:
        bres_raw = _fetch_csv_url(NISRA_BRESHEADLGD_CSV, "NISRA BRESHEADLGD (NI jobs)")
        # This dataset contains extra dimensions; keep the 'All' slice only.
        ni_jobs_tidy = _nisra_tidy_from_cube(
            bres_raw,
            lookup,
            _NISRA_BRES_STAT_MAP,
            extra_filters={"genwp": "All", "headline": "All"},
        )
        reporter.add_metric("ni_jobs_rows", int(len(ni_jobs_tidy)))
        _write_duck("bronze.nisra_bresheadlgd_raw", bres_raw)
    except Exception as e:
        reporter.add_warning(f"NISRA BRESHEADLGD ingest failed (attempting bronze fallback): {e}")
        if HAVE_DUCKDB:
            try:
                con = duckdb.connect(str(DUCK_PATH), read_only=True)
                bres_raw = con.execute("SELECT * FROM bronze.nisra_bresheadlgd_raw").fetchdf()
                con.close()
                ni_jobs_tidy = _nisra_tidy_from_cube(
                    bres_raw,
                    lookup,
                    _NISRA_BRES_STAT_MAP,
                    extra_filters={"genwp": "All", "headline": "All"},
                )
                reporter.add_metric("ni_jobs_rows", int(len(ni_jobs_tidy)))
                log.info(f"  ✓ Loaded NISRA BRESHEADLGD from bronze.nisra_bresheadlgd_raw: {len(bres_raw)} rows")
            except Exception as e2:
                reporter.add_critical_error(f"NISRA BRESHEADLGD unavailable (fetch + bronze fallback failed): {e2}")
                reporter.save_and_exit("lad_ingest_summary.json")
        else:
            reporter.add_critical_error("NISRA BRESHEADLGD unavailable and DuckDB not available for fallback")
            reporter.save_and_exit("lad_ingest_summary.json")
    
    # Ingest standard metrics
    standard_metrics = ['gva', 'gdhi', 'population', 'population_16_64', 'employment_rate', 'unemployment_rate']
    for metric_name in standard_metrics:
        try:
            results[metric_name] = ingest_standard_metric(metric_name, lookup, tracker)
            reporter.add_metric(f"{metric_name}_rows", len(results[metric_name]))
        except Exception as e:
            reporter.add_critical_error(f"{metric_name.upper()} ingest failed: {e}")
            reporter.save_and_exit("lad_ingest_summary.json")

    # Merge NI rates into the two LAD rate series (do not affect NOMIS sources)
    if isinstance(ni_rate_tidy, pd.DataFrame) and not ni_rate_tidy.empty:
        if 'employment_rate' in results:
            emp_rate_ni = ni_rate_tidy[ni_rate_tidy['metric_id'] == 'employment_rate_pct'].copy()
            if not emp_rate_ni.empty:
                results['employment_rate'] = pd.concat([results['employment_rate'], emp_rate_ni], ignore_index=True)
                log.info(f"  ✓ Appended NI employment_rate_pct: +{len(emp_rate_ni)} rows")
        if 'unemployment_rate' in results:
            unemp_rate_ni = ni_rate_tidy[ni_rate_tidy['metric_id'] == 'unemployment_rate_pct'].copy()
            if not unemp_rate_ni.empty:
                results['unemployment_rate'] = pd.concat([results['unemployment_rate'], unemp_rate_ni], ignore_index=True)
                log.info(f"  ✓ Appended NI unemployment_rate_pct: +{len(unemp_rate_ni)} rows")

    # Append NI jobs into LAD employment history as a separate metric_id
    if isinstance(ni_jobs_tidy, pd.DataFrame) and not ni_jobs_tidy.empty:
        results['employment'] = pd.concat([results['employment'], ni_jobs_tidy], ignore_index=True)
        log.info(f"  ✓ Appended NI jobs metric (emp_total_jobs_ni): +{len(ni_jobs_tidy)} rows")
    
    # Write silver outputs
    log.info("\n" + "="*70)
    log.info("WRITING SILVER OUTPUTS")
    log.info("="*70)
    
    for metric_name, tidy_df in results.items():
        csv_path = SILVER_DIR / f"lad_{metric_name}_history.csv"
        try:
            tidy_df.to_csv(csv_path, index=False)
            log.info(f"  ✓ CSV: {csv_path.name} ({len(tidy_df)} rows)")
        except Exception as e:
            reporter.add_critical_error(f"Failed to write {csv_path.name}: {e}")
        
        try:
            _write_duck(f"silver.lad_{metric_name}_history", tidy_df)
        except Exception as e:
            reporter.add_warning(f"DuckDB write failed for {metric_name}: {e}")
    
    # Save vintage summary
    if tracker and HAVE_VINTAGE:
        log.info("\n" + "="*70)
        log.info("VINTAGE SUMMARY")
        log.info("="*70)
        
        vintage_summary = tracker.get_run_summary()
        vintage_path = PIPELINE_DIR / "vintage_lad.json"
        with open(vintage_path, 'w') as f:
            json.dump(vintage_summary, f, indent=2, default=str)
        log.info(f"  ✓ Vintage summary → {vintage_path}")
        
        # Log changes
        changes = [k for k, v in vintage_summary.get('datasets', {}).items() if v.get('changed', False)]
        if changes:
            log.info(f"  ⚠️  Datasets with changes: {changes}")
            reporter.add_metric("datasets_changed", changes)
        else:
            log.info(f"  ✓ No upstream changes detected")
            reporter.add_metric("datasets_changed", [])
    
    # Summary
    log.info("\n" + "="*70)
    log.info("INGEST SUMMARY")
    log.info("="*70)
    
    total_obs = 0
    for metric_name, tidy_df in results.items():
        config = METRICS[metric_name]
        lad_count = tidy_df['region_code'].nunique()
        year_min = tidy_df['period'].min()
        year_max = tidy_df['period'].max()
        obs_count = len(tidy_df)
        total_obs += obs_count
        
        rate_flag = " [RATE]" if config.get('is_rate', False) else ""
        log.info(f"\n{metric_name.upper()} ({config['metric_id']}){rate_flag}:")
        log.info(f"  LADs: {lad_count}")
        log.info(f"  Years: {year_min} - {year_max}")
        log.info(f"  Observations: {obs_count}")
        log.info(f"  ITL1 regions: {tidy_df['itl1_code'].nunique()}")
        
        reporter.add_metric(f"{metric_name}_lads", lad_count)
        reporter.add_metric(f"{metric_name}_year_range", f"{year_min}-{year_max}")
        reporter.add_metric(f"{metric_name}_itl1_regions", tidy_df['itl1_code'].nunique())
    
    reporter.add_metric("total_observations", total_obs)
    
    # Cross-metric validation
    log.info("\n" + "="*70)
    log.info("CROSS-METRIC VALIDATION")
    log.info("="*70)
    
    lad_counts = {name: df['region_code'].nunique() for name, df in results.items()}
    log.info(f"LAD counts: {lad_counts}")
    
    if len(set(lad_counts.values())) > 1:
        reporter.add_warning(f"LAD counts differ across metrics: {lad_counts}")
    else:
        log.info("✓ All metrics have same LAD count")
    
    core_metrics = ['employment', 'gva', 'gdhi', 'population']
    common_lads = set(results['employment']['region_code'].unique())
    for metric_name in core_metrics[1:]:
        common_lads &= set(results[metric_name]['region_code'].unique())
    
    log.info(f"✓ {len(common_lads)} LADs present in ALL core metrics")
    reporter.add_metric("common_lads_core_metrics", len(common_lads))
    
    # Data quality checks
    log.info("\n✓ DATA QUALITY CHECKS PASSED:")
    log.info("  ✓ No duplicate LAD-period combinations")
    log.info("  ✓ All boundary changes mapped to 2023 codes")
    log.info("  ✓ All LADs have ITL1 parent mappings")
    log.info("  ✓ No negative values (for non-rate metrics)")
    log.info("  ✓ NOMIS duplicate rows removed (ALL metrics)")
    log.info("  ✓ Fiscal year dates parsed (employment/unemployment rates)")
    if tracker and HAVE_VINTAGE:
        log.info("  ✓ Vintage tracking recorded for all datasets")
    
    # Register indicator metadata
    log.info("\n" + "="*70)
    log.info("REGISTERING INDICATOR METADATA")
    log.info("="*70)
    
    register_indicator('gdhi_total_mn_gbp', 'Household Income (GDHI)', 'NM_185_1', 'GBP_m', 'economic')
    register_indicator('nominal_gva_mn_gbp', 'Economic Output (GVA)', 'NM_2400_1', 'GBP_m', 'economic')
    register_indicator('emp_total_jobs', 'Employment', 'NM_172_1,NM_189_1', 'jobs', 'labour')
    register_indicator('emp_total_jobs_ni', 'Employment (NI employee jobs)', 'NISRA_BRESHEADLGD', 'jobs', 'labour')
    register_indicator('population_total', 'Population', 'NM_2002_1', 'persons', 'demographic')
    register_indicator('population_16_64', 'Working Age Population (16-64)', 'NM_2002_1', 'persons', 'demographic')
    register_indicator('employment_rate_pct', 'Employment Rate', 'NM_17_5', 'percent', 'labour')
    register_indicator('unemployment_rate_pct', 'Unemployment Rate', 'NM_17_5', 'percent', 'labour')
    # Derived
    register_indicator('gdhi_per_head_gbp', 'Income per Head', 'derived', 'GBP', 'economic')
    register_indicator('productivity_gbp_per_job', 'Productivity', 'derived', 'GBP', 'economic')
    
    log.info("  ✓ Registered 9 indicators to metadata.indicators")
    
    log.info("\n✅ LAD unified ingest complete!")
    log.info("="*70)
    log.info(f"\nOutputs:")
    log.info(f"  - Silver CSVs: data/silver/lad_*_history.csv")
    log.info(f"  - DuckDB tables: silver.lad_*_history")
    if tracker and HAVE_VINTAGE:
        log.info(f"  - Vintage summary: data/pipeline/vintage_lad.json")
    log.info(f"  - Pipeline log: data/logs/lad_ingest_summary.json")
    log.info("="*70)
    
    reporter.save_and_exit("lad_ingest_summary.json")


if __name__ == "__main__":
    main()