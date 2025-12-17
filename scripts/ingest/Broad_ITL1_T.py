#!/usr/bin/env python3
# scripts/ingest/unified_itl1_to_duckdb.py
"""
Unified ITL1 ingest → DuckDB (bronze+silver) + CSVs

Indicators:
- Employment (jobs)      → emp_total_jobs (2009-2023, NM_172_1 + NM_189_1)
- GDHI total            → gdhi_total_mn_gbp (NM_185_1)
- Population (all ages)  → population_total (NM_2002_1, c_age=200)
- Population (16-64)     → population_16_64 (NM_2002_1, c_age=203)
- GVA (£m)              → nominal_gva_mn_gbp (NM_2400_1, matches MACRO)
- Employment rate (%)    → employment_rate_pct (NM_17_5, variable=45)
- Unemployment rate (%)  → unemployment_rate_pct (NM_17_5, variable=83)

Outputs:
- data/raw/* per-indicator CSVs
- data/silver/itl1_unified_history.csv  (tidy)
- data/lake/warehouse.duckdb:
    bronze.emp_itl1_raw, bronze.gdhi_itl1_raw, bronze.pop_itl1_raw, bronze.pop_16_64_itl1_raw,
    bronze.gva_itl1_raw, bronze.emp_rate_itl1_raw, bronze.unemp_rate_itl1_raw
    silver.itl1_history
- data/pipeline/vintage_itl1.json  (change detection)
"""

import os
import sys
import json
import logging
import ssl
import urllib.request
from io import StringIO
from pathlib import Path
from datetime import datetime, timezone
from typing import Tuple

import pandas as pd

try:
    import requests  # optional
    HAVE_REQUESTS = True
except Exception:
    HAVE_REQUESTS = False

try:
    import duckdb
except ImportError:
    duckdb = None

try:
    from manifest.data_vintage import VintageTracker
    HAVE_VINTAGE = True
except ImportError:
    VintageTracker = None
    HAVE_VINTAGE = False

# -----------------------------
# Paths
# -----------------------------
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

RAW_EMP_DIR = RAW_DIR / "emp"
RAW_GDHI_DIR = RAW_DIR / "incomes"
RAW_POP_DIR = RAW_DIR / "population"
RAW_GVA_DIR = RAW_DIR / "gva"
RAW_LABOUR_DIR = RAW_DIR / "labour"
for d in (RAW_EMP_DIR, RAW_GDHI_DIR, RAW_POP_DIR, RAW_GVA_DIR, RAW_LABOUR_DIR):
    d.mkdir(parents=True, exist_ok=True)

SILVER_DIR = Path("data/silver")
SILVER_DIR.mkdir(parents=True, exist_ok=True)

LAKE_DIR = Path("data/lake")
LAKE_DIR.mkdir(parents=True, exist_ok=True)
DUCK_PATH = LAKE_DIR / "warehouse.duckdb"

PIPELINE_DIR = Path("data/pipeline")
PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
VINTAGE_SUMMARY_PATH = PIPELINE_DIR / "vintage_itl1.json"

SILVER_CSV = SILVER_DIR / "itl1_unified_history.csv"

VINTAGE = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("unified_ingest")

# -----------------------------
# Helpers
# -----------------------------

def register_indicator(metric_id: str, display_name: str, nomis_datasets: str, unit: str, category: str):
    """Register indicator metadata in DuckDB for downstream reporting."""
    if duckdb is None:
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

def _fetch_nomis(url: str) -> Tuple[str, pd.DataFrame]:
    """
    Fetch NOMIS URL and return (raw_text, DataFrame).
    Captures raw response for vintage hashing before parsing.
    """
    if HAVE_REQUESTS:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        raw_text = resp.text
    else:
        insecure = os.getenv("NOMIS_INSECURE_SSL", "0") == "1"
        ctx = ssl._create_unverified_context() if insecure else ssl.create_default_context()
        with urllib.request.urlopen(url, timeout=120, context=ctx) as r:
            raw_text = r.read().decode("utf-8", errors="replace")
    df = pd.read_csv(StringIO(raw_text))
    return raw_text, df


def _value_col(df: pd.DataFrame) -> str:
    """Best-effort detection of the value column in NOMIS CSVs."""
    candidates = ["OBS_VALUE", "obs_value", "VALUE", "value"]
    for c in candidates:
        if c in df.columns:
            return c
    # Fallback: last numeric column
    for c in df.columns[::-1]:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    raise KeyError("Could not detect value column; columns: " + ", ".join(df.columns))


def _require_cols(df: pd.DataFrame, cols: list, ctx: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{ctx}: missing required columns {missing}")


def _parse_year_from_date(date_val) -> int:
    """
    Parse year from NOMIS DATE formats.
    Handles: '2005', '2005-06', 2005 (int), etc.
    Returns integer year or None.
    """
    if pd.isna(date_val):
        return None
    s = str(date_val).strip()
    # Handle "2005-06" format → take first 4 chars
    if "-" in s:
        s = s.split("-")[0]
    try:
        return int(s)
    except ValueError:
        return None


def _tidy(df: pd.DataFrame, metric_id: str, unit: str, source: str,
          date_col_name: str = "DATE", parse_fiscal_year: bool = False) -> pd.DataFrame:
    """
    Convert a NOMIS-like df to RegionIQ tidy schema:
    region_code, region_name, period, region_level, metric_id, value, unit, freq, source, vintage
    """
    value_col = _value_col(df)
    
    # Map likely column names
    rc = "GEOGRAPHY_CODE" if "GEOGRAPHY_CODE" in df.columns else "GEOGRAPHY"
    rn = "GEOGRAPHY_NAME" if "GEOGRAPHY_NAME" in df.columns else None
    
    # Find date column
    date_col = None
    for candidate in [date_col_name, "DATE", "DATE_CODE", "date"]:
        if candidate in df.columns:
            date_col = candidate
            break
    
    _require_cols(df, [rc], f"{metric_id}")
    if date_col is None:
        raise KeyError(f"{metric_id}: expected a date column in incoming data")

    cols_to_select = [rc, date_col] + ([rn] if rn else []) + [value_col]
    out = df[cols_to_select].copy()
    out.rename(columns={rc: "region_code"}, inplace=True)
    
    if rn:
        out.rename(columns={rn: "region_name"}, inplace=True)
    else:
        out["region_name"] = None

    # Parse period (year) from date column
    if parse_fiscal_year:
        out["period"] = out[date_col].apply(_parse_year_from_date)
    else:
        out["period"] = pd.to_numeric(out[date_col], errors="coerce")
    
    out["period"] = out["period"].astype("Int64")
    out.drop(columns=[date_col], inplace=True)
    
    # Handle value column
    out["value"] = pd.to_numeric(out[value_col], errors="coerce")
    if value_col != "value":
        out.drop(columns=[value_col], inplace=True)

    # Add metadata
    out["region_level"] = "ITL1"
    out["metric_id"] = metric_id
    out["unit"] = unit
    out["freq"] = "A"
    out["source"] = source
    out["vintage"] = VINTAGE

    # Clean rows
    out = out.dropna(subset=["period", "value"]).reset_index(drop=True)

    # Reorder
    cols = ["region_code", "region_name", "region_level", "metric_id",
            "period", "value", "unit", "freq", "source", "vintage"]
    return out[cols]


def _write_duck(table_fullname: str, df: pd.DataFrame):
    """
    Write dataframe to DuckDB with proper schema support.
    Accepts format: 'schema.table' or just 'table' (goes to main)
    """
    if duckdb is None:
        log.warning("duckdb not installed; skipping DuckDB writes.")
        return
    
    if "." in table_fullname:
        schema, table = table_fullname.split(".", 1)
    else:
        schema = "main"
        table = table_fullname
    
    con = duckdb.connect(str(DUCK_PATH))
    try:
        if schema != "main":
            con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        
        con.register("df_tmp", df)
        con.execute(f"CREATE OR REPLACE TABLE {schema}.{table} AS SELECT * FROM df_tmp")
        log.debug(f"Written {df.shape[0]} rows to {schema}.{table}")
    finally:
        con.close()


# -----------------------------
# Indicator fetchers (with vintage tracking)
# -----------------------------

def fetch_emp_itl1(tracker=None) -> pd.DataFrame:
    """
    Employment (jobs), ITL1, 2009–2023 via NOMIS NM_172_1 + NM_189_1.
    """
    url_2009_15 = (
        "https://www.nomisweb.co.uk/api/v01/dataset/NM_172_1.data.csv"
        "?geography=2013265921...2013265931"
        "&industry=37748736"
        "&employment_status=1"
        "&measure=1"
        "&measures=20100"
    )
    url_2015_23 = (
        "https://www.nomisweb.co.uk/api/v01/dataset/NM_189_1.data.csv"
        "?geography=2013265921...2013265931"
        "&industry=37748736"
        "&employment_status=1"
        "&measure=1"
        "&measures=20100"
    )
    
    log.info("EMP: fetching 2009–2015 (NM_172_1)…")
    raw1, df1 = _fetch_nomis(url_2009_15)
    changed1 = None
    if tracker:
        changed1 = tracker.record(
            "nomis", "NM_172_1", raw1,
            n_rows=len(df1),
            min_period=int(df1["DATE"].min()) if "DATE" in df1.columns else None,
            max_period=int(df1["DATE"].max()) if "DATE" in df1.columns else None
        )
    log.info("EMP: NM_172_1 rows=%d, changed=%s", df1.shape[0], changed1)
    
    log.info("EMP: fetching 2015–2023 (NM_189_1)…")
    raw2, df2 = _fetch_nomis(url_2015_23)
    changed2 = None
    if tracker:
        changed2 = tracker.record(
            "nomis", "NM_189_1", raw2,
            n_rows=len(df2),
            min_period=int(df2["DATE"].min()) if "DATE" in df2.columns else None,
            max_period=int(df2["DATE"].max()) if "DATE" in df2.columns else None
        )
    log.info("EMP: NM_189_1 rows=%d, changed=%s", df2.shape[0], changed2)
    
    df = pd.concat([df1, df2], ignore_index=True)
    
    raw_out = RAW_EMP_DIR / "emp_itl1_nomis.csv"
    df.to_csv(raw_out, index=False)
    log.info("EMP: saved raw → %s", raw_out)
    
    tidy = _tidy(df, metric_id="emp_total_jobs", unit="jobs", source="NOMIS")
    _write_duck("bronze.emp_itl1_raw", df)
    return tidy


def fetch_gdhi_itl1(tracker=None) -> pd.DataFrame:
    """GDHI total (£m), ITL1 via NOMIS NM_185_1."""
    url = (
        "https://www.nomisweb.co.uk/api/v01/dataset/NM_185_1.data.csv"
        "?geography=2013265921...2013265932"
        "&component_of_gdhi=0"
        "&measure=1"
        "&measures=20100"
    )
    log.info("GDHI: fetching…")
    raw_text, df = _fetch_nomis(url)
    
    # Extract period range for vintage tracking
    min_period = max_period = None
    if "DATE" in df.columns:
        periods = pd.to_numeric(df["DATE"], errors="coerce").dropna()
        if len(periods) > 0:
            min_period = int(periods.min())
            max_period = int(periods.max())
    
    changed = None
    if tracker:
        changed = tracker.record(
            "nomis", "NM_185_1", raw_text,
            n_rows=len(df),
            min_period=min_period,
            max_period=max_period
        )
    log.info("GDHI: rows=%d, changed=%s", df.shape[0], changed)
    
    raw_out = RAW_GDHI_DIR / "gdhi_itl1_nomis.csv"
    df.to_csv(raw_out, index=False)
    log.info("GDHI: saved raw → %s", raw_out)

    tidy = _tidy(df, "gdhi_total_mn_gbp", unit="GBP_m", source="NOMIS")
    _write_duck("bronze.gdhi_itl1_raw", df)

    return tidy


def fetch_population_itl1(tracker=None) -> pd.DataFrame:
    """Population (persons), ITL1 via NOMIS NM_2002_1."""
    url = (
        "https://www.nomisweb.co.uk/api/v01/dataset/NM_2002_1.data.csv"
        "?geography=2013265921...2013265932"
        "&gender=0"
        "&c_age=200"
        "&measures=20100"
    )
    log.info("POP: fetching…")
    raw_text, df = _fetch_nomis(url)
    
    min_period = max_period = None
    if "DATE" in df.columns:
        periods = pd.to_numeric(df["DATE"], errors="coerce").dropna()
        if len(periods) > 0:
            min_period = int(periods.min())
            max_period = int(periods.max())
    
    changed = None
    if tracker:
        changed = tracker.record(
            "nomis", "NM_2002_1", raw_text,
            n_rows=len(df),
            min_period=min_period,
            max_period=max_period
        )
    log.info("POP: rows=%d, changed=%s", df.shape[0], changed)
    
    raw_out = RAW_POP_DIR / "population_itl1_nomis.csv"
    df.to_csv(raw_out, index=False)
    log.info("POP: saved raw → %s", raw_out)

    tidy = _tidy(df, "population_total", unit="persons", source="NOMIS")
    _write_duck("bronze.pop_itl1_raw", df)
    return tidy


def fetch_population_16_64_itl1(tracker=None) -> pd.DataFrame:
    """Population aged 16-64 (persons), ITL1 via NOMIS NM_2002_1 (c_age=203)."""
    url = (
        "https://www.nomisweb.co.uk/api/v01/dataset/NM_2002_1.data.csv"
        "?geography=2013265921...2013265932"
        "&gender=0"
        "&c_age=203"
        "&measures=20100"
    )
    log.info("POP_16_64: fetching…")
    raw_text, df = _fetch_nomis(url)
    
    min_period = max_period = None
    if "DATE" in df.columns:
        periods = pd.to_numeric(df["DATE"], errors="coerce").dropna()
        if len(periods) > 0:
            min_period = int(periods.min())
            max_period = int(periods.max())
    
    changed = None
    if tracker:
        changed = tracker.record(
            "nomis", "NM_2002_1_16_64", raw_text,
            n_rows=len(df),
            min_period=min_period,
            max_period=max_period
        )
    log.info("POP_16_64: rows=%d, changed=%s", df.shape[0], changed)
    
    raw_out = RAW_POP_DIR / "population_16_64_itl1_nomis.csv"
    df.to_csv(raw_out, index=False)
    log.info("POP_16_64: saved raw → %s", raw_out)

    tidy = _tidy(df, "population_16_64", unit="persons", source="NOMIS")
    _write_duck("bronze.pop_16_64_itl1_raw", df)
    return tidy


def fetch_gva_itl1(tracker=None) -> pd.DataFrame:
    """GVA (£m), ITL1 via NOMIS NM_2400_1."""
    url = (
        "https://www.nomisweb.co.uk/api/v01/dataset/NM_2400_1.data.csv"
        "?geography=2013265921...2013265932"
        "&cell=0"
        "&measures=20100"
    )
    log.info("GVA: fetching…")
    raw_text, df = _fetch_nomis(url)
    
    min_period = max_period = None
    if "DATE" in df.columns:
        periods = pd.to_numeric(df["DATE"], errors="coerce").dropna()
        if len(periods) > 0:
            min_period = int(periods.min())
            max_period = int(periods.max())
    
    changed = None
    if tracker:
        changed = tracker.record(
            "nomis", "NM_2400_1", raw_text,
            n_rows=len(df),
            min_period=min_period,
            max_period=max_period
        )
    log.info("GVA: rows=%d, changed=%s", df.shape[0], changed)
    
    raw_out = RAW_GVA_DIR / "gva_itl1_nomis.csv"
    df.to_csv(raw_out, index=False)
    log.info("GVA: saved raw → %s", raw_out)
    
    tidy = _tidy(df, metric_id="nominal_gva_mn_gbp", unit="GBP_m", source="NOMIS")
    _write_duck("bronze.gva_itl1_raw", df)
    return tidy


def fetch_employment_rate_itl1(tracker=None) -> pd.DataFrame:
    """
    Employment rate (%), ITL1 via NOMIS NM_17_5 (APS).
    variable=45 is employment rate for ages 16-64.
    
    Date format: "2005-06" (fiscal year Apr-Mar) → we take first year (2005).
    """
    url = (
        "https://www.nomisweb.co.uk/api/v01/dataset/NM_17_5.data.csv"
        "?geography=2013265921...2013265932"
        "&date=latestMINUS80,latestMINUS76,latestMINUS72,latestMINUS68,latestMINUS64,"
        "latestMINUS60,latestMINUS56,latestMINUS52,latestMINUS48,latestMINUS44,"
        "latestMINUS40,latestMINUS36,latestMINUS32,latestMINUS28,latestMINUS24,"
        "latestMINUS20,latestMINUS16,latestMINUS12,latestMINUS8,latestMINUS4,latest"
        "&variable=45"
        "&measures=20599"
    )
    log.info("EMP_RATE: fetching…")
    raw_text, df = _fetch_nomis(url)
    
    # Parse fiscal year format for period range
    min_period = max_period = None
    if "DATE" in df.columns:
        years = df["DATE"].apply(_parse_year_from_date).dropna()
        if len(years) > 0:
            min_period = int(years.min())
            max_period = int(years.max())
    
    changed = None
    if tracker:
        changed = tracker.record(
            "nomis", "NM_17_5_emp_rate", raw_text,
            n_rows=len(df),
            min_period=min_period,
            max_period=max_period
        )
    log.info("EMP_RATE: rows=%d, changed=%s", df.shape[0], changed)
    
    raw_out = RAW_LABOUR_DIR / "employment_rate_itl1_nomis.csv"
    df.to_csv(raw_out, index=False)
    log.info("EMP_RATE: saved raw → %s", raw_out)
    
    tidy = _tidy(df, metric_id="employment_rate_pct", unit="pct", source="NOMIS",
                 parse_fiscal_year=True)
    _write_duck("bronze.emp_rate_itl1_raw", df)
    return tidy


def fetch_unemployment_rate_itl1(tracker=None) -> pd.DataFrame:
    """
    Unemployment rate (%), ITL1 via NOMIS NM_17_5 (APS).
    variable=83 is unemployment rate (model-based).
    
    Date format: "2005-06" (fiscal year Apr-Mar) → we take first year (2005).
    """
    url = (
        "https://www.nomisweb.co.uk/api/v01/dataset/NM_17_5.data.csv"
        "?geography=2013265921...2013265932"
        "&date=latestMINUS80,latestMINUS76,latestMINUS72,latestMINUS68,latestMINUS64,"
        "latestMINUS60,latestMINUS56,latestMINUS52,latestMINUS48,latestMINUS44,"
        "latestMINUS40,latestMINUS36,latestMINUS32,latestMINUS28,latestMINUS24,"
        "latestMINUS20,latestMINUS16,latestMINUS12,latestMINUS8,latestMINUS4,latest"
        "&variable=83"
        "&measures=20599"
    )
    log.info("UNEMP_RATE: fetching…")
    raw_text, df = _fetch_nomis(url)
    
    min_period = max_period = None
    if "DATE" in df.columns:
        years = df["DATE"].apply(_parse_year_from_date).dropna()
        if len(years) > 0:
            min_period = int(years.min())
            max_period = int(years.max())
    
    changed = None
    if tracker:
        changed = tracker.record(
            "nomis", "NM_17_5_unemp_rate", raw_text,
            n_rows=len(df),
            min_period=min_period,
            max_period=max_period
        )
    log.info("UNEMP_RATE: rows=%d, changed=%s", df.shape[0], changed)
    
    raw_out = RAW_LABOUR_DIR / "unemployment_rate_itl1_nomis.csv"
    df.to_csv(raw_out, index=False)
    log.info("UNEMP_RATE: saved raw → %s", raw_out)
    
    tidy = _tidy(df, metric_id="unemployment_rate_pct", unit="pct", source="NOMIS",
                 parse_fiscal_year=True)
    _write_duck("bronze.unemp_rate_itl1_raw", df)
    return tidy


# -----------------------------
# Main
# -----------------------------

def main():
    log.info("=== Unified ITL1 ingest starting (vintage=%s) ===", VINTAGE)
    log.info("Datasets:")
    log.info("  - Employment: NM_172_1 (2009-2015) + NM_189_1 (2015-2023)")
    log.info("  - GDHI: NM_185_1")
    log.info("  - Population (all): NM_2002_1 (c_age=200)")
    log.info("  - Population (16-64): NM_2002_1 (c_age=203)")
    log.info("  - GVA: NM_2400_1 (matches MACRO)")
    log.info("  - Employment rate: NM_17_5 (variable=45)")
    log.info("  - Unemployment rate: NM_17_5 (variable=83)")
    
    # Initialize vintage tracker (if available)
    tracker = VintageTracker() if HAVE_VINTAGE else None
    if not HAVE_VINTAGE:
        log.warning("VintageTracker not available; skipping change detection")
    
    silver_frames = []
    failures = {}

    # EMPLOYMENT (jobs)
    try:
        emp = fetch_emp_itl1(tracker)
        silver_frames.append(emp)
        log.info("EMP: tidy rows=%d | years=[%s..%s]",
                 emp.shape[0], emp["period"].min(), emp["period"].max())
    except Exception as e:
        failures["employment"] = str(e)
        log.exception("EMPLOYMENT failed")

    # GDHI
    try:
        gdhi = fetch_gdhi_itl1(tracker)
        silver_frames.append(gdhi)
        log.info("GDHI: tidy rows=%d | years=[%s..%s]",
                 gdhi.shape[0], gdhi["period"].min(), gdhi["period"].max())
    except Exception as e:
        failures["gdhi"] = str(e)
        log.exception("GDHI failed")

    # POPULATION
    try:
        pop = fetch_population_itl1(tracker)
        silver_frames.append(pop)
        log.info("POP: tidy rows=%d | years=[%s..%s]",
                 pop.shape[0], pop["period"].min(), pop["period"].max())
    except Exception as e:
        failures["population"] = str(e)
        log.exception("POPULATION failed")

    # POPULATION 16-64
    try:
        pop_16_64 = fetch_population_16_64_itl1(tracker)
        silver_frames.append(pop_16_64)
        log.info("POP_16_64: tidy rows=%d | years=[%s..%s]",
                 pop_16_64.shape[0], pop_16_64["period"].min(), pop_16_64["period"].max())
    except Exception as e:
        failures["population_16_64"] = str(e)
        log.exception("POPULATION_16_64 failed")

    # GVA
    try:
        gva = fetch_gva_itl1(tracker)
        silver_frames.append(gva)
        log.info("GVA: tidy rows=%d | years=[%s..%s]",
                 gva.shape[0], gva["period"].min(), gva["period"].max())
    except Exception as e:
        failures["gva"] = str(e)
        log.exception("GVA failed")

    # EMPLOYMENT RATE
    try:
        emp_rate = fetch_employment_rate_itl1(tracker)
        silver_frames.append(emp_rate)
        log.info("EMP_RATE: tidy rows=%d | years=[%s..%s]",
                 emp_rate.shape[0], emp_rate["period"].min(), emp_rate["period"].max())
    except Exception as e:
        failures["employment_rate"] = str(e)
        log.exception("EMPLOYMENT_RATE failed")

    # UNEMPLOYMENT RATE
    try:
        unemp_rate = fetch_unemployment_rate_itl1(tracker)
        silver_frames.append(unemp_rate)
        log.info("UNEMP_RATE: tidy rows=%d | years=[%s..%s]",
                 unemp_rate.shape[0], unemp_rate["period"].min(), unemp_rate["period"].max())
    except Exception as e:
        failures["unemployment_rate"] = str(e)
        log.exception("UNEMPLOYMENT_RATE failed")

    # Save vintage summary (if tracker available)
    if tracker:
        vintage_summary = tracker.get_run_summary()
        vintage_summary["run_timestamp"] = VINTAGE
        vintage_summary["failures"] = failures
        
        with open(VINTAGE_SUMMARY_PATH, "w") as f:
            json.dump(vintage_summary, f, indent=2, default=str)
        log.info("Vintage summary saved → %s", VINTAGE_SUMMARY_PATH)
        
        # Log change summary
        datasets_changed = [k for k, v in vintage_summary.get("datasets", {}).items() if v.get("changed")]
        if datasets_changed:
            log.info("NOMIS datasets with changes: %s", datasets_changed)
        else:
            log.info("No NOMIS datasets changed since last run")

    if not silver_frames:
        log.error("No indicators ingested; exiting with failure.")
        sys.exit(2)

    silver = pd.concat(silver_frames, ignore_index=True)

    # Basic sanity
    key_cols = ["region_code", "region_level", "metric_id", "period", "value"]
    missing_any = [c for c in key_cols if c not in silver.columns]
    if missing_any:
        log.error("Silver missing required columns: %s", missing_any)
        sys.exit(3)

    # Save silver CSV
    silver.to_csv(SILVER_CSV, index=False)
    log.info("Saved silver tidy CSV → %s (rows=%d)", SILVER_CSV, silver.shape[0])

    # Save silver to DuckDB
    if duckdb is not None:
        _write_duck("silver.itl1_history", silver)
        log.info("Wrote silver.itl1_history to %s", DUCK_PATH)
    else:
        log.warning("duckdb not installed; skipped writing silver to DuckDB.")

    # Register indicator metadata
    register_indicator('gdhi_total_mn_gbp', 'Household Income (GDHI)', 'NM_185_1', 'GBP_m', 'economic')
    register_indicator('nominal_gva_mn_gbp', 'Economic Output (GVA)', 'NM_2400_1', 'GBP_m', 'economic')
    register_indicator('emp_total_jobs', 'Employment', 'NM_172_1,NM_189_1', 'jobs', 'labour')
    register_indicator('population_total', 'Population', 'NM_2002_1', 'persons', 'demographic')
    register_indicator('population_16_64', 'Working Age Population (16-64)', 'NM_2002_1', 'persons', 'demographic')
    register_indicator('employment_rate_pct', 'Employment Rate', 'NM_17_5', 'percent', 'labour')
    register_indicator('unemployment_rate_pct', 'Unemployment Rate', 'NM_17_5', 'percent', 'labour')
    # Derived indicators
    register_indicator('gdhi_per_head_gbp', 'Income per Head', 'derived', 'GBP', 'economic')
    register_indicator('productivity_gbp_per_job', 'Productivity', 'derived', 'GBP', 'economic')
    log.info("Registered indicator metadata")

    # Final summary
    by_metric = (
        silver.groupby("metric_id")["value"]
        .count()
        .rename("rows")
        .to_frame()
        .reset_index()
        .sort_values("metric_id")
    )
    log.info("Ingest complete. Rows by metric:\n%s", by_metric.to_string(index=False))

    if failures:
        log.warning("Completed with %d indicator failures: %s", len(failures), failures)
    else:
        log.info("All indicators ingested successfully.")


if __name__ == "__main__":
    main()