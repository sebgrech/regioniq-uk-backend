#!/usr/bin/env python3
# scripts/ingest/macro_to_duckdb_nomis.py
"""
UK Macro Ingest from NOMIS API → DuckDB + CSV

**V2.3: Added VintageTracker for upstream change detection**
Fetches UK-level macro indicators directly from NOMIS for consistency with ITL1/2/3 pipeline.

Indicators:
- GDHI (total £m)                 → uk_gdhi_total_mn_gbp
- GVA (nominal £m)                → uk_nominal_gva_mn_gbp
- Employment (jobs)               → uk_emp_total_jobs (GB: 2009-2023)
- Population (persons)            → uk_population_total
- Population 16-64 (persons)      → uk_population_16_64
- Employment rate (16-64)         → uk_employment_rate_pct
- Unemployment rate (16+)         → uk_unemployment_rate_pct

Outputs:
- data/raw/macro/*.csv (per-dataset snapshots)
- data/silver/uk_macro_history.csv (unified tidy format)
- data/pipeline/vintage_macro.json (change detection summary)
- warehouse.duckdb:
    bronze.macro_gdhi_raw, bronze.macro_gva_raw, bronze.macro_emp_raw, 
    bronze.macro_pop_raw, bronze.macro_lm_rates_raw
    silver.uk_macro_history
"""

import io
import json
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

# VintageTracker import
try:
    from manifest.data_vintage import VintageTracker
    HAVE_VINTAGE = True
except ImportError:
    HAVE_VINTAGE = False

# -----------------------------
# Indicator Registry
# -----------------------------
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

# -----------------------------
# Configuration
# -----------------------------
UK_CODE = "K02000001"
UK_NAME = "United Kingdom"

RAW_DIR = Path("data/raw/macro")
RAW_DIR.mkdir(parents=True, exist_ok=True)

SILVER_DIR = Path("data/silver")
SILVER_DIR.mkdir(parents=True, exist_ok=True)
SILVER_CSV = SILVER_DIR / "uk_macro_history.csv"

PIPELINE_DIR = Path("data/pipeline")
PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
VINTAGE_SUMMARY_PATH = PIPELINE_DIR / "vintage_macro.json"

LAKE_DIR = Path("data/lake")
LAKE_DIR.mkdir(parents=True, exist_ok=True)
DUCK_PATH = LAKE_DIR / "warehouse.duckdb"

VINTAGE = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("macro_nomis")

# -----------------------------
# DuckDB Helper
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

# -----------------------------
# NOMIS API Helper
# -----------------------------
def fetch_nomis_csv(url: str) -> tuple[pd.DataFrame, str]:
    """
    Fetch CSV from NOMIS API with error handling.
    
    Returns:
        (DataFrame, raw_text) tuple for vintage tracking
    """
    log.info(f"NOMIS → {url[:100]}...")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        raw_text = response.text
        if len(raw_text) < 100:
            log.error(f"Response too short ({len(raw_text)} chars)")
            raise ValueError("Empty or invalid response from NOMIS")
        
        df = pd.read_csv(io.StringIO(raw_text), low_memory=False)
        
        if df.empty:
            log.error("Dataframe is empty after parsing")
            raise ValueError("NOMIS returned empty dataframe")
        
        log.info(f"Fetched {len(df)} rows, {len(df.columns)} columns")
        return df, raw_text
        
    except Exception as e:
        log.error(f"NOMIS fetch failed: {e}")
        raise

# -----------------------------
# Tidy Converter
# -----------------------------
def nomis_to_uk_tidy(
    df_raw: pd.DataFrame,
    metric_id: str,
    unit: str,
    source: str = "NOMIS"
) -> pd.DataFrame:
    """
    Transform NOMIS UK-level data to tidy schema.
    Data is already UK aggregate, just needs formatting.
    """
    # Find value column
    value_col = None
    for col in ["OBS_VALUE", "obs_value", "VALUE", "value", "v4_0", "v4_1"]:
        if col in df_raw.columns:
            value_col = col
            break
    
    if value_col is None:
        raise KeyError(f"Could not find value column. Columns: {df_raw.columns.tolist()}")
    
    # Find date column
    date_col = None
    for col in ["DATE", "date", "Date", "TIME", "time", "Time"]:
        if col in df_raw.columns:
            date_col = col
            break
    
    if date_col is None:
        raise KeyError(f"Could not find date column. Columns: {df_raw.columns.tolist()}")
    
    # Extract year from date (handles "2005-06" format → 2005)
    def extract_year(val):
        if pd.isna(val):
            return None
        s = str(val).strip()
        if "-" in s:
            return int(s.split("-")[0])
        try:
            return int(float(s))
        except:
            return None
    
    df_raw["_year"] = df_raw[date_col].apply(extract_year)
    
    # Ensure numeric value
    df_raw[value_col] = pd.to_numeric(df_raw[value_col], errors="coerce")
    
    # Build tidy frame
    tidy = pd.DataFrame({
        "region_code": UK_CODE,
        "region_name": UK_NAME,
        "region_level": "UK",
        "metric_id": metric_id,
        "period": df_raw["_year"].astype("Int64"),
        "value": df_raw[value_col],
        "unit": unit,
        "freq": "A",
        "source": source,
        "vintage": VINTAGE,
    })
    
    # Drop nulls
    tidy = tidy.dropna(subset=["period", "value"]).reset_index(drop=True)
    
    log.info(f"{metric_id}: {len(tidy)} observations | [{tidy['period'].min()}–{tidy['period'].max()}]")
    
    return tidy

# -----------------------------
# Indicator Fetchers
# -----------------------------

def fetch_gdhi_uk(tracker=None) -> pd.DataFrame:
    """
    Fetch UK GDHI total from NOMIS NM_185_1.
    
    Measure 1 = GDHI total (£m)
    """
    url = (
        "https://www.nomisweb.co.uk/api/v01/dataset/NM_185_1.data.csv"
        "?geography=2092957697"
        "&component_of_gdhi=0"
        "&measure=1"
        "&measures=20100"
    )
    
    df_raw, raw_text = fetch_nomis_csv(url)
    
    # Track vintage
    if tracker:
        periods = df_raw["DATE"].dropna().unique() if "DATE" in df_raw.columns else []
        years = [int(str(p).split("-")[0]) for p in periods if pd.notna(p)]
        min_p, max_p = (min(years), max(years)) if years else (None, None)
        changed = tracker.record("nomis", "NM_185_1", raw_text, n_rows=len(df_raw), min_period=min_p, max_period=max_p)
        log.info(f"GDHI vintage: {'CHANGED' if changed else 'unchanged'}")
    
    # Save raw
    raw_path = RAW_DIR / "nomis_gdhi_raw.csv"
    df_raw.to_csv(raw_path, index=False)
    log.info(f"Saved raw → {raw_path}")
    
    # Convert to tidy
    tidy = nomis_to_uk_tidy(df_raw, "uk_gdhi_total_mn_gbp", "GBP_m")
    
    # Write bronze
    _write_duck("bronze.macro_gdhi_raw", df_raw)
    
    return tidy

def fetch_gva_uk(tracker=None) -> pd.DataFrame:
    """
    Fetch UK nominal GVA from NOMIS NM_2400_1.
    """
    url = (
        "https://www.nomisweb.co.uk/api/v01/dataset/NM_2400_1.data.csv"
        "?geography=2092957720"
        "&cell=0"
        "&measures=20100"
    )
    
    df_raw, raw_text = fetch_nomis_csv(url)
    
    # Track vintage
    if tracker:
        periods = df_raw["DATE"].dropna().unique() if "DATE" in df_raw.columns else []
        years = [int(str(p).split("-")[0]) for p in periods if pd.notna(p)]
        min_p, max_p = (min(years), max(years)) if years else (None, None)
        changed = tracker.record("nomis", "NM_2400_1", raw_text, n_rows=len(df_raw), min_period=min_p, max_period=max_p)
        log.info(f"GVA vintage: {'CHANGED' if changed else 'unchanged'}")
    
    # Save raw
    raw_path = RAW_DIR / "nomis_gva_raw.csv"
    df_raw.to_csv(raw_path, index=False)
    log.info(f"Saved raw → {raw_path}")
    
    # Convert to tidy
    tidy = nomis_to_uk_tidy(df_raw, "uk_nominal_gva_mn_gbp", "GBP_m")
    
    # Write bronze
    _write_duck("bronze.macro_gva_raw", df_raw)
    
    return tidy

def fetch_employment_uk(tracker=None) -> pd.DataFrame:
    """
    Fetch Great Britain employment (total jobs) from NOMIS.
    
    Two datasets to cover full history:
    - NM_172_1: BRES 2009-2015
    - NM_189_1: BRES 2015-2023
    
    NOTE: Employment data only available for Great Britain.
    Northern Ireland not published in these datasets.
    """
    # Pre-2015 data (BRES)
    url_2009_15 = (
        "https://www.nomisweb.co.uk/api/v01/dataset/NM_172_1.data.csv"
        "?geography=2092957698"
        "&industry=37748736"
        "&employment_status=1"
        "&measure=1"
        "&measures=20100"
    )
    
    # Post-2015 data (BRES)
    url_2015_23 = (
        "https://www.nomisweb.co.uk/api/v01/dataset/NM_189_1.data.csv"
        "?geography=2092957698"
        "&industry=37748736"
        "&employment_status=1"
        "&measure=1"
        "&measures=20100"
    )
    
    log.info("EMP: fetching 2009–2015 (NM_172_1)…")
    df1, raw_text1 = fetch_nomis_csv(url_2009_15)
    
    # Track vintage for NM_172_1
    if tracker:
        periods = df1["DATE"].dropna().unique() if "DATE" in df1.columns else []
        years = [int(str(p).split("-")[0]) for p in periods if pd.notna(p)]
        min_p, max_p = (min(years), max(years)) if years else (None, None)
        changed = tracker.record("nomis", "NM_172_1", raw_text1, n_rows=len(df1), min_period=min_p, max_period=max_p)
        log.info(f"EMP (2009-15) vintage: {'CHANGED' if changed else 'unchanged'}")
    
    log.info("EMP: fetching 2015–2023 (NM_189_1)…")
    df2, raw_text2 = fetch_nomis_csv(url_2015_23)
    
    # Track vintage for NM_189_1
    if tracker:
        periods = df2["DATE"].dropna().unique() if "DATE" in df2.columns else []
        years = [int(str(p).split("-")[0]) for p in periods if pd.notna(p)]
        min_p, max_p = (min(years), max(years)) if years else (None, None)
        changed = tracker.record("nomis", "NM_189_1", raw_text2, n_rows=len(df2), min_period=min_p, max_period=max_p)
        log.info(f"EMP (2015-23) vintage: {'CHANGED' if changed else 'unchanged'}")
    
    # Combine both periods
    df_raw = pd.concat([df1, df2], ignore_index=True)
    
    # Save raw (combined)
    raw_path = RAW_DIR / "nomis_emp_raw.csv"
    df_raw.to_csv(raw_path, index=False)
    log.info(f"Saved raw → {raw_path}")
    
    # Convert to tidy (GB labeled as UK for consistency)
    tidy = nomis_to_uk_tidy(df_raw, "uk_emp_total_jobs", "jobs")
    
    # Add coverage note
    tidy["note"] = "Great Britain only (excl. Northern Ireland)"
    
    # Write bronze
    _write_duck("bronze.macro_emp_raw", df_raw)
    
    return tidy

def fetch_population_uk(tracker=None) -> pd.DataFrame:
    """
    Fetch UK population from NOMIS NM_2002_1.
    
    Same dataset as ITL1 for perfect reconciliation.
    
    Gender 0 = All
    c_age 200 = All ages
    c_age 203 = 16-64 (working age)
    """
    # Total population (all ages)
    url_total = (
        "https://www.nomisweb.co.uk/api/v01/dataset/NM_2002_1.data.csv"
        "?geography=2092957697"
        "&gender=0"
        "&c_age=200"
        "&measures=20100"
    )
    
    # Working age population (16-64)
    url_16_64 = (
        "https://www.nomisweb.co.uk/api/v01/dataset/NM_2002_1.data.csv"
        "?geography=2092957697"
        "&gender=0"
        "&c_age=203"
        "&measures=20100"
    )
    
    log.info("POP: fetching total population (all ages)…")
    df_total, raw_text_total = fetch_nomis_csv(url_total)
    
    # Track vintage for total population
    if tracker:
        periods = df_total["DATE"].dropna().unique() if "DATE" in df_total.columns else []
        years = [int(str(p).split("-")[0]) for p in periods if pd.notna(p)]
        min_p, max_p = (min(years), max(years)) if years else (None, None)
        changed = tracker.record("nomis", "NM_2002_1_total", raw_text_total, n_rows=len(df_total), min_period=min_p, max_period=max_p)
        log.info(f"Population (total) vintage: {'CHANGED' if changed else 'unchanged'}")
    
    log.info("POP: fetching working age population (16-64)…")
    df_16_64, raw_text_16_64 = fetch_nomis_csv(url_16_64)
    
    # Track vintage for 16-64 population
    if tracker:
        periods = df_16_64["DATE"].dropna().unique() if "DATE" in df_16_64.columns else []
        years = [int(str(p).split("-")[0]) for p in periods if pd.notna(p)]
        min_p, max_p = (min(years), max(years)) if years else (None, None)
        changed = tracker.record("nomis", "NM_2002_1_16_64", raw_text_16_64, n_rows=len(df_16_64), min_period=min_p, max_period=max_p)
        log.info(f"Population (16-64) vintage: {'CHANGED' if changed else 'unchanged'}")
    
    # Save raw (combined)
    df_raw = pd.concat([df_total, df_16_64], ignore_index=True)
    raw_path = RAW_DIR / "nomis_pop_raw.csv"
    df_raw.to_csv(raw_path, index=False)
    log.info(f"Saved raw → {raw_path}")
    
    # Convert to tidy - total
    tidy_total = nomis_to_uk_tidy(df_total, "uk_population_total", "persons")
    
    # Convert to tidy - 16-64
    tidy_16_64 = nomis_to_uk_tidy(df_16_64, "uk_population_16_64", "persons")
    tidy_16_64["note"] = "Working age population 16-64"
    
    # Combine
    tidy = pd.concat([tidy_total, tidy_16_64], ignore_index=True)
    
    # Write bronze
    _write_duck("bronze.macro_pop_raw", df_raw)
    
    return tidy

def fetch_labour_market_rates_uk(tracker=None) -> pd.DataFrame:
    """
    Fetch UK labour market rates from NOMIS NM_17_5 (Annual Population Survey).
    
    - Employment rate (16-64): variable=45
    - Unemployment rate (16+): variable=83
    
    Date format: "2005-06" → extract year (2005)
    measures=20599 gives the rate %
    
    Uses latestMINUS offsets for ~20 years of history (quarterly data, so MINUS80 = 20 years back).
    """
    date_params = ",".join([
        f"latestMINUS{i}" for i in range(80, -1, -4)
    ] + ["latest"])
    
    # Employment rate (16-64)
    url_emp_rate = (
        f"https://www.nomisweb.co.uk/api/v01/dataset/NM_17_5.data.csv"
        f"?geography=2092957697"
        f"&date={date_params}"
        f"&variable=45"
        f"&measures=20599"
    )
    
    # Unemployment rate (16+)
    url_unemp_rate = (
        f"https://www.nomisweb.co.uk/api/v01/dataset/NM_17_5.data.csv"
        f"?geography=2092957697"
        f"&date={date_params}"
        f"&variable=83"
        f"&measures=20599"
    )
    
    log.info("LM RATES: fetching employment rate (16-64)…")
    df_emp, raw_text_emp = fetch_nomis_csv(url_emp_rate)
    
    # Track vintage for employment rate
    if tracker:
        periods = df_emp["DATE"].dropna().unique() if "DATE" in df_emp.columns else []
        years = [int(str(p).split("-")[0]) for p in periods if pd.notna(p)]
        min_p, max_p = (min(years), max(years)) if years else (None, None)
        changed = tracker.record("nomis", "NM_17_5_emp_rate", raw_text_emp, n_rows=len(df_emp), min_period=min_p, max_period=max_p)
        log.info(f"Employment rate vintage: {'CHANGED' if changed else 'unchanged'}")
    
    log.info("LM RATES: fetching unemployment rate (16+)…")
    df_unemp, raw_text_unemp = fetch_nomis_csv(url_unemp_rate)
    
    # Track vintage for unemployment rate
    if tracker:
        periods = df_unemp["DATE"].dropna().unique() if "DATE" in df_unemp.columns else []
        years = [int(str(p).split("-")[0]) for p in periods if pd.notna(p)]
        min_p, max_p = (min(years), max(years)) if years else (None, None)
        changed = tracker.record("nomis", "NM_17_5_unemp_rate", raw_text_unemp, n_rows=len(df_unemp), min_period=min_p, max_period=max_p)
        log.info(f"Unemployment rate vintage: {'CHANGED' if changed else 'unchanged'}")
    
    # Combine for bronze storage
    df_raw = pd.concat([df_emp, df_unemp], ignore_index=True)
    
    # Save raw
    raw_path = RAW_DIR / "nomis_lm_rates_raw.csv"
    df_raw.to_csv(raw_path, index=False)
    log.info(f"Saved raw → {raw_path}")
    
    # Convert to tidy - employment rate
    tidy_emp = nomis_to_uk_tidy(df_emp, "uk_employment_rate_pct", "pct")
    tidy_emp["note"] = "Employment rate ages 16-64"
    
    # Convert to tidy - unemployment rate
    tidy_unemp = nomis_to_uk_tidy(df_unemp, "uk_unemployment_rate_pct", "pct")
    tidy_unemp["note"] = "Unemployment rate ages 16+"
    
    # APS data is quarterly but dated by year (e.g., "2005-06" = year ending June 2005)
    # We take yearly values; if duplicates exist for same year, keep latest
    tidy_emp = tidy_emp.drop_duplicates(subset=["period"], keep="last")
    tidy_unemp = tidy_unemp.drop_duplicates(subset=["period"], keep="last")
    
    tidy = pd.concat([tidy_emp, tidy_unemp], ignore_index=True)
    
    # Write bronze
    _write_duck("bronze.macro_lm_rates_raw", df_raw)
    
    return tidy

# -----------------------------
# Main Pipeline
# -----------------------------

def main():
    log.info("="*70)
    log.info("UK MACRO INGEST v2.3 - COMPLETE NOMIS + VINTAGE TRACKING")
    log.info("="*70)
    log.info(f"Vintage: {VINTAGE}")
    log.info(f"Strategy: Fetch UK-level data directly from NOMIS")
    log.info("Datasets:")
    log.info("  - GDHI: NM_185_1")
    log.info("  - GVA: NM_2400_1")
    log.info("  - Employment (jobs): NM_172_1 (2009-2015) + NM_189_1 (2015-2023)")
    log.info("  - Population (total): NM_2002_1 c_age=200")
    log.info("  - Population (16-64): NM_2002_1 c_age=203")
    log.info("  - Employment rate (16-64): NM_17_5 variable=45")
    log.info("  - Unemployment rate (16+): NM_17_5 variable=83")
    
    # Initialize vintage tracker
    tracker = None
    if HAVE_VINTAGE:
        tracker = VintageTracker()
        log.info("✓ VintageTracker initialized")
    else:
        log.warning("VintageTracker not available; skipping change detection")
    
    silver_frames = []
    failures = {}
    
    # GDHI (total + per head)
    try:
        log.info("\n--- Fetching GDHI ---")
        gdhi = fetch_gdhi_uk(tracker)
        silver_frames.append(gdhi)
    except Exception as e:
        failures["gdhi"] = str(e)
        log.exception("GDHI failed")
    
    # GVA (nominal)
    try:
        log.info("\n--- Fetching GVA ---")
        gva = fetch_gva_uk(tracker)
        silver_frames.append(gva)
    except Exception as e:
        failures["gva"] = str(e)
        log.exception("GVA failed")
    
    # Employment (jobs) - 2009-2023 via NM_172_1 + NM_189_1
    try:
        log.info("\n--- Fetching Employment (jobs) ---")
        emp = fetch_employment_uk(tracker)
        silver_frames.append(emp)
    except Exception as e:
        failures["employment"] = str(e)
        log.exception("Employment failed")
    
    # Population
    try:
        log.info("\n--- Fetching Population ---")
        pop = fetch_population_uk(tracker)
        silver_frames.append(pop)
    except Exception as e:
        failures["population"] = str(e)
        log.exception("Population failed")
    
    # Labour market rates (employment rate + unemployment rate)
    try:
        log.info("\n--- Fetching Labour Market Rates ---")
        lm_rates = fetch_labour_market_rates_uk(tracker)
        silver_frames.append(lm_rates)
    except Exception as e:
        failures["lm_rates"] = str(e)
        log.exception("Labour market rates failed")
    
    # Check if any succeeded
    if not silver_frames:
        log.error("No indicators successfully ingested. Exiting.")
        raise SystemExit(2)
    
    # Combine all indicators
    silver = pd.concat(silver_frames, ignore_index=True)
    
    # Sort and validate
    silver = silver.dropna(subset=["period", "value"]).reset_index(drop=True)
    silver = silver.sort_values(["metric_id", "period"]).reset_index(drop=True)
    
    # Check for duplicates
    dups = silver.duplicated(subset=["metric_id", "period"], keep=False)
    if dups.any():
        log.warning(f"Found {dups.sum()} duplicate period-metric combinations")
        silver = silver.drop_duplicates(subset=["metric_id", "period"], keep="last")
    
    # Save silver CSV
    silver.to_csv(SILVER_CSV, index=False)
    log.info(f"\n✓ Saved silver CSV → {SILVER_CSV} ({len(silver)} rows)")
    
    # Save to DuckDB
    if HAVE_DUCKDB:
        _write_duck("silver.uk_macro_history", silver)
        log.info(f"✓ Wrote silver.uk_macro_history to {DUCK_PATH}")
    else:
        log.warning("duckdb not installed; skipped DuckDB write")
    
    # Save vintage summary
    if tracker:
        vintage_summary = tracker.get_run_summary()
        vintage_summary["run_timestamp"] = VINTAGE
        vintage_summary["script"] = "macro_to_duckdb_nomis.py"
        vintage_summary["version"] = "2.3"
        
        with open(VINTAGE_SUMMARY_PATH, "w") as f:
            json.dump(vintage_summary, f, indent=2, default=str)
        log.info(f"✓ Saved vintage summary → {VINTAGE_SUMMARY_PATH}")
        
        # Log change summary
        n_changed = sum(1 for ds in vintage_summary.get("datasets", {}).values() if ds.get("changed"))
        n_total = len(vintage_summary.get("datasets", {}))
        log.info(f"Vintage check: {n_changed}/{n_total} datasets changed since last run")
    
    # Summary report
    log.info("\n" + "="*70)
    log.info("INGEST SUMMARY")
    log.info("="*70)
    
    summary = silver.groupby(["metric_id", "freq"]).agg({
        "period": ["min", "max", "count"]
    }).reset_index()
    summary.columns = ["metric_id", "freq", "period_min", "period_max", "obs"]
    
    log.info("\n" + summary.to_string(index=False))
    
    if failures:
        log.warning(f"\nCompleted with {len(failures)} failures:")
        for indicator, error in failures.items():
            log.warning(f"  - {indicator}: {error}")
    else:
        log.info("\n✅ All indicators ingested successfully!")
    
    # Register indicator metadata
    log.info("\n--- Registering indicator metadata ---")
    register_indicator('uk_gdhi_total_mn_gbp', 'Household Income (GDHI)', 'NM_185_1', 'GBP_m', 'economic')
    register_indicator('uk_nominal_gva_mn_gbp', 'Economic Output (GVA)', 'NM_2400_1', 'GBP_m', 'economic')
    register_indicator('uk_emp_total_jobs', 'Employment', 'NM_172_1,NM_189_1', 'jobs', 'labour')
    register_indicator('uk_population_total', 'Population', 'NM_2002_1', 'persons', 'demographic')
    register_indicator('uk_population_16_64', 'Working Age Population (16-64)', 'NM_2002_1', 'persons', 'demographic')
    register_indicator('uk_employment_rate_pct', 'Employment Rate', 'NM_17_5', 'percent', 'labour')
    register_indicator('uk_unemployment_rate_pct', 'Unemployment Rate', 'NM_17_5', 'percent', 'labour')
    # Derived
    register_indicator('uk_gdhi_per_head_gbp', 'Income per Head', 'derived', 'GBP', 'economic')
    register_indicator('uk_productivity_gbp_per_job', 'Productivity', 'derived', 'GBP', 'economic')
    log.info("✓ Indicator metadata registered")
    
    log.info("="*70)

if __name__ == "__main__":
    main()