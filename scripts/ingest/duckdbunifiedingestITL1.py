#!/usr/bin/env python3
# scripts/ingest/unified_itl1_to_duckdb.py
"""
Unified ITL1 ingest → DuckDB (bronze+silver) + CSVs

Indicators:
- Employment (jobs)      → emp_total_jobs
- GDHI total / per head  → gdhi_total_mn_gbp, gdhi_per_head_gbp
- Population (all ages)  → population_total
- Nominal GVA (total)    → nominal_gva_mn_gbp  (expects cleaned CSV from clean_gva.py)

Outputs:
- data/raw/* per-indicator CSVs
- data/silver/itl1_unified_history.csv  (tidy)
- data/lake/bronze.duckdb with:
    bronze.emp_itl1_raw, bronze.gdhi_itl1_raw, bronze.pop_itl1_raw, bronze.gva_itl1_raw
    silver.itl1_history
"""

import os
import sys
import io
import time
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

try:
    import duckdb
except ImportError:
    duckdb = None

# -----------------------------
# Paths
# -----------------------------
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

RAW_EMP_DIR = RAW_DIR / "emp"
RAW_GDHI_DIR = RAW_DIR / "incomes"
RAW_POP_DIR = RAW_DIR / "population"
RAW_GVA_DIR = RAW_DIR / "gva"
for d in (RAW_EMP_DIR, RAW_GDHI_DIR, RAW_POP_DIR, RAW_GVA_DIR):
    d.mkdir(parents=True, exist_ok=True)

SILVER_DIR = Path("data/silver")
SILVER_DIR.mkdir(parents=True, exist_ok=True)

LAKE_DIR = Path("data/lake")
LAKE_DIR.mkdir(parents=True, exist_ok=True)
DUCK_PATH = LAKE_DIR / "warehouse.duckdb"

SILVER_CSV = SILVER_DIR / "itl1_unified_history.csv"

# Optional cleaned GVA file produced by your clean_gva.py step
# Expected columns: GEOGRAPHY_CODE, GEOGRAPHY_NAME, DATE (YYYY), VALUE (in £m)
CLEAN_GVA_PATH = Path("data/clean/gva/gva_ITL1_long.csv")

# Fixed deprecation warning - use timezone-aware datetime
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
    raise KeyError("Could not detect value column in dataframe; columns: " + ", ".join(df.columns))

def _require_cols(df: pd.DataFrame, cols: list, ctx: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{ctx}: missing required columns {missing}")

def _tidy(df: pd.DataFrame, metric_id: str, unit: str, source: str) -> pd.DataFrame:
    """
    Convert a NOMIS-like df to RegionIQ tidy schema:
    region_code, region_name, period, region_level, metric_id, value, unit, freq, source, vintage
    """
    value_col = _value_col(df)
    # map likely column names
    rc = "GEOGRAPHY_CODE" if "GEOGRAPHY_CODE" in df.columns else "GEOGRAPHY"
    rn = "GEOGRAPHY_NAME" if "GEOGRAPHY_NAME" in df.columns else None
    date_col = "DATE" if "DATE" in df.columns else None

    _require_cols(df, [rc], f"{metric_id}")
    if date_col is None:
        raise KeyError(f"{metric_id}: expected a DATE column (year) in incoming data")

    out = df[[rc, date_col] + ([rn] if rn else []) + [value_col]].copy()
    out.rename(columns={rc: "region_code", date_col: "period"}, inplace=True)
    if rn:
        out.rename(columns={rn: "region_name"}, inplace=True)
    else:
        out["region_name"] = None

    # Standardize types
    out["period"] = pd.to_numeric(out["period"], errors="coerce").astype("Int64")
    out["value"] = pd.to_numeric(out[value_col], errors="coerce")
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
    
    # Parse schema.table format
    if "." in table_fullname:
        schema, table = table_fullname.split(".", 1)
    else:
        schema = "main"
        table = table_fullname
    
    con = duckdb.connect(str(DUCK_PATH))
    try:
        # Create schema if needed (except main which always exists)
        if schema != "main":
            con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        
        # Register dataframe and create/replace table in the proper schema
        con.register("df_tmp", df)
        con.execute(f"CREATE OR REPLACE TABLE {schema}.{table} AS SELECT * FROM df_tmp")
        log.debug(f"Written {df.shape[0]} rows to {schema}.{table}")
    finally:
        con.close()

def _append_duck(table_fullname: str, df: pd.DataFrame):
    """
    Append dataframe to DuckDB table with proper schema support.
    Accepts format: 'schema.table' or just 'table' (goes to main)
    """
    if duckdb is None:
        log.warning("duckdb not installed; skipping DuckDB writes.")
        return
    
    # Parse schema.table format
    if "." in table_fullname:
        schema, table = table_fullname.split(".", 1)
    else:
        schema = "main"
        table = table_fullname
    
    con = duckdb.connect(str(DUCK_PATH))
    try:
        # Create schema if needed (except main which always exists)
        if schema != "main":
            con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        
        # Create table if not exists (using df as template, but limit 0 for schema only)
        con.register("df", df)
        con.execute(f"CREATE TABLE IF NOT EXISTS {schema}.{table} AS SELECT * FROM df LIMIT 0")
        
        # Now insert the actual data
        con.register("df_tmp", df)
        con.execute(f"INSERT INTO {schema}.{table} SELECT * FROM df_tmp")
        log.debug(f"Appended {df.shape[0]} rows to {schema}.{table}")
    finally:
        con.close()

# -----------------------------
# Indicator fetchers
# -----------------------------

def fetch_emp_itl1() -> pd.DataFrame:
    """Employment (jobs), ITL1, 2009–2023 via NOMIS NM_172_1 + NM_189_1."""
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
    log.info("EMP: fetching 2009–2015…")
    df1 = pd.read_csv(url_2009_15)
    log.info("EMP: rows=%d", df1.shape[0])
    log.info("EMP: fetching 2015–2023…")
    df2 = pd.read_csv(url_2015_23)
    log.info("EMP: rows=%d", df2.shape[0])
    df = pd.concat([df1, df2], ignore_index=True)
    # Save raw for debugging
    raw_out = RAW_EMP_DIR / "emp_itl1_nomis.csv"
    df.to_csv(raw_out, index=False)
    log.info("EMP: saved raw → %s", raw_out)
    # Tidy
    tidy = _tidy(df, metric_id="emp_total_jobs", unit="jobs", source="NOMIS")
    # Write bronze table (now with proper schema)
    _write_duck("bronze.emp_itl1_raw", df)
    return tidy

def fetch_gdhi_itl1() -> pd.DataFrame:
    """GDHI total (£m) and per head (£), ITL1 via NOMIS NM_185_1."""
    url = (
        "https://www.nomisweb.co.uk/api/v01/dataset/NM_185_1.data.csv"
        "?geography=2013265921...2013265932"
        "&component_of_gdhi=0"
        "&measure=1,2"
        "&measures=20100"
    )
    log.info("GDHI: fetching…")
    df = pd.read_csv(url)
    raw_out = RAW_GDHI_DIR / "gdhi_itl1_nomis.csv"
    df.to_csv(raw_out, index=False)
    log.info("GDHI: saved raw → %s", raw_out)

    # Split measures: 1=GDHI (£m), 2=per head (£)
    _require_cols(df, ["MEASURE"], "GDHI")
    measure_col = "MEASURE"
    # Normalize measure col to int if possible
    try:
        df[measure_col] = pd.to_numeric(df[measure_col], errors="coerce").astype("Int64")
    except Exception:
        pass

    # Total (£m)
    df_total = df[df[measure_col] == 1].copy()
    gdhi_total = _tidy(df_total, "gdhi_total_mn_gbp", unit="GBP_m", source="NOMIS")

    # Per head (£)
    df_ph = df[df[measure_col] == 2].copy()
    gdhi_ph = _tidy(df_ph, "gdhi_per_head_gbp", unit="GBP", source="NOMIS")

    _write_duck("bronze.gdhi_itl1_raw", df)

    return pd.concat([gdhi_total, gdhi_ph], ignore_index=True)

def fetch_population_itl1() -> pd.DataFrame:
    """Population (persons), ITL1 via NOMIS NM_31_1; filter age=0 (All ages)."""
    url = (
        "https://www.nomisweb.co.uk/api/v01/dataset/NM_31_1.data.csv"
        "?geography=2013265921...2013265932"
        "&sex=7"
        "&age=0,24,22,25,20,21"
        "&measures=20100"
    )
    log.info("POP: fetching…")
    df = pd.read_csv(url)
    raw_out = RAW_POP_DIR / "population_nuts1_nomis.csv"
    df.to_csv(raw_out, index=False)
    log.info("POP: saved raw → %s", raw_out)

    # Filter to age=0 (All ages) only for total population
    age_col = "AGE" if "AGE" in df.columns else "age"
    if age_col not in df.columns:
        raise KeyError("Population: AGE column not found.")
    df_total = df[df[age_col].astype(str).isin(["0", "All ages", "all ages"])].copy()
    if df_total.empty:
        log.warning("POP: could not find age=0 rows; using full dataset as fallback.")
        df_total = df.copy()

    tidy = _tidy(df_total, "population_total", unit="persons", source="NOMIS")
    _write_duck("bronze.pop_itl1_raw", df)
    return tidy

def load_clean_gva_itl1() -> pd.DataFrame:
    """
    Load cleaned GVA ITL1 long CSV and map to unified schema.
    Expects columns: region, region_code, year, metric, value
    """
    if not CLEAN_GVA_PATH.exists():
        log.warning("GVA: cleaned file not found (%s). Skipping GVA.", CLEAN_GVA_PATH)
        return pd.DataFrame(columns=[
            "region_code","region_name","region_level","metric_id","period",
            "value","unit","freq","source","vintage"
        ])

    log.info("GVA: loading cleaned CSV…")
    df = pd.read_csv(CLEAN_GVA_PATH)

    # Verify expected columns
    required = {"region", "region_code", "year", "metric", "value"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"GVA cleaned file missing expected columns: {sorted(missing)}")

    # Map metrics to standard IDs and units
    # Adjust these mappings based on what's actually in your metric column
    metric_mapping = {
        "nominal_gva_mn_gbp": ("nominal_gva_mn_gbp", "GBP_m"),
        "Nominal GVA": ("nominal_gva_mn_gbp", "GBP_m"),
        "chained_gva_mn_gbp": ("chained_gva_mn_gbp", "GBP_chained_m"),
        # Add more mappings if your data has other metric values
    }

    # Create tidy dataframe matching unified schema
    tidy_rows = []
    for _, row in df.iterrows():
        metric_str = str(row["metric"]).strip()
        
        # Get metric_id and unit from mapping, default to using metric as-is
        if metric_str in metric_mapping:
            metric_id, unit = metric_mapping[metric_str]
        else:
            # Default: use the metric string as ID, assume GBP_m for unit
            metric_id = metric_str
            unit = "GBP_m"
        
        tidy_rows.append({
            "region_code": str(row["region_code"]).strip(),
            "region_name": str(row["region"]).strip(),
            "region_level": "ITL1",
            "metric_id": metric_id,
            "period": pd.to_numeric(row["year"], errors="coerce"),
            "value": pd.to_numeric(row["value"], errors="coerce"),
            "unit": unit,
            "freq": "A",
            "source": "ONS",
            "vintage": VINTAGE,
        })
    
    tidy = pd.DataFrame(tidy_rows)
    
    # Clean: remove rows with missing period or value
    tidy = tidy.dropna(subset=["period", "value"]).reset_index(drop=True)
    
    # Convert period to Int64 after dropna
    tidy["period"] = tidy["period"].astype("Int64")
    
    # Save a copy for debugging
    raw_out = RAW_GVA_DIR / "gva_itl1_from_clean.copy.csv"
    df.to_csv(raw_out, index=False)
    log.info("GVA: saved copy → %s", raw_out)
    log.info("GVA: created tidy rows=%d", tidy.shape[0])

    # Write original cleaned data to bronze for traceability
    _write_duck("bronze.gva_itl1_raw", df)
    
    return tidy

# -----------------------------
# Main
# -----------------------------

def main():
    log.info("=== Unified ITL1 ingest starting (vintage=%s) ===", VINTAGE)
    silver_frames = []
    failures = {}

    # EMPLOYMENT
    try:
        emp = fetch_emp_itl1()
        silver_frames.append(emp)
        log.info("EMP: tidy rows=%d | years=[%s..%s]",
                 emp.shape[0],
                 emp["period"].min(),
                 emp["period"].max())
    except Exception as e:
        failures["employment"] = str(e)
        log.exception("EMPLOYMENT failed")

    # GDHI
    try:
        gdhi = fetch_gdhi_itl1()
        silver_frames.append(gdhi)
        log.info("GDHI: tidy rows=%d | metrics=%s",
                 gdhi.shape[0],
                 sorted(gdhi["metric_id"].unique()))
    except Exception as e:
        failures["gdhi"] = str(e)
        log.exception("GDHI failed")

    # POPULATION
    try:
        pop = fetch_population_itl1()
        silver_frames.append(pop)
        log.info("POP: tidy rows=%d", pop.shape[0])
    except Exception as e:
        failures["population"] = str(e)
        log.exception("POPULATION failed")

    # GVA (cleaned)
    try:
        gva = load_clean_gva_itl1()
        if not gva.empty:
            silver_frames.append(gva)
            log.info("GVA: tidy rows=%d", gva.shape[0])
    except Exception as e:
        failures["gva"] = str(e)
        log.exception("GVA failed")

    if not silver_frames:
        log.error("No indicators ingested; exiting with failure.")
        sys.exit(2)

    silver = pd.concat(silver_frames, ignore_index=True)

    # Basic sanity
    key_cols = ["region_code","region_level","metric_id","period","value"]
    missing_any = [c for c in key_cols if c not in silver.columns]
    if missing_any:
        log.error("Silver missing required columns: %s", missing_any)
        sys.exit(3)

    # Save silver CSV
    silver.to_csv(SILVER_CSV, index=False)
    log.info("Saved silver tidy CSV → %s (rows=%d)", SILVER_CSV, silver.shape[0])

    # Save silver to DuckDB (now with proper schema)
    if duckdb is not None:
        _write_duck("silver.itl1_history", silver)
        log.info("Wrote silver.itl1_history to %s", DUCK_PATH)
    else:
        log.warning("duckdb not installed; skipped writing silver to DuckDB.")

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
        # Exit code 0 but with warnings—caller can decide to alert
    else:
        log.info("All indicators ingested successfully.")

if __name__ == "__main__":
    main()