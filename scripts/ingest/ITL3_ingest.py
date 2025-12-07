#!/usr/bin/env python3
# scripts/ingest/unified_itl3_to_duckdb.py
"""
Unified ITL3 ingest → DuckDB (bronze+silver) + CSVs

Indicators:
- Employment (jobs)      → emp_total_jobs (2009-2024, NM_172_1 + NM_189_1)
- GDHI total / per head  → gdhi_total_mn_gbp, gdhi_per_head_gbp (NM_185_1)
- GVA (£m)              → nominal_gva_mn_gbp (NM_2400_1)

Outputs:
- data/raw/* per-indicator CSVs
- data/silver/itl3_unified_history.csv  (tidy)
- data/lake/warehouse.duckdb:
    bronze.emp_itl3_raw, bronze.gdhi_itl3_raw, bronze.gva_itl3_raw
    silver.itl3_history

V1.1: Production-hardened with NOMIS edge case handling
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
RAW_GVA_DIR = RAW_DIR / "gva"
for d in (RAW_EMP_DIR, RAW_GDHI_DIR, RAW_GVA_DIR):
    d.mkdir(parents=True, exist_ok=True)

SILVER_DIR = Path("data/silver")
SILVER_DIR.mkdir(parents=True, exist_ok=True)

LAKE_DIR = Path("data/lake")
LAKE_DIR.mkdir(parents=True, exist_ok=True)
DUCK_PATH = LAKE_DIR / "warehouse.duckdb"

SILVER_CSV = SILVER_DIR / "itl3_unified_history.csv"

VINTAGE = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("itl3_ingest")

# -----------------------------
# Helpers
# -----------------------------

def _value_col(df: pd.DataFrame) -> str:
    """
    Robust detection of value column in NOMIS CSVs.
    Handles OBS_VALUE, OBS_VALUE (r), obs_value variants.
    """
    # First: explicit OBS_VALUE variants (handles revision flags like "(r)")
    for c in df.columns:
        if c.upper().startswith("OBS_VALUE"):
            return c
    
    # Second: common alternatives
    candidates = ["obs_value", "VALUE", "value"]
    for c in candidates:
        if c in df.columns:
            return c
    
    # Fallback: last numeric column
    for c in df.columns[::-1]:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    
    raise KeyError("Could not detect value column in dataframe; columns: " + ", ".join(df.columns))

def _require_cols(df: pd.DataFrame, cols: list, ctx: str):
    """Validate required columns exist"""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{ctx}: missing required columns {missing}")

def _tidy(df: pd.DataFrame, metric_id: str, unit: str, source: str) -> pd.DataFrame:
    """
    Convert NOMIS df to RegionIQ tidy schema.
    
    Schema:
    region_code, region_name, period, region_level, metric_id, value, unit, freq, source, vintage
    """
    value_col = _value_col(df)
    
    # Geography columns
    rc = "GEOGRAPHY_CODE" if "GEOGRAPHY_CODE" in df.columns else "GEOGRAPHY"
    rn = "GEOGRAPHY_NAME" if "GEOGRAPHY_NAME" in df.columns else None
    
    # Date column (NOMIS inconsistent naming)
    date_col = None
    for c in ["DATE", "DATE_NAME", "date_name", "date"]:
        if c in df.columns:
            date_col = c
            break
    
    _require_cols(df, [rc], f"{metric_id}")
    if date_col is None:
        raise KeyError(f"{metric_id}: expected a DATE/DATE_NAME column (year) in incoming data")

    # Build output
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
    out["region_level"] = "ITL3"
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

# -----------------------------
# Indicator fetchers
# -----------------------------

def fetch_emp_itl3() -> pd.DataFrame:
    """
    Employment (jobs), ITL3, 2009–2024 via NOMIS NM_172_1 + NM_189_1.
    
    Two datasets to cover full history:
    - NM_172_1: BRES 2009-2015
    - NM_189_1: BRES 2015-2024
    
    Handles 2015 overlap by deduplicating (keeps latest).
    """
    url_2009_15 = (
        "https://www.nomisweb.co.uk/api/v01/dataset/NM_172_1.data.csv"
        "?geography=1912602625...1912602758"
        "&industry=37748736"
        "&employment_status=1"
        "&measure=1"
        "&measures=20100"
    )
    url_2015_24 = (
        "https://www.nomisweb.co.uk/api/v01/dataset/NM_189_1.data.csv"
        "?geography=1795162113...1795162280"
        "&industry=37748736"
        "&employment_status=1"
        "&measure=1"
        "&measures=20100"
    )
    
    log.info("EMP ITL3: fetching 2009–2015 (NM_172_1)…")
    df1 = pd.read_csv(url_2009_15)
    log.info("EMP ITL3: NM_172_1 rows=%d", df1.shape[0])
    
    log.info("EMP ITL3: fetching 2015–2024 (NM_189_1)…")
    df2 = pd.read_csv(url_2015_24)
    log.info("EMP ITL3: NM_189_1 rows=%d", df2.shape[0])
    
    # Concatenate and deduplicate 2015 overlap
    df = pd.concat([df1, df2], ignore_index=True)
    
    # Find geography and date columns for deduplication
    geo_col = "GEOGRAPHY_CODE" if "GEOGRAPHY_CODE" in df.columns else "GEOGRAPHY"
    date_col = None
    for c in ["DATE", "DATE_NAME", "date_name", "date"]:
        if c in df.columns:
            date_col = c
            break
    
    if geo_col and date_col:
        before_dedup = len(df)
        df = df.drop_duplicates(subset=[geo_col, date_col], keep="last")
        after_dedup = len(df)
        if before_dedup > after_dedup:
            log.info("EMP ITL3: deduplicated %d rows (likely 2015 overlap)", before_dedup - after_dedup)
    
    # Save raw for debugging
    raw_out = RAW_EMP_DIR / "emp_itl3_nomis.csv"
    df.to_csv(raw_out, index=False)
    log.info("EMP ITL3: saved raw → %s", raw_out)
    
    # Tidy
    tidy = _tidy(df, metric_id="emp_total_jobs", unit="jobs", source="NOMIS")
    
    # Write bronze table
    _write_duck("bronze.emp_itl3_raw", df)
    
    return tidy

def fetch_gdhi_itl3() -> pd.DataFrame:
    """
    GDHI total (£m) and per head (£), ITL3 via NOMIS NM_185_1.
    
    Handles both MEASURE and MEASURE_NAME columns.
    """
    url = (
        "https://www.nomisweb.co.uk/api/v01/dataset/NM_185_1.data.csv"
        "?geography=1765801985...1765802166"
        "&component_of_gdhi=0"
        "&measure=1,2"
        "&measures=20100"
    )
    
    log.info("GDHI ITL3: fetching…")
    df = pd.read_csv(url)
    raw_out = RAW_GDHI_DIR / "gdhi_itl3_nomis.csv"
    df.to_csv(raw_out, index=False)
    log.info("GDHI ITL3: saved raw → %s", raw_out)

    # Find measure column (NOMIS inconsistent naming)
    measure_col = None
    if "MEASURE" in df.columns:
        measure_col = "MEASURE"
    elif "MEASURE_NAME" in df.columns:
        measure_col = "MEASURE_NAME"
    else:
        raise KeyError("GDHI ITL3: expected MEASURE or MEASURE_NAME column")
    
    # Normalize measure col to int if possible
    try:
        df[measure_col] = pd.to_numeric(df[measure_col], errors="coerce").astype("Int64")
    except Exception:
        pass

    # Total (£m) - measure 1
    df_total = df[df[measure_col] == 1].copy()
    gdhi_total = _tidy(df_total, "gdhi_total_mn_gbp", unit="GBP_m", source="NOMIS")

    # Per head (£) - measure 2
    df_ph = df[df[measure_col] == 2].copy()
    gdhi_ph = _tidy(df_ph, "gdhi_per_head_gbp", unit="GBP", source="NOMIS")

    _write_duck("bronze.gdhi_itl3_raw", df)

    return pd.concat([gdhi_total, gdhi_ph], ignore_index=True)

def fetch_gva_itl3() -> pd.DataFrame:
    """GVA (£m), ITL3 via NOMIS NM_2400_1."""
    url = (
        "https://www.nomisweb.co.uk/api/v01/dataset/NM_2400_1.data.csv"
        "?geography=1765801985...1765802166"
        "&cell=0"
        "&measures=20100"
    )
    
    log.info("GVA ITL3: fetching…")
    df = pd.read_csv(url)
    raw_out = RAW_GVA_DIR / "gva_itl3_nomis.csv"
    df.to_csv(raw_out, index=False)
    log.info("GVA ITL3: saved raw → %s", raw_out)
    
    # Tidy
    tidy = _tidy(df, metric_id="nominal_gva_mn_gbp", unit="GBP_m", source="NOMIS")
    
    # Write bronze table
    _write_duck("bronze.gva_itl3_raw", df)
    
    return tidy

# -----------------------------
# Main
# -----------------------------

def main():
    log.info("="*70)
    log.info("UNIFIED ITL3 INGEST v1.1 - PRODUCTION HARDENED")
    log.info("="*70)
    log.info(f"Vintage: {VINTAGE}")
    log.info("Datasets:")
    log.info("  - Employment: NM_172_1 (2009-2015) + NM_189_1 (2015-2024)")
    log.info("  - GDHI: NM_185_1 (total + per head)")
    log.info("  - GVA: NM_2400_1")
    log.info("  - Population: SKIPPED (as requested)")
    log.info("")
    
    silver_frames = []
    failures = {}

    # EMPLOYMENT
    try:
        emp = fetch_emp_itl3()
        silver_frames.append(emp)
        log.info("✓ EMP ITL3: %d rows | years=[%s..%s]",
                 emp.shape[0],
                 emp["period"].min(),
                 emp["period"].max())
    except Exception as e:
        failures["employment"] = str(e)
        log.exception("✗ EMPLOYMENT ITL3 failed")

    # GDHI
    try:
        gdhi = fetch_gdhi_itl3()
        silver_frames.append(gdhi)
        log.info("✓ GDHI ITL3: %d rows | metrics=%s",
                 gdhi.shape[0],
                 sorted(gdhi["metric_id"].unique()))
    except Exception as e:
        failures["gdhi"] = str(e)
        log.exception("✗ GDHI ITL3 failed")

    # GVA
    try:
        gva = fetch_gva_itl3()
        silver_frames.append(gva)
        log.info("✓ GVA ITL3: %d rows | years=[%s..%s]",
                 gva.shape[0],
                 gva["period"].min(),
                 gva["period"].max())
    except Exception as e:
        failures["gva"] = str(e)
        log.exception("✗ GVA ITL3 failed")

    if not silver_frames:
        log.error("No indicators ingested; exiting with failure.")
        sys.exit(2)

    # Combine all indicators
    silver = pd.concat(silver_frames, ignore_index=True)

    # Validate schema
    key_cols = ["region_code","region_level","metric_id","period","value"]
    missing_any = [c for c in key_cols if c not in silver.columns]
    if missing_any:
        log.error("Silver missing required columns: %s", missing_any)
        sys.exit(3)

    # Sort and deduplicate
    silver = silver.sort_values(["metric_id", "region_code", "period"]).reset_index(drop=True)
    dups_before = len(silver)
    silver = silver.drop_duplicates(subset=["region_code", "metric_id", "period"], keep="last")
    dups_after = len(silver)
    if dups_before > dups_after:
        log.warning("Removed %d duplicate rows", dups_before - dups_after)

    # Save silver CSV
    silver.to_csv(SILVER_CSV, index=False)
    log.info("✓ Saved silver CSV → %s (%d rows)", SILVER_CSV, silver.shape[0])

    # Save silver to DuckDB
    if duckdb is not None:
        _write_duck("silver.itl3_history", silver)
        log.info("✓ Wrote silver.itl3_history to %s", DUCK_PATH)
    else:
        log.warning("duckdb not installed; skipped writing silver to DuckDB.")

    # Summary report
    log.info("")
    log.info("="*70)
    log.info("INGEST SUMMARY")
    log.info("="*70)
    
    summary = (
        silver.groupby(["metric_id", "freq"])
        .agg({"period": ["min", "max", "count"]})
        .reset_index()
    )
    summary.columns = ["metric_id", "freq", "period_min", "period_max", "obs"]
    
    log.info("")
    log.info(summary.to_string(index=False))
    log.info("")

    if failures:
        log.warning("Completed with %d indicator failures:", len(failures))
        for indicator, error in failures.items():
            log.warning("  - %s: %s", indicator, error)
    else:
        log.info("✅ All indicators ingested successfully!")
    
    log.info("="*70)

if __name__ == "__main__":
    main()