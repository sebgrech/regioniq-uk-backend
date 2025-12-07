#!/usr/bin/env python3
# scripts/ingest/ITL2_ingest.py
"""
Unified ITL2 ingest → DuckDB (bronze+silver) + CSVs
Canonicalised to 2025 ITL2 geometry using ONS 2021→2025 lookup.

Indicators:
- Employment (jobs)      → emp_total_jobs (2015–2024, NM_189_1 ITL2 only)
- GDHI total / per head  → gdhi_total_mn_gbp, gdhi_per_head_gbp (NM_185_1)
- GVA (£m)               → nominal_gva_mn_gbp (NM_2400_1)
- Population             → SKIPPED (will aggregate from LA level later)

Reference:
- data/reference/itl2_2021_2025_lookup.csv
  Columns: ITL221CD, ITL221NM, ITL225CD, ITL225NM, ObjectId

Outputs:
- data/raw/itl2/* per-indicator CSVs
- data/silver/itl2_unified_history.csv  (tidy, canonical ITL2=2025)
- data/lake/warehouse.duckdb:
    bronze.emp_itl2_raw, bronze.gdhi_itl2_raw, bronze.gva_itl2_raw
    silver.itl2_history
"""

import sys
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
RAW_DIR = Path("data/raw/itl2")
RAW_DIR.mkdir(parents=True, exist_ok=True)

RAW_EMP_DIR = RAW_DIR / "emp"
RAW_GDHI_DIR = RAW_DIR / "incomes"
RAW_GVA_DIR = RAW_DIR / "gva"
for d in (RAW_EMP_DIR, RAW_GDHI_DIR, RAW_GVA_DIR):
    d.mkdir(parents=True, exist_ok=True)

SILVER_DIR = Path("data/silver")
SILVER_DIR.mkdir(parents=True, exist_ok=True)

REF_DIR = Path("data/reference")
REF_DIR.mkdir(parents=True, exist_ok=True)

LAKE_DIR = Path("data/lake")
LAKE_DIR.mkdir(parents=True, exist_ok=True)
DUCK_PATH = LAKE_DIR / "warehouse.duckdb"

SILVER_CSV = SILVER_DIR / "itl2_unified_history.csv"
ITL2_LOOKUP_CSV = REF_DIR / "itl2_2021_2025_lookup.csv"

VINTAGE = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("itl2_ingest")


# -----------------------------
# Helpers
# -----------------------------

def load_itl2_lookup():
    """
    Load ONS 2021→2025 ITL2 lookup and return:

    - canonical_codes: set of 2025 ITL2 codes (ITL225CD), size ~46
    - code_to_name: mapping from 2025 ITL2 code → 2025 name (ITL225NM)
    """
    if not ITL2_LOOKUP_CSV.exists():
        raise FileNotFoundError(
            f"ITL2 lookup not found at {ITL2_LOOKUP_CSV}. "
            "Expected columns: ITL221CD, ITL221NM, ITL225CD, ITL225NM."
        )

    df = pd.read_csv(ITL2_LOOKUP_CSV)
    required = {"ITL221CD", "ITL221NM", "ITL225CD", "ITL225NM"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"ITL2 lookup missing required columns: {sorted(missing)} "
            f"(found: {list(df.columns)})"
        )

    canonical_codes = sorted(df["ITL225CD"].unique())
    code_to_name = (
        df[["ITL225CD", "ITL225NM"]]
        .drop_duplicates()
        .set_index("ITL225CD")["ITL225NM"]
        .to_dict()
    )

    log.info(
        "ITL2 lookup loaded: %d canonical 2025 ITL2 codes (e.g. %s)",
        len(canonical_codes),
        ", ".join(canonical_codes[:5]),
    )
    return set(canonical_codes), code_to_name


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
    raise KeyError(
        "Could not detect value column in dataframe; columns: "
        + ", ".join(df.columns)
    )


def _require_cols(df: pd.DataFrame, cols: list, ctx: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{ctx}: missing required columns {missing}")


def _tidy(
    df: pd.DataFrame, metric_id: str, unit: str, source: str
) -> pd.DataFrame:
    """
    Convert a NOMIS-like df to RegionIQ tidy schema:
    region_code, region_name, region_level, metric_id, period, value,
    unit, freq, source, vintage
    """
    value_col = _value_col(df)

    # NOMIS variants
    rc = "GEOGRAPHY_CODE" if "GEOGRAPHY_CODE" in df.columns else "GEOGRAPHY"
    rn = "GEOGRAPHY_NAME" if "GEOGRAPHY_NAME" in df.columns else None
    date_col = "DATE" if "DATE" in df.columns else None

    _require_cols(df, [rc], f"{metric_id}")
    if date_col is None:
        raise KeyError(f"{metric_id}: expected a DATE column (year) in incoming data")

    cols = [rc, date_col, value_col]
    if rn:
        cols.insert(1, rn)

    out = df[cols].copy()

    rename_map = {rc: "region_code", date_col: "period", value_col: "value"}
    if rn:
        rename_map[rn] = "region_name"
    out.rename(columns=rename_map, inplace=True)

    if "region_name" not in out.columns:
        out["region_name"] = None

    out["period"] = pd.to_numeric(out["period"], errors="coerce").astype("Int64")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")

    # Add metadata
    out["region_level"] = "ITL2"
    out["metric_id"] = metric_id
    out["unit"] = unit
    out["freq"] = "A"
    out["source"] = source
    out["vintage"] = VINTAGE

    # Clean rows
    out = out.dropna(subset=["period", "value"]).reset_index(drop=True)

    # Reorder
    cols_final = [
        "region_code",
        "region_name",
        "region_level",
        "metric_id",
        "period",
        "value",
        "unit",
        "freq",
        "source",
        "vintage",
    ]
    return out[cols_final]


def _canonicalise_itl2(
    tidy_df: pd.DataFrame,
    canonical_codes: set,
    code_to_name: dict,
    context: str,
) -> pd.DataFrame:
    """
    Filter a tidy indicator dataframe down to canonical 2025 ITL2 codes.

    - Drops any region_code not in canonical_codes (e.g. TLC1, TLH1, TLK1, TLL1, TLM6/7/8)
    - Normalises region_name to 2025 name from code_to_name where possible
    """
    before_regions = tidy_df["region_code"].nunique()
    before_rows = len(tidy_df)

    # Filter to canonical codes
    mask = tidy_df["region_code"].isin(canonical_codes)
    filtered = tidy_df.loc[mask].copy()

    after_regions = filtered["region_code"].nunique()
    after_rows = len(filtered)

    dropped_regions = before_regions - after_regions
    dropped_rows = before_rows - after_rows

    log.info(
        "%s: canonical ITL2 filter dropped %d region codes, %d rows "
        "(%d → %d regions, %d → %d rows)",
        context,
        dropped_regions,
        dropped_rows,
        before_regions,
        after_regions,
        before_rows,
        after_rows,
    )

    # Normalise names to 2025 official names where available
    if "region_name" in filtered.columns:
        filtered["region_name"] = filtered["region_code"].map(code_to_name).fillna(
            filtered["region_name"]
        )
    else:
        filtered["region_name"] = filtered["region_code"].map(code_to_name)

    return filtered.reset_index(drop=True)


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
        schema, table = "main", table_fullname

    con = duckdb.connect(str(DUCK_PATH))
    try:
        if schema != "main":
            con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        con.register("df_tmp", df)
        con.execute(f"CREATE OR REPLACE TABLE {schema}.{table} AS SELECT * FROM df_tmp")
        log.debug("Written %d rows to %s.%s", df.shape[0], schema, table)
    finally:
        con.close()


# -----------------------------
# Indicator fetchers
# -----------------------------

def fetch_emp_itl2(canonical_codes, code_to_name) -> pd.DataFrame:
    """
    Employment (jobs), ITL2, 2015–2024 via NOMIS NM_189_1.

    - Uses BRES dataset NM_189_1
    - Geography: ITL2 codes (geography=1799356417...1799356456)
    - We deliberately ignore older NUTS2 back series to avoid mixing UKC*/UKD*

    Note: NOMIS currently only returns 2022–2024 for this geography window,
    so history is short but clean.
    """
    url_2015_24 = (
        "https://www.nomisweb.co.uk/api/v01/dataset/NM_189_1.data.csv"
        "?geography=1799356417...1799356456"
        "&industry=37748736"
        "&employment_status=1"
        "&measure=1"
        "&measures=20100"
    )

    log.info("EMP: fetching 2015–2024 (NM_189_1, ITL2)…")
    df = pd.read_csv(url_2015_24)
    log.info("EMP: raw rows=%d", df.shape[0])

    if "DATE" in df.columns:
        before = df.shape[0]
        df["DATE"] = pd.to_numeric(df["DATE"], errors="coerce")
        df = df[df["DATE"] >= 2015].copy()
        after = df.shape[0]
        if after < before:
            log.info("EMP: filtered years <2015 (%d → %d rows)", before, after)
    else:
        log.warning("EMP: DATE column missing; cannot filter by year explicitly")

    raw_out = RAW_EMP_DIR / "emp_itl2_nomis_2015_2024.csv"
    df.to_csv(raw_out, index=False)
    log.info("EMP: saved raw → %s", raw_out)

    tidy = _tidy(df, metric_id="emp_total_jobs", unit="jobs", source="NOMIS")
    tidy_canon = _canonicalise_itl2(
        tidy, canonical_codes, code_to_name, context="EMP"
    )

    _write_duck("bronze.emp_itl2_raw", df)

    return tidy_canon


def fetch_gdhi_itl2(canonical_codes, code_to_name) -> pd.DataFrame:
    """
    GDHI total (£m) and per head (£), ITL2 via NOMIS NM_185_1.

    Geography: 1761607681...1761607726
    MEASURE:
      1 = GDHI total (£m)
      2 = GDHI per head (£)
    """
    url = (
        "https://www.nomisweb.co.uk/api/v01/dataset/NM_185_1.data.csv"
        "?geography=1761607681...1761607726"
        "&component_of_gdhi=0"
        "&measure=1,2"
        "&measures=20100"
    )

    log.info("GDHI: fetching…")
    df = pd.read_csv(url)

    raw_out = RAW_GDHI_DIR / "gdhi_itl2_nomis.csv"
    df.to_csv(raw_out, index=False)
    log.info("GDHI: saved raw → %s", raw_out)

    _require_cols(df, ["MEASURE"], "GDHI")

    # Normalise measure to Int
    try:
        df["MEASURE"] = pd.to_numeric(df["MEASURE"], errors="coerce").astype("Int64")
    except Exception:
        log.warning("GDHI: could not coerce MEASURE column to Int64")

    # Total (£m)
    df_total = df[df["MEASURE"] == 1].copy()
    gdhi_total = _tidy(
        df_total, metric_id="gdhi_total_mn_gbp", unit="GBP_m", source="NOMIS"
    )
    gdhi_total = _canonicalise_itl2(
        gdhi_total, canonical_codes, code_to_name, context="GDHI_TOTAL"
    )

    # Per head (£)
    df_ph = df[df["MEASURE"] == 2].copy()
    gdhi_ph = _tidy(
        df_ph, metric_id="gdhi_per_head_gbp", unit="GBP", source="NOMIS"
    )
    gdhi_ph = _canonicalise_itl2(
        gdhi_ph, canonical_codes, code_to_name, context="GDHI_PER_HEAD"
    )

    _write_duck("bronze.gdhi_itl2_raw", df)

    combined = pd.concat([gdhi_total, gdhi_ph], ignore_index=True)
    return combined


def fetch_gva_itl2(canonical_codes, code_to_name) -> pd.DataFrame:
    """
    GVA (£m), ITL2 via NOMIS NM_2400_1.

    Geography: 1761607681...1761607726
    """
    url = (
        "https://www.nomisweb.co.uk/api/v01/dataset/NM_2400_1.data.csv"
        "?geography=1761607681...1761607726"
        "&cell=0"
        "&measures=20100"
    )

    log.info("GVA: fetching…")
    df = pd.read_csv(url)

    raw_out = RAW_GVA_DIR / "gva_itl2_nomis.csv"
    df.to_csv(raw_out, index=False)
    log.info("GVA: saved raw → %s", raw_out)

    tidy = _tidy(
        df, metric_id="nominal_gva_mn_gbp", unit="GBP_m", source="NOMIS"
    )
    tidy_canon = _canonicalise_itl2(
        tidy, canonical_codes, code_to_name, context="GVA"
    )

    _write_duck("bronze.gva_itl2_raw", df)

    return tidy_canon


# -----------------------------
# Main
# -----------------------------

def main():
    log.info("=== Unified ITL2 ingest starting (vintage=%s) ===", VINTAGE)
    log.info("Datasets:")
    log.info("  - Employment: NM_189_1 ITL2 (2015–2024 only, canonical ITL2=2025)")
    log.info("  - GDHI: NM_185_1 (canonical ITL2=2025)")
    log.info("  - GVA: NM_2400_1 (canonical ITL2=2025)")
    log.info("  - Population: SKIPPED (will aggregate from LA)")
    log.info("  - ITL2 geometry: enforced via %s", ITL2_LOOKUP_CSV)

    # 1) Load canonical ITL2 geometry
    try:
        canonical_codes, code_to_name = load_itl2_lookup()
    except Exception as e:
        log.error("Failed to load ITL2 lookup")
        log.exception(e)
        sys.exit(1)

    silver_frames = []
    failures = {}

    # 2) EMPLOYMENT
    try:
        emp = fetch_emp_itl2(canonical_codes, code_to_name)
        silver_frames.append(emp)
        log.info(
            "EMP: tidy rows=%d | regions=%d | years=[%s..%s]",
            emp.shape[0],
            emp["region_code"].nunique(),
            emp["period"].min(),
            emp["period"].max(),
        )
    except Exception as e:
        failures["employment"] = str(e)
        log.exception("EMPLOYMENT failed")

    # 3) GDHI
    try:
        gdhi = fetch_gdhi_itl2(canonical_codes, code_to_name)
        silver_frames.append(gdhi)
        log.info(
            "GDHI: tidy rows=%d | regions=%d | metrics=%s",
            gdhi.shape[0],
            gdhi["region_code"].nunique(),
            sorted(gdhi["metric_id"].unique()),
        )
    except Exception as e:
        failures["gdhi"] = str(e)
        log.exception("GDHI failed")

    # 4) GVA
    try:
        gva = fetch_gva_itl2(canonical_codes, code_to_name)
        silver_frames.append(gva)
        log.info(
            "GVA: tidy rows=%d | regions=%d | years=[%s..%s]",
            gva.shape[0],
            gva["region_code"].nunique(),
            gva["period"].min(),
            gva["period"].max(),
        )
    except Exception as e:
        failures["gva"] = str(e)
        log.exception("GVA failed")

    if not silver_frames:
        log.error("No indicators ingested; exiting with failure.")
        sys.exit(2)

    # 5) Combine silver and final sanity
    silver = pd.concat(silver_frames, ignore_index=True)

    key_cols = ["region_code", "region_level", "metric_id", "period", "value"]
    missing_any = [c for c in key_cols if c not in silver.columns]
    if missing_any:
        log.error("Silver missing required columns: %s", missing_any)
        sys.exit(3)

    # Enforce canonical ITL2 just in case
    before_regions = silver["region_code"].nunique()
    silver = silver[silver["region_code"].isin(canonical_codes)].copy()
    after_regions = silver["region_code"].nunique()
    if after_regions != before_regions:
        log.info(
            "Final canonical pass dropped %d region codes in combined silver "
            "(%d → %d regions)",
            before_regions - after_regions,
            before_regions,
            after_regions,
        )

    # 6) Save silver CSV
    silver.to_csv(SILVER_CSV, index=False)
    log.info(
        "Saved silver tidy CSV → %s (rows=%d, regions=%d)",
        SILVER_CSV,
        silver.shape[0],
        silver["region_code"].nunique(),
    )

    # 7) Save silver to DuckDB (optional)
    if duckdb is not None:
        _write_duck("silver.itl2_history", silver)
        log.info("Wrote silver.itl2_history to %s", DUCK_PATH)
    else:
        log.warning(
            "duckdb not installed; skipped writing silver.itl2_history to DuckDB."
        )

    # 8) Final summary
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
