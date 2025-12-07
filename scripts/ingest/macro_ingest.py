#!/usr/bin/env python3
# scripts/ingest/macro_to_duckdb.py
"""
Unified UK Macro ingest → DuckDB (bronze + silver) + CSVs

Sources:
- ONS beta API v4 downloads (auto-latest):
    * GDP monthly estimate (dataset: gdp-to-four-decimal-places)  → uk_gdp_m_index
    * CPIH (dataset: cpih01)                                      → uk_cpih_index
    * Labour market headline (dataset: labour-market)             → uk_unemp_rate
    * ASHE (dataset: ashe-tables-7-and-8)                         → uk_median_weekly_pay_fulltime
- Bank of England JSON endpoint (direct, daily Bank Rate)         → boe_base_rate

Outputs:
- data/raw/macro/*  (per-dataset raw CSV snapshots)
- data/silver/uk_macro_history.csv  (tidy, unified)
- data/lake/warehouse.duckdb:
    bronze.macro_gdp_raw, bronze.macro_cpih_raw,
    bronze.macro_labour_raw, bronze.macro_ashe_raw, bronze.boe_base_rate_raw
    silver.uk_macro_history
"""

import re
import io
import os
import json
import time
import logging
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

try:
    import duckdb
except ImportError:
    duckdb = None

# -----------------------------
# Constants & Paths
# -----------------------------
ONS_API = "https://api.beta.ons.gov.uk/v1"
ONS_DL = "https://download.ons.gov.uk/downloads/datasets"

UK_CODE = "K02000001"
UK_NAME = "United Kingdom"

RAW_DIR = Path("data/raw/macro")
RAW_DIR.mkdir(parents=True, exist_ok=True)

SILVER_DIR = Path("data/silver")
SILVER_DIR.mkdir(parents=True, exist_ok=True)
SILVER_CSV = SILVER_DIR / "uk_macro_history.csv"

LAKE_DIR = Path("data/lake")
LAKE_DIR.mkdir(parents=True, exist_ok=True)
DUCK_PATH = LAKE_DIR / "warehouse.duckdb"

VINTAGE = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("macro_ingest")

# -----------------------------
# DuckDB helpers
# -----------------------------
def _write_duck(table_fullname: str, df: pd.DataFrame):
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
    finally:
        con.close()

# -----------------------------
# Generic ONS v4 helpers
# -----------------------------
def _get(url: str, as_json=True):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/119.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json() if as_json else r.text

def ons_latest_version_meta(dataset_id: str, edition_hint: str | None = None) -> dict:
    """
    Resolve dataset → latest edition/version → return version metadata JSON
    (contains stable CSV download link under downloads.csv.href)
    """
    if edition_hint:
        ed_url = f"{ONS_API}/datasets/{dataset_id}/editions/{edition_hint}"
    else:
        # Get dataset root to find latest_version directly
        dataset_meta = _get(f"{ONS_API}/datasets/{dataset_id}")
        latest_v_url = dataset_meta.get("links", {}).get("latest_version", {}).get("href")
        
        if latest_v_url:
            # Use the dataset's explicit latest_version link (bypasses edition guessing)
            log.info("%s: Using dataset-level latest_version link", dataset_id)
            return _get(latest_v_url)
        
        # Fallback: pick from editions list
        eds = _get(f"{ONS_API}/datasets/{dataset_id}/editions")
        items = eds.get("items", [])
        if not items:
            raise RuntimeError(f"No editions for dataset {dataset_id}")
        
        # Sort by last_updated to get most recent edition
        items_sorted = sorted(items, key=lambda x: x.get("last_updated", ""), reverse=True)
        edition = items_sorted[0]
        ed_url = edition["links"]["self"]["href"]
        log.info("%s: Auto-selected edition %s (last_updated=%s)", 
                 dataset_id, edition.get("edition"), edition.get("last_updated"))

    # Get edition metadata, then hop to latest_version
    ed_meta = _get(ed_url)
    latest_v = ed_meta["links"]["latest_version"]["href"]
    return _get(latest_v)  # this is the version JSON (has downloads)

def ons_read_latest_csv(dataset_id: str, edition_hint: str | None = None) -> pd.DataFrame:
    """
    Get latest CSV for a dataset via version metadata.
    """
    vmeta = ons_latest_version_meta(dataset_id, edition_hint)
    dl = vmeta.get("downloads", {}).get("csv", {})
    csv_url = dl.get("href")
    if not csv_url:
        # Fallback: construct from IDs
        ed_parts = vmeta["links"]["edition"]["href"].split("/editions/")[-1]
        ver_id = vmeta.get("id") or vmeta.get("version")
        csv_url = f"{ONS_DL}/{dataset_id}/editions/{ed_parts}/versions/{ver_id}.csv"
    log.info("ONS CSV → %s", csv_url)
    csv = requests.get(csv_url, timeout=60)
    csv.raise_for_status()
    df = pd.read_csv(io.StringIO(csv.text))
    return df

# -----------------------------
# Time & value parsing
# -----------------------------
MONTHS = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05", "JUN": "06",
    "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12"
}

def parse_ons_time(val: str) -> tuple[str, str]:
    """
    Parse ONS 'time' strings into (period, freq).
    Returns period as ISO-like:
      - Monthly: YYYY-MM
      - Quarterly: YYYY-Qn
      - Annual: YYYY
    """
    s = str(val).strip().upper()
    # Monthly patterns: "MMM-YY", "YYYY MMM", "MMM YYYY"
    m1 = re.match(r"([A-Z]{3})[-\s](\d{2,4})$", s)          # e.g., "AUG 24" or "AUG-2024"
    m2 = re.match(r"(\d{4})[-\s]([A-Z]{3})$", s)            # e.g., "2024 AUG"
    if m1:
        mon, yy = m1.groups()
        year = int(yy) + 2000 if len(yy) == 2 else int(yy)
        month = MONTHS.get(mon)
        if month: return (f"{year:04d}-{month}", "M")
    if m2:
        year, mon = m2.groups()
        month = MONTHS.get(mon)
        if month: return (f"{int(year):04d}-{month}", "M")

    # Quarterly: "YYYY Qn" or "Qn YYYY"
    q1 = re.match(r"(\d{4})\s*Q([1-4])$", s)
    q2 = re.match(r"Q([1-4])\s*(\d{4})$", s)
    if q1:
        y, q = q1.groups()
        return (f"{int(y):04d}-Q{q}", "Q")
    if q2:
        q, y = q2.groups()
        return (f"{int(y):04d}-Q{q}", "Q")

    # Annual: "YYYY"
    if re.match(r"^\d{4}$", s):
        return (s, "A")

    # If already date-like "YYYY-MM" or "YYYY-MM-DD"
    if re.match(r"^\d{4}-\d{2}(-\d{2})?$", s):
        # Infer monthly if day missing; daily if day present
        freq = "D" if len(s) == 10 else "M"
        return (s, freq)

    # Fallback: return as-is, unknown freq
    return (s, "U")

def find_value_col(df: pd.DataFrame) -> str:
    cand = ["observation", "OBS_VALUE", "Value", "VALUE", "v4_0", "v4_1", "v4_2"]
    for c in cand:
        if c in df.columns:
            return c
    # last numeric column
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        return num_cols[-1]
    raise KeyError(f"Could not detect value column in columns={list(df.columns)}")

def find_time_col(df: pd.DataFrame) -> str:
    for c in ["time", "Time", "DATE", "date", "Period", "period"]:
        if c in df.columns: return c
    raise KeyError("No 'time' column found")

# -----------------------------
# Tidy & write helpers
# -----------------------------
def to_tidy_uk(df: pd.DataFrame, metric_id: str, unit: str, source: str,
               geography_col_guess: list[str] | None = None) -> pd.DataFrame:
    """
    Normalise to RegionIQ tidy schema for UK-only macro.
    """
    time_col = find_time_col(df)
    val_col = find_value_col(df)

    # Filter to UK geography if present
    geog_cols = geography_col_guess or ["geography", "GEOGRAPHY", "Geography", "Region", "Area", "country"]
    geo_present = [c for c in geog_cols if c in df.columns]
    if geo_present:
        gcol = geo_present[0]
        # Match United Kingdom, UK, or ONS code
        mask = df[gcol].astype(str).str.contains("United Kingdom|UK|K02000001", case=False, regex=True)
        df = df[mask].copy()
        if df.empty:
            log.warning("%s: UK filter produced empty frame; using full df", metric_id)
            df = df.copy()
    else:
        df = df.copy()

    # Parse time → (period, freq)
    periods, freqs = [], []
    for t in df[time_col].astype(str).tolist():
        p, f = parse_ons_time(t)
        periods.append(p); freqs.append(f)

    out = pd.DataFrame({
        "region_code": UK_CODE,
        "region_name": UK_NAME,
        "region_level": "UK",
        "metric_id": metric_id,
        "period": periods,
        "value": pd.to_numeric(df[val_col], errors="coerce"),
        "unit": unit,
        "freq": freqs,
        "source": source,
        "vintage": VINTAGE,
    })
    out = out.dropna(subset=["value"]).reset_index(drop=True)
    return out

def save_raw(df: pd.DataFrame, name: str) -> Path:
    p = RAW_DIR / f"{name}.csv"
    df.to_csv(p, index=False)
    log.info("Saved raw → %s (rows=%d)", p, len(df))
    return p

# -----------------------------
# Bank of England - Global-Rates.com scrape (fallback)
# -----------------------------
def fetch_boe_base_rate() -> pd.DataFrame:
    """
    Fetch Bank of England Bank Rate from Global-Rates.com.
    
    Uses static HTML table scraping as a reliable fallback since official
    BoE endpoints are currently inaccessible. This is acceptable for 
    low-frequency ingestion (daily/weekly snapshots).
    
    Source: https://www.global-rates.com/en/interest-rates/central-banks/1003/
    """
    url = "https://www.global-rates.com/en/interest-rates/central-banks/1003/british-boe-official-bank-rate/"
    
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/119.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.global-rates.com/en/interest-rates/central-banks/",
    }
    
    log.info("BoE (Global-Rates fallback) → %s", url)
    
    # Fetch and parse HTML table
    tables = pd.read_html(url, storage_options={"User-Agent": headers["User-Agent"]})
    
    if not tables:
        raise RuntimeError("Global-Rates: No tables found on page")
    
    # First table is the Bank Rate history
    df = tables[0].copy()
    
    # Save raw
    save_raw(df, "boe_base_rate_raw")
    
    # Clean column names (usually "Date" and "Rate")
    if len(df.columns) >= 2:
        df.columns = ["Date", "Rate"] + list(df.columns[2:]) if len(df.columns) > 2 else ["Date", "Rate"]
    else:
        raise RuntimeError(f"Global-Rates: Expected at least 2 columns, got {len(df.columns)}")
    
    # Parse dates and rates
    # Dates are in format "dd-mm-yyyy" (e.g., "08-07-2025")
    df["period"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce").dt.strftime("%Y-%m-%d")
    
    # Rates have "%" suffix, strip and convert
    df["value"] = df["Rate"].astype(str).str.replace("%", "").str.strip()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    
    # Build tidy frame
    tidy = pd.DataFrame({
        "region_code": UK_CODE,
        "region_name": UK_NAME,
        "region_level": "UK",
        "metric_id": "boe_base_rate",
        "period": df["period"],
        "value": df["value"],
        "unit": "percent",
        "freq": "D",
        "source": "Bank of England (via Global-Rates.com)",
        "vintage": VINTAGE,
    }).dropna(subset=["period", "value"]).reset_index(drop=True)
    
    log.info("BoE (Global-Rates): tidy rows=%d | range [%s → %s]", 
             len(tidy), tidy["period"].min(), tidy["period"].max())
    
    return tidy

# -----------------------------
# Dataset-specific filters
# -----------------------------
def filter_labour_unemployment(df: pd.DataFrame) -> pd.DataFrame:
    """
    From 'labour-market' CSV, retain the UK headline unemployment rate (16+).
    Heuristic: look for a 'series' or descriptor containing 'Unemployment rate' and 'aged 16 years and over' (or similar).
    """
    lower_cols = {c.lower(): c for c in df.columns}
    text_cols = [lower_cols.get(c) for c in ["series", "Series", "name", "TITLE", "title", "description", "Indicator"] if lower_cols.get(c)]
    if not text_cols:
        log.warning("labour-market: no obvious series/description column; returning full df")
        return df

    col = text_cols[0]
    mask = df[col].astype(str).str.contains("UNEMPLOYMENT RATE", case=False, regex=True)
    if not mask.any():
        # broader fallback
        mask = df[col].astype(str).str.contains("UNEMPLOYMENT", case=False, regex=True)

    out = df[mask].copy()
    if out.empty:
        log.warning("labour-market: could not isolate unemployment rate; returning full df")
        out = df.copy()
    return out

def filter_ashe_median_fulltime_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    From ASHE 7/8 tables CSV, isolate Median gross weekly pay, Full-time, All employees, UK.
    Heuristics based on column names seen in ASHE exports.
    """
    cols = {c.lower(): c for c in df.columns}
    # Likely columns: 'Earnings', 'Statistic', 'Working pattern', 'Sex', 'geography'
    def pick(name: str) -> str | None:
        return cols.get(name.lower())

    # filter geography later in to_tidy_uk; here filter dimension values
    f = df.copy()
    # Earnings measure
    e_col = pick("Earnings") or pick("EARNINGS") or pick("PAY TYPE") or pick("Variable")
    if e_col:
        f = f[f[e_col].astype(str).str.contains("Weekly pay.*gross", case=False, regex=True)]
    # Statistic = Median
    s_col = pick("Statistic") or pick("STATISTIC")
    if s_col:
        f = f[f[s_col].astype(str).str.contains("Median", case=False, regex=True)]
    # Working pattern = Full-time
    w_col = pick("Working pattern") or pick("working pattern") or pick("Hours")
    if w_col:
        f = f[f[w_col].astype(str).str.contains("Full-time", case=False, regex=True)]
    # Sex = All employees (if present)
    sex_col = pick("Sex") or pick("sex")
    if sex_col and "All" in f[sex_col].astype(str).unique():
        f = f[f[sex_col].astype(str).str.contains("All", case=False, regex=True)]

    return f if not f.empty else df

# -----------------------------
# Fetchers
# -----------------------------
def fetch_gdp() -> pd.DataFrame:
    df = ons_read_latest_csv("gdp-to-four-decimal-places", edition_hint="time-series")
    save_raw(df, "ons_gdp_raw")
    tidy = to_tidy_uk(df, metric_id="uk_gdp_m_index", unit="index(2016=100)", source="ONS")
    # ensure monthly only (gdp monthly dataset is M)
    tidy["freq"] = "M"
    return tidy

def fetch_cpih() -> pd.DataFrame:
    df = ons_read_latest_csv("cpih01", edition_hint="time-series")
    save_raw(df, "ons_cpih_raw")
    tidy = to_tidy_uk(df, metric_id="uk_cpih_index", unit="index(2015=100)", source="ONS")
    tidy["freq"] = "M"
    return tidy

def fetch_labour_unemp() -> pd.DataFrame:
    """
    Fetch Labour Market data from ONS (auto-detects latest edition, e.g. PWT24).
    Uses latest edition by last_updated date - future-proof for PWT25, etc.
    """
    df = ons_read_latest_csv("labour-market", edition_hint=None)
    save_raw(df, "ons_labour_raw")
    df_u = filter_labour_unemployment(df)
    tidy = to_tidy_uk(df_u, metric_id="uk_unemp_rate", unit="percent", source="ONS")
    # Labour series can be M or Q; keep whatever parse_ons_time inferred
    return tidy

def fetch_ashe_median_weekly() -> pd.DataFrame:
    # Latest edition is usually the reference year; pick automatically
    df = ons_read_latest_csv("ashe-tables-7-and-8", edition_hint=None)
    save_raw(df, "ons_ashe_raw")
    df_m = filter_ashe_median_fulltime_weekly(df)
    tidy = to_tidy_uk(df_m, metric_id="uk_median_weekly_pay_fulltime", unit="GBP_per_week", source="ONS")
    # ASHE is annual
    tidy["freq"] = "A"
    # Periods may be year or April YYYY; normalise to YYYY
    tidy["period"] = tidy["period"].str.replace(r"-\d{2}$", "", regex=True)
    return tidy

# -----------------------------
# Main
# -----------------------------
def main():
    log.info("=== UK Macro ingest starting (vintage=%s) ===", VINTAGE)
    silver_frames = []
    failures = {}

    # GDP
    try:
        gdp = fetch_gdp()
        silver_frames.append(gdp)
        _write_duck("bronze.macro_gdp_raw", gdp)  # write tidy as traceable snapshot too
        log.info("GDP: tidy rows=%d | period [%s..%s]", len(gdp), gdp["period"].min(), gdp["period"].max())
    except Exception as e:
        failures["gdp"] = str(e)
        log.exception("GDP failed")

    # CPIH
    try:
        cpih = fetch_cpih()
        silver_frames.append(cpih)
        _write_duck("bronze.macro_cpih_raw", cpih)
        log.info("CPIH: tidy rows=%d | period [%s..%s]", len(cpih), cpih["period"].min(), cpih["period"].max())
    except Exception as e:
        failures["cpih"] = str(e)
        log.exception("CPIH failed")

    # Labour (Unemployment rate)
    try:
        unemp = fetch_labour_unemp()
        silver_frames.append(unemp)
        _write_duck("bronze.macro_labour_raw", unemp)
        log.info("LABOUR (unemp): tidy rows=%d | sample period [%s..%s]", len(unemp), unemp["period"].min(), unemp["period"].max())
    except Exception as e:
        failures["labour_unemp"] = str(e)
        log.exception("Labour market failed")

    # ASHE (Median weekly pay FT)
    try:
        ashe = fetch_ashe_median_weekly()
        silver_frames.append(ashe)
        _write_duck("bronze.macro_ashe_raw", ashe)
        log.info("ASHE: tidy rows=%d | years [%s..%s]", len(ashe), ashe["period"].min(), ashe["period"].max())
    except Exception as e:
        failures["ashe"] = str(e)
        log.exception("ASHE failed")

    # BoE base rate
    try:
        boe = fetch_boe_base_rate()
        silver_frames.append(boe)
        _write_duck("bronze.boe_base_rate_raw", boe)
        log.info("BoE: tidy rows=%d | dates [%s..%s]", len(boe), boe["period"].min(), boe["period"].max())
    except Exception as e:
        failures["boe_base_rate"] = str(e)
        log.exception("BoE base rate failed")

    if not silver_frames:
        log.error("No macro indicators ingested; exiting.")
        raise SystemExit(2)

    # Unite
    silver = pd.concat(silver_frames, ignore_index=True)

    # Sort & basic sanity
    silver = silver.dropna(subset=["period", "value"]).reset_index(drop=True)
    silver = silver.sort_values(["metric_id", "period"]).reset_index(drop=True)

    # Save silver CSV
    SILVER_CSV.parent.mkdir(parents=True, exist_ok=True)
    silver.to_csv(SILVER_CSV, index=False)
    log.info("Saved silver tidy CSV → %s (rows=%d)", SILVER_CSV, silver.shape[0])

    # Save to DuckDB
    if duckdb is not None:
        _write_duck("silver.uk_macro_history", silver)
        log.info("Wrote silver.uk_macro_history to %s", DUCK_PATH)
    else:
        log.warning("duckdb not installed; skipped writing silver to DuckDB.")

    # Summary
    by_metric = (
        silver.groupby(["metric_id", "freq"])["value"].count()
        .rename("rows").reset_index().sort_values(["metric_id", "freq"])
    )
    log.info("Ingest complete. Rows by metric:\n%s", by_metric.to_string(index=False))

    if failures:
        log.warning("Completed with %d failures: %s", len(failures), failures)
    else:
        log.info("All macro indicators ingested successfully.")

if __name__ == "__main__":
    main()