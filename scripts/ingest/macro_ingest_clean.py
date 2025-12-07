#!/usr/bin/env python3
# scripts/ingest/macro_to_duckdb.py
"""
Unified UK Macro ingest → DuckDB (bronze + silver) + CSVs

Sources:
- ONS beta API v4 downloads (auto-latest):
    * GDP monthly estimate (dataset: gdp-to-four-decimal-places)  → uk_gdp_m_index
    * CPIH (dataset: cpih01)                                      → uk_cpih_index
    * Labour market headline (dataset: labour-market)             → uk_unemp_rate, uk_emp_rate
    * ASHE (dataset: ashe-tables-7-and-8)                         → uk_median_weekly_pay_fulltime
- Bank of England via Global-Rates.com (daily Bank Rate)          → boe_base_rate

Outputs:
- data/raw/macro/*  (per-dataset raw CSV snapshots)
- data/silver/uk_macro_history.csv  (tidy, unified)
- data/lake/warehouse.duckdb:
    bronze.macro_gdp_raw, bronze.macro_cpih_raw,
    bronze.macro_labour_headlines, bronze.macro_ashe_raw, bronze.boe_base_rate_raw
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

# Regex for 3-month rolling windows
M3_WINDOW_RE = re.compile(r"^\s*([A-Za-z]{3})[-–]([A-Za-z]{3})\s+(\d{4})\s*$")

def parse_ons_time(val: str) -> tuple[str, str]:
    """
    Parse ONS time strings into (period, freq).
    Supports:
      - M3 windows like 'FEB-APR 2025'  → ('2025-04', 'M3')  (end-month)
      - Monthly: 'MMM YYYY' / 'YYYY MMM' / 'MMM-YYYY'       → ('YYYY-MM','M')
      - Quarterly: 'YYYY Qn' / 'Qn YYYY'                    → ('YYYY-Qn','Q')
      - Annual: 'YYYY'                                      → ('YYYY','A')
      - ISO-like: 'YYYY-MM' / 'YYYY-MM-DD'                  → ('YYYY-MM','M') or ('YYYY-MM-DD','D')
    """
    s = str(val).strip()
    u = s.upper()

    # Rolling 3-month window: e.g., 'FEB-APR 2025'
    m3 = M3_WINDOW_RE.match(u)
    if m3:
        m1, m2, y = m3.groups()
        if m2[:3] in MONTHS:
            return (f"{int(y):04d}-{MONTHS[m2[:3]]}", "M3")

    # Monthly patterns
    m1 = re.match(r"^([A-Z]{3})[-\s](\d{2,4})$", u)          # 'AUG 24' or 'AUG-2024'
    m2 = re.match(r"^(\d{4})[-\s]([A-Z]{3})$", u)            # '2024 AUG'
    if m1:
        mon, yy = m1.groups()
        if len(yy) == 2:
            # Smart two-digit year handling: 
            # If YY > 50, assume 1900s; if YY <= 50, assume 2000s
            # (This handles data from 1951-2050)
            year = int(yy) + (1900 if int(yy) > 50 else 2000)
        else:
            year = int(yy)
        if mon in MONTHS: return (f"{year:04d}-{MONTHS[mon]}", "M")
    if m2:
        year, mon = m2.groups()
        if mon in MONTHS: return (f"{int(year):04d}-{MONTHS[mon]}", "M")

    # Quarterly
    q1 = re.match(r"^(\d{4})\s*Q([1-4])$", u)
    q2 = re.match(r"^Q([1-4])\s*(\d{4})$", u)
    if q1: y, q = q1.groups(); return (f"{int(y):04d}-Q{q}", "Q")
    if q2: q, y = q2.groups(); return (f"{int(y):04d}-Q{q}", "Q")

    # Annual
    if re.match(r"^\d{4}$", u):
        return (u, "A")

    # ISO-like monthly/daily
    if re.match(r"^\d{4}-\d{2}(-\d{2})?$", s):
        return (s, "D" if len(s) == 10 else "M")

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
    # Check for various possible time column names (including the format from your data)
    for c in ["time", "Time", "DATE", "date", "Period", "period", "mmm-mmm-Time"]:
        if c in df.columns: return c
    # Check if any column contains "time" in its name
    for c in df.columns:
        if "time" in c.lower():
            return c
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
    geog_cols = geography_col_guess or ["geography", "GEOGRAPHY", "Geography", "Region", "Area", "country", "uk-only"]
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
# Labour headline selectors
# -----------------------------
LABOUR_TEXT_COLS = [
    "series", "Series", "name", "NAME", "title", "TITLE",
    "description", "Description", "Indicator", "indicator",
    "economic-activity", "EconomicActivity", "economic-a", "EconomicActi"
]

def _first_present(df: pd.DataFrame, options: list[str]) -> str | None:
    for c in options:
        if c in df.columns:
            return c
    return None

def _labour_pref_filter(df: pd.DataFrame, keyword: str, prefer_age: list[str]) -> pd.DataFrame:
    """
    More flexible filter for messy ONS labour data with truncated columns.
    Targets unemployment/employment rates, excluding levels.
    """
    f = df.copy()
    
    # Find any column that might contain the indicator/economic activity
    indicator_col = None
    for col in f.columns:
        col_lower = col.lower()
        if any(x in col_lower for x in ['economic', 'activity', 'indicator', 'series', 'measure']):
            indicator_col = col
            break
    
    # Filter by indicator keyword if we found a suitable column
    if indicator_col:
        if keyword.lower() == "unemployment rate":
            # Look for unemployment-related terms
            mask = f[indicator_col].astype(str).str.contains(
                r'unemploy|Unemploy', case=False, regex=True
            )
            f = f[mask]
        elif keyword.lower() == "employment rate":
            # Look for employment rate, but exclude "In employment" levels
            mask = (
                f[indicator_col].astype(str).str.contains(r'employ', case=False) & 
                ~f[indicator_col].astype(str).str.contains(r'In\s+employ|in employ', case=False, regex=True)
            )
            f = f[mask]
    
    # Try to filter by unit (Rates vs Levels) - be flexible with column names
    for col in f.columns:
        if 'unit' in col.lower():
            # Prefer "Rates" but also check if we need to exclude "Levels"
            rates_mask = f[col].astype(str).str.contains('Rate', case=False)
            levels_mask = f[col].astype(str).str.contains('Level', case=False)
            if rates_mask.any():
                f = f[rates_mask]
            elif levels_mask.any():
                f = f[~levels_mask]  # Exclude levels if that's all we have
            break
    
    # Try to filter to UK - be flexible with geography columns
    for col in f.columns:
        if any(x in col.lower() for x in ['uk', 'geog', 'country', 'region']):
            uk_mask = f[col].astype(str).str.contains(
                r'United Kingdom|K02000001|\bUK\b', case=False, regex=True
            )
            if uk_mask.any():
                f = f[uk_mask]
            break
    
    # Try seasonal adjustment filter
    for col in f.columns:
        if 'seasonal' in col.lower() or 'adjust' in col.lower():
            sa_mask = f[col].astype(str).str.contains('Seasonally Adjusted', case=False)
            if sa_mask.any():
                f = f[sa_mask]
            break
    
    # Age filter - look for age-related columns
    if prefer_age:
        for col in f.columns:
            if 'age' in col.lower():
                age_mask = pd.Series(False, index=f.index)
                for age_str in prefer_age:
                    age_mask |= f[col].astype(str).str.contains(
                        age_str.replace('+', r'\+'), case=False, regex=True
                    )
                if age_mask.any():
                    f = f[age_mask]
                break
        
        # Also check if age info is embedded in the indicator column
        if indicator_col and prefer_age:
            age_mask = pd.Series(False, index=f.index)
            for age_str in prefer_age:
                age_mask |= f[indicator_col].astype(str).str.contains(
                    age_str.replace('+', r'\+'), case=False, regex=True
                )
            if age_mask.any():
                f = f[age_mask]
    
    # Sex filter - prefer "All" if available
    for col in f.columns:
        if 'sex' in col.lower() or col.lower() == 'gender':
            all_mask = f[col].astype(str).str.contains('All', case=False)
            if all_mask.any():
                f = f[all_mask]
            break
    
    return f

def _unique_guard(df: pd.DataFrame, key: list[str], ctx: str):
    dups = df.duplicated(key, keep=False)
    if dups.any():
        sample = df.loc[dups, key + ["value"]].head(6).to_dict(orient="records")
        log.warning("%s: non-unique rows on %s; dropping duplicates. sample=%s", ctx, key, sample)
        df.drop_duplicates(key, keep="last", inplace=True)

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
    
    # Filter for overall GDP (A-T : Monthly GDP), not sector-specific series
    if 'UnofficialStandardIndustrialClassification' in df.columns:
        df = df[df['UnofficialStandardIndustrialClassification'] == 'A-T : Monthly GDP']
        log.info("GDP: Filtered for A-T : Monthly GDP")
    elif 'Aggregate' in df.columns:
        # Fallback - look for overall/total indicators
        df = df[df['Aggregate'].str.contains('Monthly GDP|Total|Overall', case=False, na=False)]
        log.info("GDP: Filtered using Aggregate column")
    else:
        log.warning("GDP: No filter column found. Columns: %s", df.columns.tolist()[:10])
    
    tidy = to_tidy_uk(df, metric_id="uk_gdp_m_index", unit="index(2016=100)", source="ONS")
    # ensure monthly only (gdp monthly dataset is M)
    tidy["freq"] = "M"
    
    # Debug: show recent values to verify we're getting ~102-105 not 102.0054
    if not tidy.empty:
        recent = tidy.sort_values('period').tail(5)[['period', 'value']]
        log.info("GDP recent values: %s", recent.to_dict('records'))
    
    return tidy

def fetch_cpih() -> pd.DataFrame:
    df = ons_read_latest_csv("cpih01", edition_hint="time-series")
    save_raw(df, "ons_cpih_raw")
    
    # Filter for Overall Index only (not subcategories)
    # Look for the aggregate column and filter for overall index
    if 'Aggregate' in df.columns:
        df = df[df['Aggregate'] == 'Overall Index']
        log.info("CPIH: Filtered for Overall Index using Aggregate column")
    elif 'cpih1dim1aggid' in df.columns:
        df = df[df['cpih1dim1aggid'] == 'CP00']  # CP00 is the code for Overall Index
        log.info("CPIH: Filtered for CP00 using cpih1dim1aggid column")
    else:
        # Log what columns exist if we can't find the expected ones
        log.warning("CPIH: No Aggregate or cpih1dim1aggid column found. Columns: %s", df.columns.tolist()[:10])
    
    tidy = to_tidy_uk(df, metric_id="uk_cpih_index", unit="index(2015=100)", source="ONS")
    tidy["freq"] = "M"
    
    # Debug: show recent values to verify we're getting ~139 not 173
    if not tidy.empty:
        recent = tidy.sort_values('period').tail(5)[['period', 'value']]
        log.info("CPIH recent values: %s", recent.to_dict('records'))
    
    return tidy

def fetch_labour_headlines() -> pd.DataFrame:
    """
    Build UK macro labour 'anchor' series:
      - uk_unemp_rate  (headline unemployment rate, 16+)
      - uk_emp_rate    (headline employment rate, 16–64)
    Output periods are monthly stamps of rolling 3-month windows → freq='M3'.
    """
    df = ons_read_latest_csv("labour-market", edition_hint=None)
    save_raw(df, "ons_labour_raw")  # raw snapshot (unaltered)
    
    # Debug: Check what seasonal adjustment values actually exist
    log.info("Seasonal adjustment values in data: %s", df['seasonal-adjustment'].unique().tolist())
    
    # Unemployment rate: unemployed as % of economically active
    # Looking for all-adults + 16-64 (16+ doesn't exist for unemployment)
    unemp_df = df[
        (df['economic-activity'] == 'unemployed') & 
        (df['unit-of-measure'] == 'rates') &
        (df['seasonal-adjustment'] == 'seasonal-adjustment') &  # The actual value is 'seasonal-adjustment'
        (df['sex'] == 'all-adults') &
        (df['age-groups'] == '16-64')  # 16+ doesn't exist for unemployment, use 16-64
    ].copy()
    
    log.info("Unemployment rate: found %d rows after filtering", len(unemp_df))
    if not unemp_df.empty:
        log.info("Unemployment sample:\n%s", unemp_df[['Time', 'v4_0', 'age-groups', 'sex']].head(3))
    
    tidy_u = to_tidy_uk(unemp_df, metric_id="uk_unemp_rate", unit="percent", source="ONS")

    # Employment rate: in-employment as % of population  
    # Looking for all-adults + 16+ (which exists for employment)
    emp_df = df[
        (df['economic-activity'] == 'in-employment') & 
        (df['unit-of-measure'] == 'rates') &
        (df['seasonal-adjustment'] == 'seasonal-adjustment') &  # The actual value is 'seasonal-adjustment'
        (df['sex'] == 'all-adults') &
        (df['age-groups'] == '16+')  # 16+ exists for employment
    ].copy()
    
    log.info("Employment rate: found %d rows after filtering", len(emp_df))
    if not emp_df.empty:
        log.info("Employment sample:\n%s", emp_df[['Time', 'v4_0', 'age-groups', 'sex']].head(3))
    
    tidy_e = to_tidy_uk(emp_df, metric_id="uk_emp_rate", unit="percent", source="ONS")

    # Combine both series
    tidy = pd.concat([tidy_u, tidy_e], ignore_index=True)

    # Labour series are rolling 3-month windows; force freq label to 'M3'
    tidy.loc[:, "freq"] = "M3"

    # Drop impossible percentages (catches any lingering level rows)
    bad = (tidy["metric_id"].isin(["uk_emp_rate", "uk_unemp_rate"])) & (
        (tidy["value"] < 0) | (tidy["value"] > 100)
    )
    if bad.any():
        dropped = tidy.loc[bad, ["metric_id", "period", "value"]].head(6).to_dict("records")
        log.warning("Dropping %d non-% rate rows (sample=%s)", bad.sum(), dropped)
        tidy = tidy[~bad]

    # One row per metric_id + period
    tidy.sort_values(["metric_id", "period"], inplace=True)
    _unique_guard(tidy, key=["metric_id", "period"], ctx="labour_headlines")

    # Debug: verify what was found
    if tidy.empty:
        log.warning("No labour headline rows found! Check column names and keywords.")
    else:
        log.info("Labour headlines preview:\n%s", tidy.head(6).to_string(index=False))

    # Keep only tidy essentials (already in your canonical macro tidy schema)
    return tidy.reset_index(drop=True)
    
    # Direct filtering based on actual ONS data structure
    # All data is already UK-only (K02000001), so no geography filter needed
    
    # Unemployment rate: unemployed as % of economically active
    unemp_df = df[
        (df['economic-activity'] == 'unemployed') & 
        (df['unit-of-measure'] == 'rates') &
        (df['seasonal-adjustment'].str.contains('Seasonally Adjusted', case=False))
    ].copy()
    
    # Filter for broadest age group and all sexes
    if 'age-groups' in unemp_df.columns:
        # Check what's available - prefer 16+ or all-adults
        age_options = unemp_df['age-groups'].unique()
        if '16+' in age_options:
            unemp_df = unemp_df[unemp_df['age-groups'] == '16+']
        elif 'all-adults' in age_options:
            unemp_df = unemp_df[unemp_df['age-groups'] == 'all-adults']
        else:
            log.warning("Unemployment: No 16+ found, using: %s", age_options[:5])
    
    if 'sex' in unemp_df.columns:
        sex_options = unemp_df['sex'].unique()
        if 'all-adults' in sex_options:
            unemp_df = unemp_df[unemp_df['sex'] == 'all-adults']
        elif 'all' in sex_options:
            unemp_df = unemp_df[unemp_df['sex'] == 'all']
        else:
            log.warning("Unemployment: No 'all' sex found, using: %s", sex_options)
    
    log.info("Unemployment rate: found %d rows after filtering", len(unemp_df))
    if not unemp_df.empty:
        log.info("Unemployment sample:\n%s", unemp_df[['Time', 'v4_0', 'age-groups', 'sex']].head(3))
    
    tidy_u = to_tidy_uk(unemp_df, metric_id="uk_unemp_rate", unit="percent", source="ONS")

    # Employment rate: in-employment as % of population
    emp_df = df[
        (df['economic-activity'] == 'in-employment') & 
        (df['unit-of-measure'] == 'rates') &
        (df['seasonal-adjustment'].str.contains('Seasonally Adjusted', case=False))
    ].copy()
    
    # Filter for appropriate age group and all sexes
    if 'age-groups' in emp_df.columns:
        age_options = emp_df['age-groups'].unique()
        if '16-64' in age_options:
            emp_df = emp_df[emp_df['age-groups'] == '16-64']
        elif '16+' in age_options:
            emp_df = emp_df[emp_df['age-groups'] == '16+']
        elif 'all-adults' in age_options:
            emp_df = emp_df[emp_df['age-groups'] == 'all-adults']
        else:
            log.warning("Employment: No standard age group found, using: %s", age_options[:5])
    
    if 'sex' in emp_df.columns:
        sex_options = emp_df['sex'].unique()
        if 'all-adults' in sex_options:
            emp_df = emp_df[emp_df['sex'] == 'all-adults']
        elif 'all' in sex_options:
            emp_df = emp_df[emp_df['sex'] == 'all']
        else:
            log.warning("Employment: No 'all' sex found, using: %s", sex_options)
    
    log.info("Employment rate: found %d rows after filtering", len(emp_df))
    if not emp_df.empty:
        log.info("Employment sample:\n%s", emp_df[['Time', 'v4_0', 'age-groups', 'sex']].head(3))

    tidy = pd.concat([tidy_u, tidy_e], ignore_index=True)

    # Labour series are rolling 3-month windows; force freq label to 'M3'
    tidy.loc[:, "freq"] = "M3"

    # Drop impossible percentages (catches any lingering level rows)
    bad = (tidy["metric_id"].isin(["uk_emp_rate", "uk_unemp_rate"])) & (
        (tidy["value"] < 0) | (tidy["value"] > 100)
    )
    if bad.any():
        dropped = tidy.loc[bad, ["metric_id", "period", "value"]].head(6).to_dict("records")
        log.warning("Dropping %d non-% rate rows (sample=%s)", bad.sum(), dropped)
        tidy = tidy[~bad]

    # One row per metric_id + period
    tidy.sort_values(["metric_id", "period"], inplace=True)
    _unique_guard(tidy, key=["metric_id", "period"], ctx="labour_headlines")

    # Debug: verify what was found
    if tidy.empty:
        log.warning("No labour headline rows found! Check column names and keywords.")
    else:
        log.info("Labour headlines preview:\n%s", tidy.head(6).to_string(index=False))

    # Keep only tidy essentials (already in your canonical macro tidy schema)
    return tidy.reset_index(drop=True)

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

    # Labour headlines (unemp 16+, emp 16–64), tidy only
    try:
        labour = fetch_labour_headlines()
        silver_frames.append(labour)
        _write_duck("bronze.macro_labour_headlines", labour)  # tidy snapshot of the two series
        log.info(
            "LABOUR (headlines): rows=%d | metrics=%s | sample [%s..%s]",
            len(labour),
            sorted(labour["metric_id"].unique()),
            labour["period"].min(),
            labour["period"].max(),
        )
    except Exception as e:
        failures["labour_headlines"] = str(e)
        log.exception("Labour headlines failed")

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

    # Expect uniqueness within the unified table too
    _unique_guard(silver, key=["metric_id", "period", "region_code"], ctx="uk_macro_history")

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