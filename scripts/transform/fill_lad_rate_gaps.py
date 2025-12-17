#!/usr/bin/env python3
"""
Fill LAD Rate Gaps for Complete Time Series

Fills missing unemployment_rate_pct and employment_rate_pct values
using interpolation or parent inheritance.

CRITICAL: This script ONLY fills gaps BETWEEN two known ONS data points.
It does NOT extrapolate beyond the last known ONS year. Years beyond the
last ONS year are left empty and will be handled by the forecasting script.

Examples:
- Cornwall: ONS has 2020, 2021, 2022, 2025 → fills 2023-2024 (gap between 2022 and 2025)
- Hackney: ONS has 2020, 2021 → stops at 2021 (no 2022-2025, forecasting will handle)

Inputs:
- silver.lad_history (with gaps)
- silver.itl3_history (for inheritance)
- bronze.unemp_rate_lad_raw / bronze.emp_rate_lad_raw (to identify original ONS years)
- data/reference/master_2025_geography_lookup.csv

Outputs:
- silver.lad_history (updated with filled gaps, but NO extrapolation beyond last ONS year)
- data/logs/lad_gap_fill_summary.json

Data quality values:
- 'ONS': Original ONS data (from bronze layer)
- 'interpolated': Gap-filled data (interpolated or inherited from parent ITL3)
  NOTE: Only gaps BETWEEN known ONS years are filled. Years beyond the last
  known ONS year are left empty for the forecasting script to handle.
"""

import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import duckdb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("fill_lad_gaps")

DUCK_PATH = Path("data/lake/warehouse.duckdb")
LOOKUP_PATH = Path("data/reference/master_2025_geography_lookup.csv")
LOG_DIR = Path("data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

RATE_METRICS = ['unemployment_rate_pct', 'employment_rate_pct']
MIN_YEAR = 2005  # Earliest year to consider
MAX_YEAR = 2025  # Latest historical year


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, set]]:
    """Load LAD history, ITL3 history, geography lookup, and original bronze years."""
    con = duckdb.connect(str(DUCK_PATH), read_only=True)
    
    # Load LAD rate data
    lad_rates = con.execute(f"""
        SELECT region_code, region_name, metric_id, period, value
        FROM silver.lad_history
        WHERE metric_id IN ({','.join([f"'{m}'" for m in RATE_METRICS])})
        ORDER BY region_code, metric_id, period
    """).fetchdf()
    
    # Load ITL3 rate data (for inheritance)
    itl3_rates = con.execute(f"""
        SELECT region_code, metric_id, period, value
        FROM silver.itl3_history
        WHERE metric_id IN ({','.join([f"'{m}'" for m in RATE_METRICS])})
        ORDER BY region_code, metric_id, period
    """).fetchdf()
    
    # Load original bronze data to identify which years were originally available
    # This helps us identify which values were gap-filled vs original ONS
    original_years_by_lad = {}
    try:
        # Try to get original years from bronze layer
        # Bronze schema: GEOGRAPHY_CODE, DATE_CODE (format "YYYY-06"), OBS_VALUE
        # Extract year from DATE_CODE and map to region_code/period
        bronze_data = con.execute("""
            SELECT 
                GEOGRAPHY_CODE as region_code,
                CAST(SUBSTRING(DATE_CODE, 1, 4) AS INTEGER) as period
            FROM bronze.unemp_rate_lad_raw
            WHERE OBS_VALUE IS NOT NULL
        """).fetchdf()
        
        if not bronze_data.empty:
            for _, row in bronze_data.iterrows():
                key = (row['region_code'], 'unemployment_rate_pct')
                if key not in original_years_by_lad:
                    original_years_by_lad[key] = set()
                original_years_by_lad[key].add(int(row['period']))
        
        # Also check employment rate
        bronze_emp = con.execute("""
            SELECT 
                GEOGRAPHY_CODE as region_code,
                CAST(SUBSTRING(DATE_CODE, 1, 4) AS INTEGER) as period
            FROM bronze.emp_rate_lad_raw
            WHERE OBS_VALUE IS NOT NULL
        """).fetchdf()
        
        if not bronze_emp.empty:
            for _, row in bronze_emp.iterrows():
                key = (row['region_code'], 'employment_rate_pct')
                if key not in original_years_by_lad:
                    original_years_by_lad[key] = set()
                original_years_by_lad[key].add(int(row['period']))
        
        log.info(f"Loaded original years for {len(original_years_by_lad)} LAD/metric combinations from bronze")
    except Exception as e:
        log.warning(f"Could not load original years from bronze: {e}")
        log.warning("Will use pattern detection to identify interpolated values")
    
    con.close()
    
    # Load lookup for LAD → ITL mapping (includes all parent geographies)
    lookup = pd.read_csv(LOOKUP_PATH)
    lookup.columns = [c.replace('\ufeff', '') for c in lookup.columns]
    
    # Build LAD → ITL3 mapping
    lad_to_itl3 = lookup[['LAD25CD', 'ITL325CD']].drop_duplicates()
    lad_to_itl3 = lad_to_itl3.rename(columns={'LAD25CD': 'region_code', 'ITL325CD': 'itl3_code'})
    
    # Build full geography mapping (LAD → ITL3/ITL2/ITL1)
    geo_mapping = lookup[[
        'LAD25CD', 'ITL325CD', 'ITL325NM', 
        'ITL225CD', 'ITL225NM', 
        'ITL125CD', 'ITL125NM'
    ]].drop_duplicates().rename(columns={
        'LAD25CD': 'region_code',
        'ITL325CD': 'itl3_code',
        'ITL325NM': 'itl3_name',
        'ITL225CD': 'itl2_code',
        'ITL225NM': 'itl2_name',
        'ITL125CD': 'itl1_code',
        'ITL125NM': 'itl1_name'
    })
    
    # Merge geography mapping into LAD rates
    if not lad_rates.empty:
        lad_rates = lad_rates.merge(
            geo_mapping,
            on='region_code',
            how='left'
        )
    
    log.info(f"Loaded {len(lad_rates)} LAD rate observations")
    log.info(f"Loaded {len(itl3_rates)} ITL3 rate observations")
    log.info(f"Loaded {len(lad_to_itl3)} LAD→ITL3 mappings")
    
    return lad_rates, itl3_rates, lad_to_itl3, original_years_by_lad


def get_expected_years(df: pd.DataFrame) -> List[int]:
    """Get the range of years we expect data for."""
    if df.empty:
        return list(range(MIN_YEAR, MAX_YEAR + 1))
    
    min_year = max(int(df['period'].min()), MIN_YEAR)
    max_year = min(int(df['period'].max()), MAX_YEAR)
    return list(range(min_year, max_year + 1))


def interpolate_gaps(series: pd.Series, original_years: Optional[set] = None) -> Tuple[pd.Series, pd.Series]:
    """
    Interpolate missing values in a time series.
    
    ONLY fills gaps BETWEEN two known ONS years (not beyond last known year).
    
    Args:
        series: Time series with potential gaps
        original_years: Set of years that exist in original ONS data (bronze)
    
    Returns:
        (filled_series, quality_series) where quality is 'ONS' or 'interpolated'
    """
    filled = series.copy()
    quality = pd.Series('ONS', index=series.index)
    
    # Mark NaN positions
    is_gap = series.isna()
    
    if not is_gap.any():
        return filled, quality
    
    # If we have original_years, only interpolate WITHIN the range of original years
    # Don't extrapolate beyond the last known ONS year
    if original_years is not None and len(original_years) > 0:
        min_original = min(original_years)
        max_original = max(original_years)
        
        # Create a mask for years within original ONS range
        mask_within_range = (series.index >= min_original) & (series.index <= max_original)
        
        # Only interpolate gaps that are WITHIN the original range
        # AND have bookends on both sides (limit_area='inside' ensures this)
        series_within_range = series[mask_within_range].copy()
        
        if series_within_range.isna().any():
            # Interpolate only within this range
            filled_within = series_within_range.interpolate(method='linear', limit_area='inside')
            
            # Update filled series only for years within range
            filled[mask_within_range] = filled_within
            
            # Mark interpolated values (only those that were gaps and got filled)
            interp_mask = mask_within_range & is_gap & filled.notna()
            quality[interp_mask] = 'interpolated'
        
        # Explicitly ensure years beyond max_original remain NaN
        beyond_mask = series.index > max_original
        filled[beyond_mask] = np.nan
        quality[beyond_mask] = 'ONS'  # Will be overwritten if filled later, but ensures we don't mark as interpolated
    else:
        # Fallback: interpolate all gaps (if we don't know original years)
        filled = series.interpolate(method='linear', limit_area='inside')
        quality[is_gap & filled.notna()] = 'interpolated'
    
    return filled, quality


def fill_from_parent(
    lad_code: str,
    metric: str,
    missing_years: List[int],
    itl3_rates: pd.DataFrame,
    lad_to_itl3: pd.DataFrame,
    con: duckdb.DuckDBPyConnection = None
) -> Dict[int, Tuple[float, str]]:
    """
    Fill missing years from parent ITL3, with fallback to ITL2/ITL1.
    
    Returns:
        Dict mapping year -> (value, quality)
    """
    filled = {}
    
    # Get parent ITL3
    mapping = lad_to_itl3[lad_to_itl3['region_code'] == lad_code]
    if mapping.empty:
        return filled
    
    itl3_code = mapping['itl3_code'].iloc[0]
    
    # Get ITL3 values for this metric
    itl3_metric = itl3_rates[
        (itl3_rates['region_code'] == itl3_code) &
        (itl3_rates['metric_id'] == metric)
    ].set_index('period')['value']
    
    # Fill from ITL3
    for year in missing_years:
        if year in itl3_metric.index:
            filled[year] = (float(itl3_metric[year]), 'interpolated')
    
    # If still missing, try ITL2 as fallback
    still_missing = [y for y in missing_years if y not in filled]
    if still_missing and con is not None:
        try:
            # Get ITL2 code from lookup
            lookup = pd.read_csv(LOOKUP_PATH)
            lookup.columns = [c.replace('\ufeff', '') for c in lookup.columns]
            lad_lookup = lookup[lookup['LAD25CD'] == lad_code]
            if not lad_lookup.empty:
                itl2_code = lad_lookup['ITL225CD'].iloc[0]
                
                # Get ITL2 values
                itl2_metric = con.execute(f"""
                    SELECT period, value
                    FROM silver.itl2_history
                    WHERE region_code = '{itl2_code}'
                    AND metric_id = '{metric}'
                """).fetchdf()
                
                if not itl2_metric.empty:
                    itl2_series = itl2_metric.set_index('period')['value']
                    for year in still_missing:
                        if year in itl2_series.index:
                            filled[year] = (float(itl2_series[year]), 'interpolated')
        except Exception as e:
            log.warning(f"  Could not fill from ITL2 for {lad_code}: {e}")
    
    return filled


def fill_gaps_for_lad(
    lad_code: str,
    lad_name: str,
    metric: str,
    lad_data: pd.DataFrame,
    itl3_rates: pd.DataFrame,
    lad_to_itl3: pd.DataFrame,
    expected_years: List[int],
    original_years: Optional[set] = None,
    con: duckdb.DuckDBPyConnection = None
) -> pd.DataFrame:
    """
    Fill all gaps for a single LAD/metric combination.
    
    Returns:
        DataFrame with complete time series and data_quality column
    """
    # Get existing data as series
    existing = lad_data.set_index('period')['value']
    
    # Identify which years were originally present
    # If we have original_years from bronze, use that. Otherwise, use what's in lad_data.
    if original_years is None:
        # Fallback: use years in lad_data that actually have observed values.
        #
        # IMPORTANT:
        # Some series (notably NI LAD rates) may have placeholder rows for earlier years
        # with NULL values (e.g. created by prior pipeline runs). Those years must NOT
        # be treated as "original", otherwise we will try to gap-fill across long
        # leading NULL stretches and can accidentally propagate NULLs/zeros upstream.
        original_years = set(lad_data.loc[lad_data["value"].notna(), "period"].values)
    
    # CRITICAL: Filter out years beyond the last known ONS year
    # These should be handled by forecasting, not gap-filling
    # This ensures we don't treat previously extrapolated data as "historical"
    if original_years is not None and len(original_years) > 0:
        max_original = max(original_years)
        # Remove any existing data beyond max_original (it was extrapolated, not ONS)
        # The forecasting script will detect max_original as the last historical year
        existing = existing[existing.index <= max_original]
        if len(existing[existing.index > max_original]) > 0:
            log.info(f"  {lad_code}/{metric}: Removed {len(existing[existing.index > max_original])} years beyond {max_original} (last ONS year) - will be forecast")
    
    # Create full series with all expected years
    full_series = pd.Series(index=expected_years, dtype=float)
    full_series.update(existing)
    
    # Track quality - mark years that weren't in original data as interpolated
    quality = pd.Series('ONS', index=expected_years)
    
    # If we have original_years from bronze, use that
    if original_years is not None:
        for year in expected_years:
            if year not in original_years:
                # This year was missing from original data, will be filled - mark as None first
                quality[year] = None
    else:
        # Fallback: Use pattern detection - check if values match linear interpolation
        # If a year's value is exactly between two surrounding years, it's likely interpolated
        pattern_detected = []
        for year in expected_years:
            if year in full_series.index and not pd.isna(full_series[year]):
                # Check if this year is between two other years with data
                # Find the closest previous and next years with data
                prev_years = [y for y in expected_years if y < year and y in full_series.index and not pd.isna(full_series[y])]
                next_years = [y for y in expected_years if y > year and y in full_series.index and not pd.isna(full_series[y])]
                
                if prev_years and next_years:
                    # Use the closest previous and next years (not just max/min)
                    prev_year = max(prev_years)  # Closest previous
                    next_year = min(next_years)  # Closest next
                    prev_val = full_series[prev_year]
                    next_val = full_series[next_year]
                    current_val = full_series[year]
                    
                    # Calculate expected interpolated value
                    total_gap = next_year - prev_year
                    if total_gap > 0:
                        current_gap = year - prev_year
                        expected_val = prev_val + (next_val - prev_val) * (current_gap / total_gap)
                        
                        # If value matches interpolation pattern (within 0.01 tolerance), mark as interpolated
                        diff = abs(current_val - expected_val)
                        if diff < 0.01:
                            quality.loc[year] = 'interpolated'  # Use .loc to ensure proper assignment
                            pattern_detected.append(year)
        
        if pattern_detected:
            log.info(f"  Pattern detected {len(pattern_detected)} interpolated years for {lad_code}/{metric}: {pattern_detected}")
            # Debug: verify quality was set
            for year in pattern_detected:
                if quality.loc[year] != 'interpolated':
                    log.warning(f"  WARNING: {year} was pattern-detected but quality is {quality.loc[year]}")
    
    # Step 1: Interpolate where bookends exist (ONLY between known ONS years)
    filled, interp_quality = interpolate_gaps(full_series, original_years)
    # Update quality for interpolated values (from actual gaps)
    # Only update years that were actually interpolated (had NaN values)
    # Don't overwrite pattern-detected interpolated values
    for year in interp_quality.index:
        if interp_quality[year] == 'interpolated' and quality[year] != 'interpolated':
            quality[year] = 'interpolated'
    full_series = filled
    
    # Step 2: Fill remaining gaps from parent ITL3 (with ITL2 fallback)
    # BUT ONLY if the parent has ONS data for that year (not forecast)
    still_missing = full_series[full_series.isna()].index.tolist()
    if still_missing and original_years is not None:
        # Only fill from parent if the missing year is within the range of original ONS years
        # (i.e., it's a gap, not beyond the last known year)
        max_original = max(original_years) if original_years else None
        if max_original:
            # Only fill gaps up to max_original - beyond that is forecasting territory
            gaps_to_fill = [y for y in still_missing if y <= max_original]
            if gaps_to_fill:
                parent_fills = fill_from_parent(
                    lad_code, metric, gaps_to_fill, itl3_rates, lad_to_itl3, con
                )
                for year, (value, qual) in parent_fills.items():
                    full_series[year] = value
                    quality[year] = qual
    
    # Step 3: REMOVED - Do NOT extrapolate beyond last known ONS year
    # Years beyond the last known ONS year should be left empty for the forecasting script
    # The forecasting script will:
    #   1. Detect the last ONS year (max(original_years))
    #   2. Forecast from (last_ONS_year + 1) onwards
    #   3. Mark those years as 'forecast' (not 'interpolated')
    
    # Get parent geography from original data (first non-null row)
    geo_cols = ['itl3_code', 'itl3_name', 'itl2_code', 'itl2_name', 'itl1_code', 'itl1_name']
    geo_dict = {}
    for col in geo_cols:
        if col in lad_data.columns:
            non_null = lad_data[lad_data[col].notna()]
            if not non_null.empty:
                geo_dict[col] = non_null[col].iloc[0]
    
    # Build output DataFrame
    # Ensure quality index matches full_series index
    quality_aligned = quality.reindex(full_series.index, fill_value='ONS')
    
    # Replace None values with 'interpolated' (these are filled gaps)
    quality_aligned = quality_aligned.fillna('interpolated')
    
    result = pd.DataFrame({
        'region_code': lad_code,
        'region_name': lad_name,
        'metric_id': metric,
        'period': full_series.index,
        'value': full_series.values,
        'data_quality': quality_aligned.values
    })
    
    # Add parent geography columns if available
    for col, val in geo_dict.items():
        result[col] = val
    
    # Drop still-missing (shouldn't happen if ITL3 is complete)
    result = result.dropna(subset=['value'])
    
    # CRITICAL: Filter out any years beyond the last known ONS year
    # These should be handled by forecasting, not gap-filling
    if original_years is not None and len(original_years) > 0:
        max_original = max(original_years)
        before_filter = len(result)
        result = result[result['period'] <= max_original]
        after_filter = len(result)
        if before_filter > after_filter:
            log.debug(f"  {lad_code}/{metric}: Filtered out {before_filter - after_filter} years beyond {max_original} (last ONS year)")
    
    return result


def fill_all_gaps(
    lad_rates: pd.DataFrame,
    itl3_rates: pd.DataFrame,
    lad_to_itl3: pd.DataFrame,
    original_years_by_lad: Dict[str, set],
    con: duckdb.DuckDBPyConnection = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Fill gaps for all LADs and rate metrics.
    
    Returns:
        (filled_df, summary_stats)
    """
    results = []
    stats = {
        'total_lads': 0,
        'total_filled_interpolated': 0,
        'total_filled_inherited': 0,
        'total_still_missing': 0,
        'by_metric': {}
    }
    
    expected_years = get_expected_years(lad_rates)
    log.info(f"Expected years: {min(expected_years)} - {max(expected_years)}")
    
    for metric in RATE_METRICS:
        log.info(f"\nProcessing {metric}...")
        metric_stats = {'interpolated': 0, 'inherited': 0, 'missing': 0}
        
        metric_data = lad_rates[lad_rates['metric_id'] == metric]
        lad_codes = metric_data['region_code'].unique()
        
        for lad_code in lad_codes:
            lad_data = metric_data[metric_data['region_code'] == lad_code]
            lad_name = lad_data['region_name'].iloc[0]
            
            # Get original years for this LAD/metric from bronze
            original_years = original_years_by_lad.get((lad_code, metric))
            
            filled = fill_gaps_for_lad(
                lad_code, lad_name, metric,
                lad_data, itl3_rates, lad_to_itl3, expected_years, original_years, con
            )
            
            # Count fills
            quality_counts = filled['data_quality'].value_counts()
            metric_stats['interpolated'] += quality_counts.get('interpolated', 0)
            metric_stats['inherited'] += quality_counts.get('interpolated', 0)  # Both interpolated and inherited are 'interpolated'
            
            results.append(filled)
        
        stats['by_metric'][metric] = metric_stats
        stats['total_filled_interpolated'] += metric_stats['interpolated']
        stats['total_filled_inherited'] += metric_stats['inherited']
        
        total_interpolated = metric_stats['interpolated'] + metric_stats['inherited']
        log.info(f"  {metric}: {total_interpolated} interpolated (gap-filled)")
    
    stats['total_lads'] = lad_rates['region_code'].nunique()
    
    filled_df = pd.concat(results, ignore_index=True)
    return filled_df, stats


def update_silver_layer(filled_rates: pd.DataFrame):
    """
    Update silver.lad_history with filled rate data.
    
    Preserves non-rate metrics, replaces rate metrics with filled versions.
    """
    con = duckdb.connect(str(DUCK_PATH))
    
    # Load existing non-rate data
    non_rates = con.execute(f"""
        SELECT *
        FROM silver.lad_history
        WHERE metric_id NOT IN ({','.join([f"'{m}'" for m in RATE_METRICS])})
    """).fetchdf()
    
    # Add data_quality column to non-rates if missing
    if 'data_quality' not in non_rates.columns:
        non_rates['data_quality'] = 'ONS'
    
    # Get existing rate data to preserve other columns
    existing_rates = con.execute(f"""
        SELECT *
        FROM silver.lad_history
        WHERE metric_id IN ({','.join([f"'{m}'" for m in RATE_METRICS])})
        LIMIT 1
    """).fetchdf()
    
    # Ensure filled_rates has all required columns
    if not existing_rates.empty:
        required_cols = existing_rates.columns.tolist()
        for col in required_cols:
            if col not in filled_rates.columns:
                if col == 'unit':
                    filled_rates[col] = 'percent'
                elif col == 'freq':
                    filled_rates[col] = 'A'
                elif col == 'source':
                    filled_rates[col] = 'NOMIS_LAD_gap_filled'
                elif col == 'vintage':
                    filled_rates[col] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                elif col == 'region_level':
                    filled_rates[col] = 'LAD'
                elif col == 'geo_hierarchy':
                    filled_rates[col] = 'LAD>ITL3>ITL2>ITL1'
                else:
                    filled_rates[col] = None
    else:
        # If no existing rate data, add standard columns
        if 'unit' not in filled_rates.columns:
            filled_rates['unit'] = 'percent'
        if 'freq' not in filled_rates.columns:
            filled_rates['freq'] = 'A'
        if 'source' not in filled_rates.columns:
            filled_rates['source'] = 'NOMIS_LAD_gap_filled'
        if 'vintage' not in filled_rates.columns:
            filled_rates['vintage'] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if 'region_level' not in filled_rates.columns:
            filled_rates['region_level'] = 'LAD'
        if 'geo_hierarchy' not in filled_rates.columns:
            filled_rates['geo_hierarchy'] = 'LAD>ITL3>ITL2>ITL1'
    
    # Combine
    combined = pd.concat([non_rates, filled_rates], ignore_index=True)
    combined = combined.sort_values(['metric_id', 'region_code', 'period']).reset_index(drop=True)
    
    # Write back
    con.execute("CREATE SCHEMA IF NOT EXISTS silver")
    con.register("df_tmp", combined)
    con.execute("CREATE OR REPLACE TABLE silver.lad_history AS SELECT * FROM df_tmp")
    
    log.info(f"Updated silver.lad_history: {len(combined)} total rows")

    # IMPORTANT:
    # Broad_transform.py loads LAD rate metrics from the metric-specific tables
    # (silver.lad_employment_rate_history / silver.lad_unemployment_rate_history),
    # not from silver.lad_history. If we only update silver.lad_history, transform
    # will re-introduce any NULL placeholder rows from the old rate tables.
    #
    # So we also fully replace the metric-specific rate tables here, using the
    # filled (non-null) series.
    for metric_id, table_name in [
        ("employment_rate_pct", "lad_employment_rate_history"),
        ("unemployment_rate_pct", "lad_unemployment_rate_history"),
    ]:
        df_m = filled_rates[filled_rates["metric_id"] == metric_id].copy()
        # Ensure no NULL values are written (these cause bogus 0s when aggregated)
        df_m = df_m.dropna(subset=["value"])

        # Align to existing table schema if it exists
        try:
            cols = con.execute(f"pragma table_info('silver.{table_name}')").fetchdf()["name"].tolist()
            df_m = df_m.reindex(columns=cols)
        except Exception:
            # Table might not exist yet; write what we have (duckdb will infer)
            pass

        con.register("df_metric_tmp", df_m)
        con.execute(f"CREATE OR REPLACE TABLE silver.{table_name} AS SELECT * FROM df_metric_tmp")
        log.info(f"Updated silver.{table_name}: {len(df_m)} rows")
    
    con.close()


def derive_data_quality(qualities: pd.Series) -> str:
    """
    Derive aggregate data_quality from constituent LADs.
    
    If any LAD is 'interpolated', the aggregate is 'interpolated'.
    Otherwise, 'ONS'.
    """
    if qualities.isna().all():
        return 'ONS'
    
    # Remove NaN values
    qualities_clean = qualities.dropna()
    
    if len(qualities_clean) == 0:
        return 'ONS'
    
    # If any LAD is interpolated, the aggregate is interpolated
    if 'interpolated' in qualities_clean.values:
        return 'interpolated'
    return 'ONS'


def re_aggregate_to_itl_levels(filled_rates: pd.DataFrame):
    """
    Re-aggregate filled LAD rate data to ITL3/ITL2/ITL1 levels.
    
    Uses population-weighted averaging for rate metrics.
    Propagates data_quality from constituent LADs.
    Updates silver.itl3_history, silver.itl2_history, silver.itl1_history.
    """
    con = duckdb.connect(str(DUCK_PATH))
    
    # Load population data for weighting
    pop_data = con.execute("""
        SELECT region_code, period, value as population
        FROM silver.lad_history
        WHERE metric_id = 'population_total'
    """).fetchdf()
    
    if pop_data.empty:
        log.warning("No population data found - skipping ITL re-aggregation")
        con.close()
        return
    
    # Ensure filled_rates has parent geography columns
    required_cols = ['itl3_code', 'itl3_name', 'itl2_code', 'itl2_name', 'itl1_code', 'itl1_name']
    missing_cols = [c for c in required_cols if c not in filled_rates.columns]
    if missing_cols:
        log.warning(f"Missing parent geography columns: {missing_cols} - skipping ITL re-aggregation")
        con.close()
        return
    
    vintage = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Helper function to aggregate rates to a level
    def aggregate_rates_to_level(rate_df: pd.DataFrame, level_code_col: str, level_name_col: str, region_level: str) -> pd.DataFrame:
        """Aggregate rate metrics using population-weighted averaging."""
        if rate_df.empty:
            return pd.DataFrame()
        
        # Get max and min population years
        max_pop_year = pop_data['period'].max()
        min_pop_year = pop_data['period'].min()
        
        # Merge rates with population (using clipped periods for years beyond population range)
        rate_df_adj = rate_df.copy()
        rate_df_adj['pop_period'] = rate_df_adj['period'].clip(lower=min_pop_year, upper=max_pop_year)
        pop_for_merge = pop_data.rename(columns={'period': 'pop_period'})
        
        rate_with_pop = rate_df_adj.merge(
            pop_for_merge,
            on=['region_code', 'pop_period'],
            how='left'
        )
        
        # Drop rows without population or parent geography
        rate_with_pop = rate_with_pop.dropna(subset=['population', level_code_col, level_name_col])
        
        if rate_with_pop.empty:
            return pd.DataFrame()
        
        # Calculate weighted values
        rate_with_pop['weighted_value'] = rate_with_pop['value'] * rate_with_pop['population']
        
        # Drop temporary pop_period column
        rate_with_pop = rate_with_pop.drop(columns=['pop_period'])
        
        # Aggregate: sum(weighted_value) / sum(population)
        agg = rate_with_pop.groupby([
            level_code_col, level_name_col, 'period', 'metric_id', 'unit'
        ], as_index=False).agg({
            'weighted_value': 'sum',
            'population': 'sum'
        })
        
        # Calculate final weighted average
        agg['value'] = agg['weighted_value'] / agg['population']
        agg = agg.drop(columns=['weighted_value', 'population'])
        
        # Rename to standard schema
        agg = agg.rename(columns={
            level_code_col: 'region_code',
            level_name_col: 'region_name'
        })
        
        # Derive data_quality from constituent LADs (before adding metadata)
        if 'data_quality' in rate_with_pop.columns:
            # Group by the same keys as aggregation to get data_quality for each aggregated row
            quality_agg = rate_with_pop.groupby([
                level_code_col, 'metric_id', 'period'
            ])['data_quality'].apply(derive_data_quality).reset_index()
            quality_agg = quality_agg.rename(columns={
                level_code_col: 'region_code',
                'data_quality': 'data_quality'
            })
            
            # Merge data_quality into aggregated results
            agg = agg.merge(quality_agg, on=['region_code', 'metric_id', 'period'], how='left')
            agg['data_quality'] = agg['data_quality'].fillna('ONS')
        else:
            agg['data_quality'] = 'ONS'
        
        # Add metadata
        agg['region_level'] = region_level
        agg['freq'] = 'A'
        agg['source'] = 'NOMIS_LAD_aggregated_popweighted'
        agg['vintage'] = vintage
        
        # Add geo_hierarchy
        if region_level == 'ITL3':
            agg['geo_hierarchy'] = 'ITL3>ITL2>ITL1'
        elif region_level == 'ITL2':
            agg['geo_hierarchy'] = 'ITL2>ITL1'
        elif region_level == 'ITL1':
            agg['geo_hierarchy'] = 'ITL1'
        
        # Reorder columns to match silver schema
        cols = ['region_code', 'region_name', 'region_level', 'metric_id',
                'period', 'value', 'unit', 'freq', 'source', 'vintage', 'geo_hierarchy', 'data_quality']
        agg = agg[[c for c in cols if c in agg.columns]]
        
        return agg.sort_values(['metric_id', 'region_code', 'period']).reset_index(drop=True)
    
    # Aggregate to ITL3
    log.info("\n[5/5] Re-aggregating LAD rates to ITL3...")
    itl3_rates = aggregate_rates_to_level(filled_rates, 'itl3_code', 'itl3_name', 'ITL3')
    
    if not itl3_rates.empty:
        # Load existing ITL3 data (non-rate metrics)
        existing_itl3 = con.execute("""
            SELECT *
            FROM silver.itl3_history
            WHERE metric_id NOT IN ('unemployment_rate_pct', 'employment_rate_pct')
        """).fetchdf()
        
        # Add data_quality column to existing ITL3 if missing
        if 'data_quality' not in existing_itl3.columns:
            existing_itl3['data_quality'] = 'ONS'
        
        # Combine and update
        combined_itl3 = pd.concat([existing_itl3, itl3_rates], ignore_index=True)
        combined_itl3 = combined_itl3.sort_values(['metric_id', 'region_code', 'period']).reset_index(drop=True)
        
        con.register("df_itl3", combined_itl3)
        con.execute("CREATE OR REPLACE TABLE silver.itl3_history AS SELECT * FROM df_itl3")
        
        # Log data_quality distribution
        quality_dist = itl3_rates['data_quality'].value_counts()
        log.info(f"  ✓ Updated silver.itl3_history: {len(combined_itl3)} total rows ({len(itl3_rates)} rate observations)")
        log.info(f"    data_quality: {dict(quality_dist)}")
    
    # Aggregate to ITL2
    log.info("  Re-aggregating LAD rates to ITL2...")
    itl2_rates = aggregate_rates_to_level(filled_rates, 'itl2_code', 'itl2_name', 'ITL2')
    
    if not itl2_rates.empty:
        # Load existing ITL2 data (non-rate metrics)
        existing_itl2 = con.execute("""
            SELECT *
            FROM silver.itl2_history
            WHERE metric_id NOT IN ('unemployment_rate_pct', 'employment_rate_pct')
        """).fetchdf()
        
        # Add data_quality column to existing ITL2 if missing
        if 'data_quality' not in existing_itl2.columns:
            existing_itl2['data_quality'] = 'ONS'
        
        # Combine and update
        combined_itl2 = pd.concat([existing_itl2, itl2_rates], ignore_index=True)
        combined_itl2 = combined_itl2.sort_values(['metric_id', 'region_code', 'period']).reset_index(drop=True)
        
        con.register("df_itl2", combined_itl2)
        con.execute("CREATE OR REPLACE TABLE silver.itl2_history AS SELECT * FROM df_itl2")
        
        # Log data_quality distribution
        quality_dist = itl2_rates['data_quality'].value_counts()
        log.info(f"  ✓ Updated silver.itl2_history: {len(combined_itl2)} total rows ({len(itl2_rates)} rate observations)")
        log.info(f"    data_quality: {dict(quality_dist)}")
    
    # Aggregate to ITL1
    log.info("  Re-aggregating LAD rates to ITL1...")
    itl1_rates = aggregate_rates_to_level(filled_rates, 'itl1_code', 'itl1_name', 'ITL1')
    
    if not itl1_rates.empty:
        # Load existing ITL1 data (non-rate metrics)
        existing_itl1 = con.execute("""
            SELECT *
            FROM silver.itl1_history
            WHERE metric_id NOT IN ('unemployment_rate_pct', 'employment_rate_pct')
        """).fetchdf()
        
        # Combine and update
        combined_itl1 = pd.concat([existing_itl1, itl1_rates], ignore_index=True)
        combined_itl1 = combined_itl1.sort_values(['metric_id', 'region_code', 'period']).reset_index(drop=True)
        
        con.register("df_itl1", combined_itl1)
        con.execute("CREATE OR REPLACE TABLE silver.itl1_history AS SELECT * FROM df_itl1")
        log.info(f"  ✓ Updated silver.itl1_history: {len(combined_itl1)} total rows ({len(itl1_rates)} rate observations)")
    
    con.close()


def main():
    log.info("="*70)
    log.info("FILL LAD RATE GAPS")
    log.info("="*70)
    
    # Load data
    log.info("\n[1/5] Loading data...")
    lad_rates, itl3_rates, lad_to_itl3, original_years_by_lad = load_data()
    
    if lad_rates.empty:
        log.warning("No LAD rate data found. Exiting.")
        return
    
    # Fill gaps
    log.info("\n[2/5] Filling gaps...")
    con = duckdb.connect(str(DUCK_PATH))
    filled_rates, stats = fill_all_gaps(lad_rates, itl3_rates, lad_to_itl3, original_years_by_lad, con)
    con.close()
    
    # Update silver layer
    log.info("\n[3/5] Updating silver layer...")
    update_silver_layer(filled_rates)
    
    # Re-aggregate to ITL3/ITL2/ITL1
    log.info("\n[4/5] Re-aggregating filled LAD rates to ITL levels...")
    re_aggregate_to_itl_levels(filled_rates)
    
    # Save summary
    log.info("\n[5/5] Saving summary...")
    summary_path = LOG_DIR / "lad_gap_fill_summary.json"
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj
    
    stats_native = convert_to_native(stats)
    
    with open(summary_path, 'w') as f:
        json.dump(stats_native, f, indent=2)
    log.info(f"Summary saved to {summary_path}")
    
    # Print summary
    log.info("\n" + "="*70)
    log.info("SUMMARY")
    log.info("="*70)
    log.info(f"Total LADs processed: {stats['total_lads']}")
    total_interpolated = stats['total_filled_interpolated'] + stats['total_filled_inherited']
    log.info(f"Values interpolated (gap-filled): {total_interpolated}")
    log.info("="*70)


if __name__ == "__main__":
    main()

