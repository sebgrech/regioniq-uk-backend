#!/usr/bin/env python3
"""
RegionIQ - ITL3 Forecasting Engine V5.3
=======================================

V5.3: SHARE-BASED ALLOCATION + DETERMINISTIC gdhi_per_head

ARCHITECTURE:
  - ADDITIVE metrics: Share allocation from ITL2 (no share drift)
  - RATE metrics: Forecast at ITL3 where history exists, else inherit ITL2
  - gdhi_per_head_gbp: Deterministic ratio (matches Macro/ITL1/ITL2)
  - DERIVED metrics: Monte Carlo for productivity + income_per_worker only
  - gdhi_per_head_gbp stored in BASE table (gold.itl3_forecast)
  - productivity + income_per_worker stored in DERIVED table (gold.itl3_derived)

V5.2 → V5.3 changes:
  - gdhi_per_head computed deterministically (not Monte Carlo) - aligns with upstream
  - Removed gdhi_per_head from derived_metrics config
  - CI for gdhi_per_head: lo_gdhi/hi_pop and hi_gdhi/lo_pop (conservative bounds)

Core principle:
  ITL3_value(t) = ITL2_value(t) × share_ITL3_in_ITL2

Where share is computed as 5-year historical average (stable).

Benefits:
  1. ITL3 inherits ITL2's dampened growth rates automatically
  2. No runaway trajectories (TLI41 can't grow faster than TLI4)
  3. Perfect reconciliation by construction (sum of shares = 1)
  4. Share drift impossible (shares are FIXED from history)
  5. Simpler, faster, more robust

Metrics handling:
  - ADDITIVE metrics (GVA, GDHI, employment, population): share allocation
  - RATE metrics (employment_rate, unemployment_rate): weighted average from ITL2
  - DERIVED metrics (productivity, gdhi_per_head): calculated from allocated totals

Cascade: UK Macro → ITL1 → ITL2 → ITL3 (this script)

Inputs:
  - silver.itl3_history (DuckDB) or CSV
  - gold.itl2_forecast (ITL2 forecasts - already dampened)
  - data/reference/master_2025_geography_lookup.csv

Outputs:
  - data/forecast/itl3_forecast_long.csv
  - data/forecast/itl3_forecast_wide.csv
  - data/forecast/itl3_derived.csv
  - data/forecast/itl3_confidence_intervals.csv
  - data/forecast/itl3_shares.csv
  - data/forecast/itl3_metadata.json
  - gold.itl3_forecast (allocated + rates + gdhi_per_head)
  - gold.itl3_derived (productivity, income_per_worker only)

Author: RegionIQ
Version: 5.3
"""

import json
import logging
import math
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
np.random.seed(42)
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# Optional dependencies
try:
    import duckdb
    HAVE_DUCKDB = True
except ImportError:
    HAVE_DUCKDB = False
    raise ImportError("DuckDB required: pip install duckdb")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.stattools import adfuller
    HAVE_STATSMODELS = True
except ImportError:
    HAVE_STATSMODELS = False
    logging.warning("statsmodels not available - rate forecasting will use fallback")

# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ForecastConfigV5:
    """ITL3 V5 Configuration - Share-based allocation"""
    
    # Paths
    duck_path: Path = field(default_factory=lambda: Path("data/lake/warehouse.duckdb"))
    output_dir: Path = field(default_factory=lambda: Path("data/forecast"))
    lookup_path: Path = field(default_factory=lambda: Path("data/reference/master_2025_geography_lookup.csv"))
    
    # Forecast horizon (inherited from ITL2)
    forecast_start: int = 2024
    forecast_end: int = 2050
    
    # Share computation parameters
    share_lookback_years: int = 5  # Years to average for stable shares
    min_share_history: int = 3     # Minimum years needed to compute share
    
    # Monte Carlo for derived metrics
    monte_carlo_samples: int = 5000
    
    # Additive metrics (allocated via shares)
    additive_metrics: List[str] = field(default_factory=lambda: [
        'nominal_gva_mn_gbp',
        'gdhi_total_mn_gbp',
        'emp_total_jobs',
        'emp_total_jobs_ni',
        'population_total',
        'population_16_64'
    ])
    
    # Rate metrics (use ITL2 values directly - not additive)
    rate_metrics: List[str] = field(default_factory=lambda: [
        'employment_rate_pct',
        'unemployment_rate_pct'
    ])
    
    # Rate forecasting parameters
    rate_min_history_years: int = 8  # Minimum years to forecast rates at ITL3
    rate_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'employment_rate_pct': (40.0, 95.0),
        'unemployment_rate_pct': (1.0, 20.0)
    })
    
    # Derived metrics (calculated from allocated totals) - Monte Carlo only
    # Note: gdhi_per_head_gbp computed deterministically, not here
    derived_metrics: Dict[str, Dict] = field(default_factory=lambda: {
        'productivity_gbp_per_job': {
            'numerator': 'nominal_gva_mn_gbp',
            'denominator': 'emp_total_jobs',
            'multiplier': 1e6,  # Convert millions to pounds
            'unit': 'GBP'
        },
        'income_per_worker_gbp': {
            'numerator': 'gdhi_total_mn_gbp',
            'denominator': 'emp_total_jobs',
            'multiplier': 1e6,
            'unit': 'GBP'
        }
    })
    
    # Metric metadata
    metric_units: Dict[str, str] = field(default_factory=lambda: {
        'nominal_gva_mn_gbp': 'GBP_m',
        'gdhi_total_mn_gbp': 'GBP_m',
        'emp_total_jobs': 'jobs',
        'emp_total_jobs_ni': 'jobs',
        'population_total': 'persons',
        'population_16_64': 'persons',
        'employment_rate_pct': 'percent',
        'unemployment_rate_pct': 'percent',
        'productivity_gbp_per_job': 'GBP',
        'gdhi_per_head_gbp': 'GBP',
        'income_per_worker_gbp': 'GBP'
    })


# =============================================================================
# Data Loaders
# =============================================================================

class DataLoaderV5:
    """Load all required data for ITL3 share allocation"""
    
    def __init__(self, config: ForecastConfigV5):
        self.config = config
        self.con = None
    
    def connect(self):
        """Open DuckDB connection"""
        if not self.config.duck_path.exists():
            raise FileNotFoundError(f"DuckDB not found: {self.config.duck_path}")
        self.con = duckdb.connect(str(self.config.duck_path), read_only=True)
    
    def close(self):
        """Close DuckDB connection"""
        if self.con:
            self.con.close()
    
    def load_itl3_history(self) -> pd.DataFrame:
        """Load ITL3 historical data"""
        log.info("Loading ITL3 history...")
        
        df = self.con.execute("""
            SELECT 
                region_code,
                region_name,
                metric_id as metric,
                period as year,
                value
            FROM silver.itl3_history
            WHERE value IS NOT NULL AND value > 0
        """).fetchdf()
        
        df['year'] = df['year'].astype(int)
        df['value'] = df['value'].astype(float)
        
        log.info(f"  ✓ {len(df):,} rows, {df['region_code'].nunique()} regions, "
                 f"{df['metric'].nunique()} metrics")
        
        return df
    
    def load_itl2_forecasts(self) -> pd.DataFrame:
        """Load ITL2 forecasts (already dampened)"""
        log.info("Loading ITL2 forecasts...")
        
        df = self.con.execute("""
            SELECT 
                region_code as itl2_code,
                region_name as itl2_name,
                metric_id as metric,
                period as year,
                value,
                ci_lower,
                ci_upper,
                data_type
            FROM gold.itl2_forecast
        """).fetchdf()
        
        df['year'] = df['year'].astype(int)
        
        n_forecast = (df['data_type'] == 'forecast').sum()
        log.info(f"  ✓ {len(df):,} rows ({n_forecast:,} forecast), "
                 f"{df['itl2_code'].nunique()} ITL2 regions")
        
        return df
    
    def load_lookup(self) -> pd.DataFrame:
        """Load ITL3→ITL2 mapping"""
        log.info("Loading geography lookup...")
        
        if not self.config.lookup_path.exists():
            raise FileNotFoundError(f"Lookup not found: {self.config.lookup_path}")
        
        lookup = pd.read_csv(self.config.lookup_path)
        lookup.columns = [c.replace('\ufeff', '') for c in lookup.columns]
        
        # Extract ITL3 → ITL2 mapping
        mapping = lookup[['ITL325CD', 'ITL225CD', 'ITL325NM']].drop_duplicates()
        mapping.columns = ['itl3_code', 'itl2_code', 'itl3_name']
        
        log.info(f"  ✓ {len(mapping)} ITL3 → {mapping['itl2_code'].nunique()} ITL2")
        
        return mapping


# =============================================================================
# Share Calculator
# =============================================================================

class ShareCalculator:
    """
    Compute stable historical shares of ITL3 within ITL2 parents.
    
    Key design:
    - Use trailing N-year average (not single year) for robustness
    - Handle missing years gracefully
    - Normalize shares to sum to 1.0 within each ITL2 parent
    """
    
    def __init__(self, config: ForecastConfigV5):
        self.config = config
    
    def compute_shares(
        self,
        itl3_history: pd.DataFrame,
        lookup: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute ITL3 shares of ITL2 parents for each additive metric.
        
        Returns DataFrame with columns:
          itl3_code, itl2_code, itl3_name, metric, share, share_years, share_std
        """
        log.info("Computing ITL3 shares of ITL2 parents...")
        
        # Merge ITL2 parent codes
        df = itl3_history.merge(
            lookup[['itl3_code', 'itl2_code', 'itl3_name']],
            left_on='region_code',
            right_on='itl3_code',
            how='left'
        )
        
        # Drop regions without ITL2 parent
        orphans = df[df['itl2_code'].isna()]['region_code'].unique()
        if len(orphans) > 0:
            log.warning(f"  ⚠️ {len(orphans)} ITL3 codes without ITL2 parent (dropped)")
        df = df.dropna(subset=['itl2_code'])
        
        shares_list = []
        
        for metric in self.config.additive_metrics:
            metric_df = df[df['metric'] == metric].copy()
            if metric_df.empty:
                log.warning(f"  ⚠️ No data for {metric}")
                continue

            # IMPORTANT (dynamic):
            # Compute share window per-metric, anchored to that metric's own tail.
            # This prevents shorter NI-only series (emp_total_jobs_ni) being penalized
            # by later-ending metrics (e.g. rates) elsewhere in the combined table.
            metric_max_year = int(metric_df['year'].max())
            lookback_start = metric_max_year - self.config.share_lookback_years + 1

            metric_data = metric_df[
                (metric_df['year'] >= lookback_start) &
                (metric_df['year'] <= metric_max_year)
            ].copy()
            
            if metric_data.empty:
                log.warning(f"  ⚠️ No data for {metric} in share window {lookback_start}-{metric_max_year}")
                continue
            
            # Compute ITL2 totals per year
            itl2_totals = metric_data.groupby(['itl2_code', 'year'])['value'].sum().reset_index()
            itl2_totals.columns = ['itl2_code', 'year', 'itl2_total']
            
            # Merge back
            metric_data = metric_data.merge(itl2_totals, on=['itl2_code', 'year'])
            
            # Compute share per year
            metric_data['share'] = metric_data['value'] / metric_data['itl2_total']
            
            # Average share across years
            share_stats = metric_data.groupby(['itl3_code', 'itl2_code', 'itl3_name']).agg({
                'share': ['mean', 'std', 'count'],
                'region_name': 'first'
            }).reset_index()
            
            share_stats.columns = ['itl3_code', 'itl2_code', 'itl3_name', 
                                   'share', 'share_std', 'share_years', 'region_name']
            
            share_stats['metric'] = metric
            
            # Filter by minimum history
            # NI-only jobs series is short historically; allow shares with fewer years.
            min_share_years = 2 if metric == "emp_total_jobs_ni" else self.config.min_share_history
            valid = share_stats['share_years'] >= min_share_years
            if (~valid).sum() > 0:
                log.info(f"  {metric}: {(~valid).sum()} regions with insufficient history")
            share_stats = share_stats[valid]
            
            shares_list.append(share_stats)
        
        if not shares_list:
            raise ValueError("No shares computed - check data availability")
        
        shares_df = pd.concat(shares_list, ignore_index=True)
        
        # Normalize shares within each ITL2 parent (ensure sum = 1.0)
        shares_df = self._normalize_shares(shares_df)
        
        # Summary
        log.info(f"  ✓ Computed shares for {shares_df['itl3_code'].nunique()} ITL3 regions")
        log.info(f"  ✓ Metrics: {shares_df['metric'].unique().tolist()}")
        
        return shares_df
    
    def _normalize_shares(self, shares_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize shares to sum to 1.0 within each (ITL2, metric) group"""
        
        # Compute sum per ITL2/metric
        totals = shares_df.groupby(['itl2_code', 'metric'])['share'].sum().reset_index()
        totals.columns = ['itl2_code', 'metric', 'share_sum']
        
        # Merge and normalize
        shares_df = shares_df.merge(totals, on=['itl2_code', 'metric'])
        shares_df['share'] = shares_df['share'] / shares_df['share_sum']
        shares_df = shares_df.drop(columns=['share_sum'])
        
        # Verify normalization
        check = shares_df.groupby(['itl2_code', 'metric'])['share'].sum()
        max_deviation = abs(check - 1.0).max()
        
        if max_deviation > 1e-10:
            log.warning(f"  ⚠️ Share normalization error: max deviation = {max_deviation:.2e}")
        else:
            log.info(f"  ✓ Shares normalized (max deviation = {max_deviation:.2e})")
        
        return shares_df


# =============================================================================
# Share Allocator
# =============================================================================

class ShareAllocator:
    """
    Allocate ITL2 forecasts to ITL3 using fixed historical shares.
    
    This is the core of V5 - no independent ITL3 forecasting.
    ITL3 inherits ITL2's growth rates by construction.
    """
    
    def __init__(self, config: ForecastConfigV5):
        self.config = config
    
    def allocate(
        self,
        itl2_forecasts: pd.DataFrame,
        shares: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Allocate ITL2 forecast values to ITL3 children using shares.
        
        ITL3_value = ITL2_value × share
        ITL3_ci_lower = ITL2_ci_lower × share
        ITL3_ci_upper = ITL2_ci_upper × share
        """
        log.info("Allocating ITL2 forecasts to ITL3...")
        
        results = []
        
        for metric in self.config.additive_metrics:
            # Get ITL2 forecasts for this metric
            itl2_metric = itl2_forecasts[itl2_forecasts['metric'] == metric].copy()
            
            if itl2_metric.empty:
                log.warning(f"  ⚠️ No ITL2 data for {metric}")
                continue
            
            # Get shares for this metric
            metric_shares = shares[shares['metric'] == metric].copy()
            
            if metric_shares.empty:
                log.warning(f"  ⚠️ No shares for {metric}")
                continue
            
            # Select only needed columns
            share_cols = ['itl3_code', 'itl2_code', 'itl3_name', 'region_name', 'share']
            metric_shares = metric_shares[[c for c in share_cols if c in metric_shares.columns]]
            
            itl2_cols = ['itl2_code', 'year', 'value', 'ci_lower', 'ci_upper', 'data_type']
            itl2_subset = itl2_metric[[c for c in itl2_cols if c in itl2_metric.columns]]
            
            # Merge: each ITL3 gets its share of ITL2
            allocated = metric_shares.merge(
                itl2_subset,
                on='itl2_code',
                how='inner'
            )
            
            # Apply share allocation
            allocated['value'] = allocated['value'] * allocated['share']
            if 'ci_lower' in allocated.columns:
                allocated['ci_lower'] = allocated['ci_lower'] * allocated['share']
            if 'ci_upper' in allocated.columns:
                allocated['ci_upper'] = allocated['ci_upper'] * allocated['share']
            
            allocated['metric'] = metric
            
            # Rename columns
            allocated = allocated.rename(columns={
                'itl3_code': 'region_code',
                'itl3_name': 'region_name'
            })
            
            results.append(allocated)
            
            log.info(f"  {metric}: {len(allocated):,} rows allocated")
        
        if not results:
            raise ValueError("No allocations performed - check data")
        
        result_df = pd.concat(results, ignore_index=True)
        
        # Validation: check reconciliation
        self._validate_reconciliation(result_df, itl2_forecasts)
        
        return result_df
    
    def _validate_reconciliation(
        self,
        itl3_allocated: pd.DataFrame,
        itl2_forecasts: pd.DataFrame
    ):
        """Verify that sum(ITL3) = ITL2 for each parent/metric/year"""
        
        log.info("\nValidating reconciliation...")
        
        for metric in self.config.additive_metrics:
            itl3_metric = itl3_allocated[itl3_allocated['metric'] == metric]
            itl2_metric = itl2_forecasts[itl2_forecasts['metric'] == metric]
            
            if itl3_metric.empty or itl2_metric.empty:
                continue
            
            # Sum ITL3 by ITL2 parent and year
            itl3_sums = itl3_metric.groupby(['itl2_code', 'year'])['value'].sum().reset_index()
            itl3_sums.columns = ['itl2_code', 'year', 'itl3_sum']
            
            # Merge with ITL2 values
            check = itl3_sums.merge(
                itl2_metric[['itl2_code', 'year', 'value']].rename(columns={'value': 'itl2_value'}),
                on=['itl2_code', 'year'],
                how='inner'
            )
            
            if check.empty:
                continue
            
            # Compute deviation
            check['deviation'] = abs(check['itl3_sum'] - check['itl2_value']) / check['itl2_value']
            max_deviation = check['deviation'].max()
            
            for year in [2025, 2030, 2040, 2050]:
                year_dev = check[check['year'] == year]['deviation'].max()
                if not pd.isna(year_dev):
                    status = "✓" if year_dev < 1e-10 else "⚠️"
                    log.info(f"  {metric} {year}: max deviation = {year_dev:.6e} {status}")


# =============================================================================
# Rate Metrics Handler (V5.2 - LAD-aligned for 1:1 mappings)
# =============================================================================

class RateMetricsHandler:
    """
    Handle rate metrics (employment_rate, unemployment_rate).
    
    V5.2 Strategy:
    - For 1:1 ITL3-LAD mappings: Use LAD historical rates with mean reversion to ITL2
    - For 1:many mappings: Aggregate LAD rates weighted by population_16_64
    - Otherwise: Forecast where ITL3 has history, inherit from ITL2 parent
    
    This ensures consistency between ITL3 and LAD forecasts for identical geographies.
    """
    
    def __init__(self, config: ForecastConfigV5):
        self.config = config
    
    def apply_rates(
        self,
        itl2_forecasts: pd.DataFrame,
        itl3_history: pd.DataFrame,
        shares: pd.DataFrame,
        loader: Optional[DataLoaderV5] = None  # Pass loader from pipeline
    ) -> pd.DataFrame:
        """
        Apply rate metrics to ITL3:
        - For 1:1 ITL3-LAD pairs: Copy LAD historical rates with mean reversion
        - For 1:many mappings: Aggregate LAD rates weighted by population_16_64
        - Otherwise: Forecast where ITL3 has history, inherit from ITL2 parent
        """
        log.info("Processing rate metrics for ITL3...")
        
        # Load 1:1 ITL3-LAD mapping
        lookup_path = self.config.lookup_path
        if not lookup_path.exists():
            log.warning(f"Lookup file not found: {lookup_path}, skipping LAD alignment")
            return self._apply_rates_fallback(itl2_forecasts, itl3_history, shares)
        
        lookup = pd.read_csv(lookup_path)
        lookup.columns = [c.replace('\ufeff', '') for c in lookup.columns]
        
        # Identify 1:1 ITL3-LAD mappings
        itl3_lad_counts = lookup.groupby('ITL325CD')['LAD25CD'].nunique()
        one_to_one_itl3 = itl3_lad_counts[itl3_lad_counts == 1].index.tolist()
        
        if not one_to_one_itl3:
            log.info("  No 1:1 ITL3-LAD mappings found, using standard forecasting")
            return self._apply_rates_fallback(itl2_forecasts, itl3_history, shares)
        
        # Build 1:1 mapping dict: {ITL3_code: LAD_code}
        one_to_one_map = (
            lookup[lookup['ITL325CD'].isin(one_to_one_itl3)]
            .groupby('ITL325CD')['LAD25CD']
            .first()
            .to_dict()
        )
        
        log.info(f"  Found {len(one_to_one_map)} 1:1 ITL3-LAD mappings")
        
        # Load LAD historical rates from silver layer
        # Use passed loader if available, otherwise create connection
        if loader is not None and loader.con is not None:
            con = loader.con
            close_conn = False
        else:
            con = duckdb.connect(str(self.config.duck_path), read_only=True)
            close_conn = True
        
        try:
            lad_rates_history = self._load_lad_rates_history(con)
        except Exception as e:
            log.warning(f"  Failed to load LAD rates: {e}, using fallback")
            if close_conn:
                con.close()
            return self._apply_rates_fallback(itl2_forecasts, itl3_history, shares)
        
        # Main processing loop - connection must stay open for _aggregate_lad_rates_to_itl3()
        try:
            results = []
            
            # Get unique ITL3→ITL2 mapping from shares
            itl3_to_itl2 = shares[['itl3_code', 'itl2_code', 'itl3_name', 'region_name']].drop_duplicates(subset=['itl3_code'])
            
            for metric in self.config.rate_metrics:
                log.info(f"  Processing {metric}...")
                
                # Get ITL2 forecasts for fallback and mean reversion target
                itl2_metric = itl2_forecasts[itl2_forecasts['metric'] == metric].copy()
                
                if itl2_metric.empty:
                    log.info(f"    No ITL2 data (skipping)")
                    continue
                
                # Get LAD rates for this metric
                lad_metric_history = lad_rates_history[lad_rates_history['metric'] == metric].copy() if not lad_rates_history.empty else pd.DataFrame()
                
                # Get ITL3 history for this metric
                itl3_metric_history = itl3_history[itl3_history['metric'] == metric].copy()
                
                # Identify regions with sufficient ITL3 history (for non-1:1 regions)
                regions_with_history = self._get_regions_with_history(itl3_metric_history)
                
                forecast_count = 0
                inherit_count = 0
                lad_copy_count = 0
                lad_aggregate_count = 0
                
                # Process each ITL3 region
                for _, row in itl3_to_itl2.iterrows():
                    itl3_code = row['itl3_code']
                    itl2_code = row['itl2_code']
                    itl3_name = row.get('itl3_name', row.get('region_name', itl3_code))
                    region_name = row.get('region_name', itl3_name)
                    
                    # Check if 1:1 with LAD
                    if itl3_code in one_to_one_map:
                        lad_code = one_to_one_map[itl3_code]
                        
                        # Get LAD historical rates
                        lad_region_history = lad_metric_history[
                            lad_metric_history['region_code'] == lad_code
                        ] if not lad_metric_history.empty else pd.DataFrame()
                        
                        if not lad_region_history.empty:
                            # Copy LAD historical rates with mean reversion to ITL2
                            result_df = self._copy_lad_rates_to_itl3(
                                lad_region_history, itl3_code, itl3_name, region_name,
                                itl2_code, metric, itl2_metric
                            )
                            
                            if result_df is not None and len(result_df) > 0:
                                # Ensure continuous coverage across forecast horizon
                                result_df = self._fill_missing_forecast_years_from_itl2(
                                    result_df, itl2_code, metric, itl2_metric
                                )
                                results.append(result_df)
                                lad_copy_count += 1
                                continue
                    
                    # Check if 1:many (multiple LADs map to this ITL3)
                    elif itl3_code in lookup['ITL325CD'].values:
                        # Get all LADs for this ITL3
                        lad_codes = lookup[lookup['ITL325CD'] == itl3_code]['LAD25CD'].dropna().unique().tolist()
                        
                        if len(lad_codes) > 1 and not lad_metric_history.empty:
                            # Aggregate LAD rates weighted by population_16_64
                            result_df = self._aggregate_lad_rates_to_itl3(
                                lad_codes, lad_metric_history, itl3_code, itl3_name, region_name,
                                itl2_code, metric, itl2_metric, con
                            )
                            
                            if result_df is not None and len(result_df) > 0:
                                # Ensure continuous coverage across forecast horizon
                                result_df = self._fill_missing_forecast_years_from_itl2(
                                    result_df, itl2_code, metric, itl2_metric
                                )
                                results.append(result_df)
                                lad_aggregate_count += 1
                                continue
                    
                    # Fall back to standard forecasting or ITL2 inheritance
                    if itl3_code in regions_with_history:
                        # Forecast at ITL3 level
                        region_history = itl3_metric_history[
                            itl3_metric_history['region_code'] == itl3_code
                        ].sort_values('year')
                        
                        forecast_df = self._forecast_rate(
                            region_history, itl3_code, itl3_name, region_name,
                            itl2_code, metric
                        )
                        
                        if forecast_df is not None and len(forecast_df) > 0:
                            # Ensure continuous coverage across forecast horizon (covers cases where
                            # ITL3 history ends early and the model only forecasts from a later year)
                            forecast_df = self._fill_missing_forecast_years_from_itl2(
                                forecast_df, itl2_code, metric, itl2_metric
                            )
                            results.append(forecast_df)
                            forecast_count += 1
                            continue
                    
                    # Fall back to ITL2 parent
                    itl2_subset = itl2_metric[itl2_metric['itl2_code'] == itl2_code].copy()
                    
                    if itl2_subset.empty:
                        continue
                    
                    # Inherit ITL2 values across the full forecast horizon, regardless of whether
                    # ITL2 rows are marked historical or forecast. This is critical to avoid gaps
                    # when ITL3 (or LAD) histories have suppressed tail years.
                    # Cover from 2020 onward because ITL3 QA loads period>=2020 and
                    # requires continuous coverage for LAD gating.
                    itl2_horizon = itl2_subset[
                        (itl2_subset['year'] >= 2020)
                        & (itl2_subset['year'] <= self.config.forecast_end)
                    ].copy()
                    
                    if itl2_horizon.empty:
                        continue
                    
                    inherited = pd.DataFrame({
                        'region_code': itl3_code,
                        'region_name': region_name,
                        'itl2_code': itl2_code,
                        'year': itl2_horizon['year'].values,
                        'metric': metric,
                        'value': itl2_horizon['value'].values,
                        'ci_lower': itl2_horizon['ci_lower'].values if 'ci_lower' in itl2_horizon.columns else itl2_horizon['value'].values * 0.95,
                        'ci_upper': itl2_horizon['ci_upper'].values if 'ci_upper' in itl2_horizon.columns else itl2_horizon['value'].values * 1.05,
                        'data_type': 'forecast',
                        'method': 'itl2_inheritance'
                    })
                    
                    results.append(inherited)
                    inherit_count += 1
                
                log.info(f"    {metric}: {lad_copy_count} from LAD (1:1), {lad_aggregate_count} aggregated from LAD, "
                        f"{forecast_count} forecast, {inherit_count} inherited from ITL2")
            
            if not results:
                return pd.DataFrame()
            
            return pd.concat(results, ignore_index=True)
        
        finally:
            # Close connection only after all processing is complete
            if close_conn:
                con.close()

    def _fill_missing_forecast_years_from_itl2(
        self,
        df: pd.DataFrame,
        itl2_code: str,
        metric: str,
        itl2_metric: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Ensure each ITL3 series has continuous coverage for the forecast horizon.
        We fill only missing years using the ITL2 parent series (historical or forecast),
        marking fills as data_type='forecast' and method='itl2_inheritance_fill'.
        """
        if df is None or df.empty:
            return df

        # LAD gating is evaluated on period>=2020 (see ITL3 QA base query).
        # Fill any missing years from 2020 through the forecast end year.
        desired_years = set(range(2020, self.config.forecast_end + 1))
        have_years = set(df['year'].astype(int).tolist()) if 'year' in df.columns else set()
        missing = sorted(desired_years - have_years)
        if not missing:
            return df

        itl2_subset = itl2_metric[
            (itl2_metric['itl2_code'] == itl2_code)
            & (itl2_metric['year'].isin(missing))
        ].copy()
        if itl2_subset.empty:
            return df

        # Build fill rows using ITL2 values (CI fallback if missing)
        fill = pd.DataFrame({
            'region_code': df['region_code'].iloc[0],
            'region_name': df['region_name'].iloc[0] if 'region_name' in df.columns else '',
            'itl2_code': itl2_code,
            'year': itl2_subset['year'].astype(int).values,
            'metric': metric,
            'value': itl2_subset['value'].astype(float).values,
            'ci_lower': itl2_subset['ci_lower'].astype(float).values if 'ci_lower' in itl2_subset.columns else (itl2_subset['value'].astype(float).values * 0.95),
            'ci_upper': itl2_subset['ci_upper'].astype(float).values if 'ci_upper' in itl2_subset.columns else (itl2_subset['value'].astype(float).values * 1.05),
            'data_type': 'forecast',
            'method': 'itl2_inheritance_fill'
        })

        out = pd.concat([df, fill], ignore_index=True)
        out = out.drop_duplicates(subset=['region_code', 'metric', 'year'], keep='first')
        return out
    
    def _load_lad_rates_history(self, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
        """Load LAD historical rate metrics from silver layer"""
        log.info("  Loading LAD historical rates from silver layer...")
        
        rate_metrics = self.config.rate_metrics
        
        # Load from DuckDB silver layer
        query = f"""
            SELECT 
                region_code,
                region_name,
                metric_id as metric,
                period as year,
                value
            FROM silver.lad_history
            WHERE metric_id IN ({','.join([f"'{m}'" for m in rate_metrics])})
            AND value IS NOT NULL AND value > 0
        """
        
        df = con.execute(query).fetchdf()
        
        if df.empty:
            log.warning("    No LAD rate history found in silver layer")
            return pd.DataFrame()
        
        df['year'] = df['year'].astype(int)
        df['value'] = df['value'].astype(float)
        
        log.info(f"    ✓ Loaded {len(df):,} LAD rate records")
        
        return df
    
    def _copy_lad_rates_to_itl3(
        self,
        lad_history: pd.DataFrame,
        itl3_code: str,
        itl3_name: str,
        region_name: str,
        itl2_code: str,
        metric: str,
        itl2_forecasts: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """
        Copy LAD historical rates to ITL3 with mean reversion toward ITL2 parent.
        
        Uses exponential decay toward ITL2 forecast values to avoid flat projection.
        """
        if lad_history.empty:
            return None
        
        # Get forecast years from ITL2
        itl2_metric = itl2_forecasts[itl2_forecasts['itl2_code'] == itl2_code]
        if itl2_metric.empty:
            return None
        
        itl2_forecast_years = itl2_metric[itl2_metric['data_type'] == 'forecast'].copy()
        if itl2_forecast_years.empty:
            return None
        
        forecast_years = itl2_forecast_years['year'].unique()
        itl2_forecast_values = itl2_forecast_years.set_index('year')['value'].to_dict()
        
        # Get last historical value from LAD
        last_year = int(lad_history['year'].max())
        last_value = float(lad_history[lad_history['year'] == last_year]['value'].iloc[0])
        
        # Get ITL2 forecast for first forecast year (mean reversion target)
        first_forecast_year = int(min(forecast_years))
        itl2_target = itl2_forecast_values.get(first_forecast_year, last_value)
        
        # Mean reversion: exponential decay toward ITL2 with half-life of 10 years
        # This prevents flat projection while maintaining LAD-ITL3 consistency
        half_life_years = 10.0
        decay_rate = np.log(2) / half_life_years
        
        forecast_values = []
        for year in sorted(forecast_years):
            years_ahead = year - last_year
            # Exponential decay: start at LAD value, converge to ITL2
            # Use ITL2 forecast for this specific year if available
            itl2_year_value = itl2_forecast_values.get(year, itl2_target)
            
            # Blend: LAD value decays toward ITL2
            weight_lad = np.exp(-decay_rate * years_ahead)
            blended_value = weight_lad * last_value + (1 - weight_lad) * itl2_year_value
            
            forecast_values.append(blended_value)
        
        forecast_values = np.array(forecast_values)
        
        # Estimate CIs from historical variation and ITL2 CIs
        historical_std = lad_history['value'].std()
        if pd.isna(historical_std) or historical_std == 0:
            historical_std = last_value * 0.05
        
        # Get ITL2 CI width if available
        itl2_ci_lower = itl2_forecast_years['ci_lower'].values if 'ci_lower' in itl2_forecast_years.columns else None
        itl2_ci_upper = itl2_forecast_years['ci_upper'].values if 'ci_upper' in itl2_forecast_years.columns else None
        
        if itl2_ci_lower is not None and itl2_ci_upper is not None:
            itl2_ci_width = (itl2_ci_upper - itl2_ci_lower).mean() / 2
            ci_width = np.maximum(historical_std * 1.96, itl2_ci_width)
        else:
            ci_width = historical_std * 1.96
        
        bounds = self.config.rate_bounds.get(metric, (0, 100))
        
        result = pd.DataFrame({
            'region_code': itl3_code,
            'region_name': region_name,
            'itl2_code': itl2_code,
            'year': sorted(forecast_years),
            'metric': metric,
            'value': forecast_values,
            'ci_lower': np.clip(forecast_values - ci_width, bounds[0], bounds[1]),
            'ci_upper': np.clip(forecast_values + ci_width, bounds[0], bounds[1]),
            'data_type': 'forecast',
            'method': 'lad_1to1_mean_revert'
        })
        
        return result
    
    def _aggregate_lad_rates_to_itl3(
        self,
        lad_codes: List[str],
        lad_rates_history: pd.DataFrame,
        itl3_code: str,
        itl3_name: str,
        region_name: str,
        itl2_code: str,
        metric: str,
        itl2_forecasts: pd.DataFrame,
        con: duckdb.DuckDBPyConnection
    ) -> Optional[pd.DataFrame]:
        """
        Aggregate LAD rates to ITL3 using population_16_64 weighted average.
        Also applies mean reversion toward ITL2 parent.
        """
        # Load population weights
        try:
            pop_query = f"""
                SELECT 
                    region_code,
                    period as year,
                    value as population_16_64
                FROM silver.lad_history
                WHERE metric_id = 'population_16_64'
                AND region_code IN ({','.join([f"'{c}'" for c in lad_codes])})
            """
            
            pop_df = con.execute(pop_query).fetchdf()
            pop_df['year'] = pop_df['year'].astype(int)
            
        except Exception as e:
            log.warning(f"    Failed to load population weights: {e}")
            return None
        
        # Get LAD rates for these codes
        lad_metric = lad_rates_history[lad_rates_history['region_code'].isin(lad_codes)].copy()
        
        if lad_metric.empty:
            return None
        
        # Get forecast years from ITL2
        itl2_metric = itl2_forecasts[itl2_forecasts['itl2_code'] == itl2_code]
        if itl2_metric.empty:
            return None
        
        itl2_forecast_years = itl2_metric[itl2_metric['data_type'] == 'forecast'].copy()
        if itl2_forecast_years.empty:
            return None
        
        forecast_years = itl2_forecast_years['year'].unique()
        itl2_forecast_values = itl2_forecast_years.set_index('year')['value'].to_dict()
        
        # For each forecast year, compute weighted average from last historical year
        last_hist_year = int(lad_metric['year'].max())
        last_pop = pop_df[pop_df['year'] == last_hist_year].copy()
        
        if last_pop.empty:
            return None
        
        # Get last historical rates
        last_rates = lad_metric[lad_metric['year'] == last_hist_year].copy()
        
        # Merge rates with population weights
        merged = last_rates.merge(last_pop, on='region_code', how='inner')
        
        if merged.empty:
            return None
        
        # Weighted average
        total_pop = merged['population_16_64'].sum()
        if total_pop == 0:
            return None
        
        weighted_rate = (merged['value'] * merged['population_16_64']).sum() / total_pop
        
        # Mean reversion toward ITL2 (same as 1:1 case)
        last_year = last_hist_year
        half_life_years = 10.0
        decay_rate = np.log(2) / half_life_years
        
        forecast_values = []
        for year in sorted(forecast_years):
            years_ahead = year - last_year
            itl2_year_value = itl2_forecast_values.get(year, weighted_rate)
            
            weight_lad = np.exp(-decay_rate * years_ahead)
            blended_value = weight_lad * weighted_rate + (1 - weight_lad) * itl2_year_value
            
            forecast_values.append(blended_value)
        
        forecast_values = np.array(forecast_values)
        
        # Estimate CIs from component variation
        rate_std = merged['value'].std() if len(merged) > 1 else weighted_rate * 0.05
        
        # Blend CI width with ITL2 CIs
        itl2_ci_lower = itl2_forecast_years['ci_lower'].values if 'ci_lower' in itl2_forecast_years.columns else None
        itl2_ci_upper = itl2_forecast_years['ci_upper'].values if 'ci_upper' in itl2_forecast_years.columns else None
        
        if itl2_ci_lower is not None and itl2_ci_upper is not None:
            itl2_ci_width = (itl2_ci_upper - itl2_ci_lower).mean() / 2
            ci_width = np.maximum(rate_std * 1.96, itl2_ci_width)
        else:
            ci_width = rate_std * 1.96
        
        bounds = self.config.rate_bounds.get(metric, (0, 100))
        
        result = pd.DataFrame({
            'region_code': itl3_code,
            'region_name': region_name,
            'itl2_code': itl2_code,
            'year': sorted(forecast_years),
            'metric': metric,
            'value': forecast_values,
            'ci_lower': np.clip(forecast_values - ci_width, bounds[0], bounds[1]),
            'ci_upper': np.clip(forecast_values + ci_width, bounds[0], bounds[1]),
            'data_type': 'forecast',
            'method': 'lad_weighted_aggregate_mean_revert'
        })
        
        return result
    
    def _apply_rates_fallback(
        self,
        itl2_forecasts: pd.DataFrame,
        itl3_history: pd.DataFrame,
        shares: pd.DataFrame
    ) -> pd.DataFrame:
        """Fallback to original V5.1 logic when LAD data unavailable"""
        log.info("  Using fallback rate forecasting (no LAD alignment)")
        
        results = []
        itl3_to_itl2 = shares[['itl3_code', 'itl2_code', 'itl3_name', 'region_name']].drop_duplicates(subset=['itl3_code'])
        
        for metric in self.config.rate_metrics:
            itl2_metric = itl2_forecasts[itl2_forecasts['metric'] == metric].copy()
            if itl2_metric.empty:
                continue
            
            itl3_metric_history = itl3_history[itl3_history['metric'] == metric].copy()
            regions_with_history = self._get_regions_with_history(itl3_metric_history)
            
            for _, row in itl3_to_itl2.iterrows():
                itl3_code = row['itl3_code']
                itl2_code = row['itl2_code']
                itl3_name = row.get('itl3_name', row.get('region_name', itl3_code))
                region_name = row.get('region_name', itl3_name)
                
                if itl3_code in regions_with_history:
                    region_history = itl3_metric_history[
                        itl3_metric_history['region_code'] == itl3_code
                    ].sort_values('year')
                    
                    forecast_df = self._forecast_rate(
                        region_history, itl3_code, itl3_name, region_name,
                        itl2_code, metric
                    )
                    
                    if forecast_df is not None and len(forecast_df) > 0:
                        results.append(forecast_df)
                        continue
                
                itl2_subset = itl2_metric[itl2_metric['itl2_code'] == itl2_code].copy()
                if itl2_subset.empty:
                    continue
                
                itl2_forecast = itl2_subset[itl2_subset['data_type'] == 'forecast'].copy()
                if itl2_forecast.empty:
                    continue
                
                inherited = pd.DataFrame({
                    'region_code': itl3_code,
                    'region_name': region_name,
                    'itl2_code': itl2_code,
                    'year': itl2_forecast['year'].values,
                    'metric': metric,
                    'value': itl2_forecast['value'].values,
                    'ci_lower': itl2_forecast['ci_lower'].values if 'ci_lower' in itl2_forecast.columns else itl2_forecast['value'].values * 0.95,
                    'ci_upper': itl2_forecast['ci_upper'].values if 'ci_upper' in itl2_forecast.columns else itl2_forecast['value'].values * 1.05,
                    'data_type': 'forecast',
                    'method': 'itl2_inheritance'
                })
                
                results.append(inherited)
        
        if not results:
            return pd.DataFrame()
        
        return pd.concat(results, ignore_index=True)
    
    def _get_regions_with_history(self, metric_history: pd.DataFrame) -> set:
        """Identify ITL3 regions with sufficient rate history"""
        if metric_history.empty:
            return set()
        
        region_counts = metric_history.groupby('region_code')['year'].nunique()
        sufficient = region_counts[region_counts >= self.config.rate_min_history_years]
        
        return set(sufficient.index)
    
    def _forecast_rate_mean_revert(
        self,
        history: pd.DataFrame,
        forecast_years: List[int],
        metric: str,
        bounds: Tuple[float, float],
        fallback_mean: Optional[float] = None
    ) -> Optional[Dict]:
        """
        Mean-reverting rate forecast using regional long-term average.
        
        Uses Ornstein-Uhlenbeck exponential decay: V(t) = μ + (V₀ - μ) * e^(-θt)
        where θ = 0.10 (10% reversion per year).
        
        Args:
            history: DataFrame with 'year' and 'value' columns
            forecast_years: List of years to forecast
            metric: Metric name (for logging)
            bounds: (min, max) tuple for clipping
            fallback_mean: Fallback mean if insufficient history (< 5 years)
        
        Returns dict with keys: 'values', 'ci_lower', 'ci_upper', 'method'
        """
        if history.empty:
            return None
        
        # Minimum history check: require at least 5 years for reliable mean
        if len(history) < 5:
            if fallback_mean is not None:
                regional_mean = fallback_mean
            else:
                # Use last value as fallback
                regional_mean = float(history['value'].iloc[-1])
        else:
            regional_mean = float(history['value'].mean())
        
        # Get last observed value and year
        last_year = int(history['year'].max())
        last_value = float(history[history['year'] == last_year]['value'].iloc[0])
        
        # Apply exponential decay with theta = 0.10
        theta = 0.10
        forecast_values = []
        for year in sorted(forecast_years):
            years_ahead = year - last_year
            forecast_value = regional_mean + (last_value - regional_mean) * np.exp(-theta * years_ahead)
            forecast_values.append(forecast_value)
        
        forecast_values = np.array(forecast_values)
        
        # CI calculation: Steady-state variance for O-U process
        historical_std = float(history['value'].std())
        if historical_std <= 0:
            historical_std = abs(regional_mean * 0.05)  # Fallback: 5% of mean
        
        steady_state_std = historical_std / np.sqrt(2 * theta)
        ci_width = 1.96 * steady_state_std  # Constant width for all years
        ci_lower = forecast_values - ci_width
        ci_upper = forecast_values + ci_width
        
        # Apply bounds clipping
        forecast_values = np.clip(forecast_values, bounds[0], bounds[1])
        ci_lower = np.clip(ci_lower, bounds[0], bounds[1])
        ci_upper = np.clip(ci_upper, bounds[0], bounds[1])
        
        return {
            'values': forecast_values,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'method': 'mean_revert_regional'
        }
    
    def _forecast_rate(
        self,
        history: pd.DataFrame,
        itl3_code: str,
        itl3_name: str,
        region_name: str,
        itl2_code: str,
        metric: str
    ) -> Optional[pd.DataFrame]:
        """Forecast a single rate metric using mean reversion to regional long-term average"""
        
        if len(history) < self.config.rate_min_history_years:
            return None
        
        # Determine forecast horizon
        last_year = int(history['year'].max())
        horizon = self.config.forecast_end - last_year
        
        if horizon <= 0:
            return None
        
        forecast_years = list(range(last_year + 1, self.config.forecast_end + 1))
        
        # Get bounds for this metric
        bounds = self.config.rate_bounds.get(metric, (0, 100))
        
        # Get ITL2 parent mean for fallback (from itl2_forecasts passed to apply_rates)
        # Note: fallback_mean will be None here, but can be passed from apply_rates if needed
        fallback_mean = None
        
        # Use mean reversion instead of ensemble
        forecast_result = self._forecast_rate_mean_revert(history, forecast_years, metric, bounds, fallback_mean)
        
        if forecast_result is None:
            return None
        
        # Build output DataFrame
        result = pd.DataFrame({
            'region_code': itl3_code,
            'region_name': region_name,
            'itl2_code': itl2_code,
            'year': forecast_years,
            'metric': metric,
            'value': forecast_result['values'],
            'ci_lower': forecast_result['ci_lower'],
            'ci_upper': forecast_result['ci_upper'],
            'data_type': 'forecast',
            'method': forecast_result['method']
        })
        
        return result
    
    def _fit_ensemble(
        self,
        series: pd.Series,
        horizon: int,
        bounds: Tuple[float, float]
    ) -> Optional[Dict]:
        """Fit ARIMA/ETS ensemble for rate forecasting"""
        
        models = []
        
        # Try ARIMA
        if HAVE_STATSMODELS:
            arima_result = self._fit_arima(series, horizon)
            if arima_result:
                models.append(arima_result)
            
            # Try ETS
            ets_result = self._fit_ets(series, horizon)
            if ets_result:
                models.append(ets_result)
        
        # Linear trend fallback
        linear_result = self._fit_linear(series, horizon)
        if linear_result:
            models.append(linear_result)
        
        if not models:
            return None
        
        # Combine using AIC weights (or equal weights if AIC unavailable)
        if len(models) > 1:
            combined = self._combine_forecasts(models)
        else:
            combined = models[0]
        
        # Apply bounds
        combined['values'] = np.clip(combined['values'], bounds[0], bounds[1])
        combined['ci_lower'] = np.clip(combined['ci_lower'], bounds[0], bounds[1])
        combined['ci_upper'] = np.clip(combined['ci_upper'], bounds[0], bounds[1])
        
        return combined
    
    def _fit_arima(self, series: pd.Series, horizon: int) -> Optional[Dict]:
        """Fit ARIMA model"""
        if not HAVE_STATSMODELS:
            return None
        
        try:
            # Test stationarity
            adf_p = adfuller(series, maxlag=5)[1]
            d = 0 if adf_p < 0.05 else 1
            
            best_aic = np.inf
            best_model = None
            best_order = None
            
            for p in [0, 1, 2]:
                for q in [0, 1]:
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_model = fitted
                            best_order = (p, d, q)
                    except:
                        continue
            
            if best_model is None:
                return None
            
            forecast_obj = best_model.get_forecast(steps=horizon)
            fc = forecast_obj.predicted_mean.values
            ci = forecast_obj.conf_int(alpha=0.05)
            
            return {
                'method': f'ARIMA{best_order}',
                'values': fc,
                'ci_lower': ci.iloc[:, 0].values,
                'ci_upper': ci.iloc[:, 1].values,
                'aic': best_aic
            }
        except Exception:
            return None
    
    def _fit_ets(self, series: pd.Series, horizon: int) -> Optional[Dict]:
        """Fit Exponential Smoothing model"""
        if not HAVE_STATSMODELS:
            return None
        
        try:
            model = ExponentialSmoothing(
                series,
                trend='add',
                damped_trend=True,
                seasonal=None
            )
            fitted = model.fit()
            fc = fitted.forecast(steps=horizon)
            
            # Estimate CIs from residuals
            residuals = series - fitted.fittedvalues
            sigma = residuals.std()
            ci_width = 1.96 * sigma * np.sqrt(np.arange(1, horizon + 1))
            
            return {
                'method': 'ETS(A,Ad,N)',
                'values': fc.values,
                'ci_lower': fc.values - ci_width,
                'ci_upper': fc.values + ci_width,
                'aic': fitted.aic if hasattr(fitted, 'aic') else np.inf
            }
        except Exception:
            return None
    
    def _fit_linear(self, series: pd.Series, horizon: int) -> Optional[Dict]:
        """Fit linear trend as fallback"""
        try:
            x = np.arange(len(series))
            y = series.values
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            x_future = np.arange(len(series), len(series) + horizon)
            fc = intercept + slope * x_future
            
            # CIs based on residuals
            residuals = y - (intercept + slope * x)
            sigma = residuals.std()
            ci_width = 1.96 * sigma * np.sqrt(1 + 1/len(series) + (x_future - x.mean())**2 / np.sum((x - x.mean())**2))
            
            # Penalize linear with high AIC
            aic = len(series) * np.log(sigma**2) + 4  # Simple AIC approximation
            
            return {
                'method': 'Linear',
                'values': fc,
                'ci_lower': fc - ci_width,
                'ci_upper': fc + ci_width,
                'aic': aic
            }
        except Exception:
            return None
    
    def _combine_forecasts(self, models: List[Dict]) -> Dict:
        """Combine forecasts using AIC weights"""
        aics = np.array([m.get('aic', np.inf) for m in models])
        
        # Handle infinite AICs
        finite_mask = np.isfinite(aics)
        if not finite_mask.any():
            weights = np.ones(len(models)) / len(models)
        else:
            delta_aic = aics - np.nanmin(aics)
            weights = np.exp(-0.5 * delta_aic)
            weights = np.where(np.isfinite(weights), weights, 0)
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = np.ones(len(models)) / len(models)
        
        # Weighted average
        n_steps = len(models[0]['values'])
        combined_values = np.zeros(n_steps)
        combined_ci_lower = np.zeros(n_steps)
        combined_ci_upper = np.zeros(n_steps)
        
        for model, w in zip(models, weights):
            combined_values += w * model['values']
            combined_ci_lower += w * model['ci_lower']
            combined_ci_upper += w * model['ci_upper']
        
        methods_used = [m['method'] for m in models]
        
        return {
            'method': f'Ensemble({",".join(methods_used)})',
            'values': combined_values,
            'ci_lower': combined_ci_lower,
            'ci_upper': combined_ci_upper,
            'aic': np.average(aics[np.isfinite(aics)]) if np.isfinite(aics).any() else np.inf
        }


# =============================================================================
# Derived Metrics Calculator
# =============================================================================

class DerivedMetricsCalculator:
    """
    Calculate derived metrics from allocated totals using Monte Carlo.
    
    Derived metrics (Monte Carlo):
    - productivity_gbp_per_job = GVA / Employment × 1e6
    - income_per_worker_gbp = GDHI / Employment × 1e6
    
    Deterministic (base table):
    - gdhi_per_head_gbp = GDHI / Population × 1e6
    """
    
    def __init__(self, config: ForecastConfigV5):
        self.config = config
        self.n_samples = config.monte_carlo_samples
    
    def calculate(self, allocated_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all derived metrics with uncertainty propagation"""
        
        log.info("Calculating derived metrics (Monte Carlo)...")
        
        # Get years that need derived metrics (any year with at least one forecast)
        forecast_years = allocated_df[allocated_df['data_type'] == 'forecast']['year'].unique()
        
        # Use ALL data for those years (handles mixed-vintage where components have different data_types)
        forecast_data = allocated_df[allocated_df['year'].isin(forecast_years)].copy()
        
        if forecast_data.empty:
            log.warning("  No data for derived metrics")
            return pd.DataFrame()
        
        # Simplify: just use region_code, itl2_code, year as index
        pivot = forecast_data.pivot_table(
            index=['region_code', 'itl2_code', 'year'],
            columns='metric',
            values=['value', 'ci_lower', 'ci_upper'],
            aggfunc='first'
        )
        
        # Flatten column names
        pivot.columns = ['_'.join(col).strip() for col in pivot.columns]
        pivot = pivot.reset_index()
        
        # Get region_name mapping
        name_map = forecast_data.drop_duplicates('region_code').set_index('region_code')['region_name'].to_dict()
        
        derived_rows = []
        
        for derived_name, config in self.config.derived_metrics.items():
            num_col = f"value_{config['numerator']}"
            den_col = f"value_{config['denominator']}"
            
            if num_col not in pivot.columns or den_col not in pivot.columns:
                log.info(f"  {derived_name}: Missing inputs (skipping)")
                continue
            
            log.info(f"  Computing {derived_name}...")
            
            num_ci_low = f"ci_lower_{config['numerator']}"
            num_ci_high = f"ci_upper_{config['numerator']}"
            den_ci_low = f"ci_lower_{config['denominator']}"
            den_ci_high = f"ci_upper_{config['denominator']}"
            
            for _, row in pivot.iterrows():
                num_val = row.get(num_col)
                den_val = row.get(den_col)
                
                if pd.isna(num_val) or pd.isna(den_val) or den_val <= 0:
                    continue
                
                # Get CIs (with defaults if missing)
                num_low = row.get(num_ci_low, num_val * 0.9)
                num_high = row.get(num_ci_high, num_val * 1.1)
                den_low = row.get(den_ci_low, den_val * 0.95)
                den_high = row.get(den_ci_high, den_val * 1.05)
                
                # Monte Carlo
                result = self._monte_carlo_ratio(
                    num_val * config['multiplier'],
                    num_low * config['multiplier'],
                    num_high * config['multiplier'],
                    den_val,
                    den_low,
                    den_high
                )
                
                derived_rows.append({
                    'region_code': row['region_code'],
                    'region_name': name_map.get(row['region_code'], ''),
                    'itl2_code': row['itl2_code'],
                    'year': row['year'],
                    'metric': derived_name,
                    'value': result['value'],
                    'ci_lower': result['ci_lower'],
                    'ci_upper': result['ci_upper'],
                    'data_type': 'forecast',
                    'share': np.nan
                })
        
        if not derived_rows:
            return pd.DataFrame()
        
        derived_df = pd.DataFrame(derived_rows)
        log.info(f"  ✓ Computed {len(derived_df):,} derived metric values")
        
        return derived_df
    
    def _monte_carlo_ratio(
        self,
        num_val: float, num_low: float, num_high: float,
        den_val: float, den_low: float, den_high: float
    ) -> Dict[str, float]:
        """Monte Carlo simulation for ratio with uncertainty"""
        
        # Estimate standard deviations from CI width
        # Handle NaN CIs (historical components in mixed-vintage scenarios)
        if math.isnan(num_low) or math.isnan(num_high):
            num_sigma = max(num_val * 0.02, 1e-10)
        else:
            num_sigma = max((num_high - num_low) / 3.92, num_val * 0.01)
        
        if math.isnan(den_low) or math.isnan(den_high):
            den_sigma = max(den_val * 0.02, 1e-10)
        else:
            den_sigma = max((den_high - den_low) / 3.92, den_val * 0.01)
        
        # Sample
        num_samples = np.random.normal(num_val, num_sigma, self.n_samples)
        den_samples = np.random.normal(den_val, den_sigma, self.n_samples)
        
        # Avoid division by zero
        den_samples = np.maximum(den_samples, 1)
        
        # Compute ratio
        ratio_samples = num_samples / den_samples
        
        # Use mean (not median) for point estimate
        result_val = max(0, float(np.mean(ratio_samples)))
        result_low = max(0, float(np.percentile(ratio_samples, 2.5)))
        result_high = max(0, float(np.percentile(ratio_samples, 97.5)))
        
        # Ensure CI ordering (can invert with skewed distributions)
        if result_low > result_high:
            result_low, result_high = result_high, result_low
        
        # Clamp point estimate within CI bounds
        result_val = max(result_low, min(result_val, result_high))
        
        return {
            'value': result_val,
            'ci_lower': result_low,
            'ci_upper': result_high
        }


# =============================================================================
# Output Builder
# =============================================================================

class OutputBuilder:
    """Build and save all output formats"""
    
    def __init__(self, config: ForecastConfigV5):
        self.config = config
    
    def build_final_output(
        self,
        itl3_history: pd.DataFrame,
        allocated: pd.DataFrame,
        rates: pd.DataFrame,
        gdhi_per_head: pd.DataFrame,
        lookup: pd.DataFrame
    ) -> pd.DataFrame:
        """Combine all components into final output"""
        
        log.info("Building final output...")
        
        # Define standard columns
        std_cols = [
            'region_code', 'region_name', 'itl2_code',
            'metric', 'year', 'value',
            'ci_lower', 'ci_upper',
            'data_type', 'method', 'source'
        ]
        
        def standardize_df(df, data_type, method, source):
            """Standardize a dataframe to common schema"""
            # Remove duplicate columns if any
            df = df.loc[:, ~df.columns.duplicated()].copy()
            
            df['data_type'] = data_type
            df['method'] = method
            df['source'] = source
            
            # Build new dataframe with only standard columns
            out = pd.DataFrame()
            for col in std_cols:
                if col in df.columns:
                    out[col] = df[col].values
                else:
                    out[col] = np.nan
            
            return out
        
        # Prepare historical data
        hist = itl3_history.copy()
        hist = hist.loc[:, ~hist.columns.duplicated()]
        
        # Filter out derived metrics from history - they belong in gold.itl3_derived only
        derived_metrics = ['productivity_gbp_per_job', 'income_per_worker_gbp']
        if 'metric' in hist.columns:
            hist = hist[~hist['metric'].isin(derived_metrics)]
        
        # Add ITL2 parent codes to history
        lookup_clean = lookup[['itl3_code', 'itl2_code']].drop_duplicates()
        hist = hist.merge(
            lookup_clean,
            left_on='region_code',
            right_on='itl3_code',
            how='left'
        )
        if 'itl3_code' in hist.columns:
            hist = hist.drop(columns=['itl3_code'])
        hist = standardize_df(hist, 'historical', 'observed', 'ONS')
        
        # Prepare allocated forecasts
        alloc = standardize_df(allocated, 'forecast', 'share_allocation_v5', 'RegionIQ')
        
        # Prepare rates
        if rates is not None and len(rates) > 0:
            rates_std = standardize_df(rates, 'forecast', 'itl2_inheritance_v5', 'RegionIQ')
        else:
            rates_std = None
        
        # Prepare gdhi_per_head (deterministic, not Monte Carlo)
        if gdhi_per_head is not None and len(gdhi_per_head) > 0:
            gdhi_std = standardize_df(gdhi_per_head, 'forecast', 'deterministic_ratio', 'RegionIQ')
        else:
            gdhi_std = None
        
        # Combine all - using list comprehension to filter None/empty
        dfs = [df for df in [hist, alloc, rates_std, gdhi_std] if df is not None and len(df) > 0]
        
        # Concat row by row to avoid index issues
        final = pd.concat(dfs, axis=0, ignore_index=True)
        
        # Add standard columns
        final['region_level'] = 'ITL3'
        final['unit'] = final['metric'].map(self.config.metric_units)
        final['freq'] = 'A'
        
        # Final column order
        final_cols = [
            'region_code', 'region_name', 'region_level', 'itl2_code',
            'metric', 'year', 'value', 'unit', 'freq',
            'data_type', 'method', 'source',
            'ci_lower', 'ci_upper'
        ]
        
        for col in final_cols:
            if col not in final.columns:
                final[col] = np.nan
        
        final = final[final_cols].sort_values(['region_code', 'metric', 'year']).reset_index(drop=True)
        
        # Fix any inverted CIs (belt-and-suspenders)
        ci_mask = final['ci_lower'] > final['ci_upper']
        if ci_mask.sum() > 0:
            final.loc[ci_mask, ['ci_lower', 'ci_upper']] = final.loc[ci_mask, ['ci_upper', 'ci_lower']].values
            log.info(f"  ✓ Fixed {ci_mask.sum()} inverted CIs")
        
        # Deduplicate: prefer historical over forecast for overlapping region/metric/year
        before = len(final)
        final['_priority'] = final['data_type'].map({'historical': 0, 'forecast': 1}).fillna(1)
        final = final.sort_values('_priority').drop_duplicates(
            subset=['region_code', 'metric', 'year'],
            keep='first'
        ).drop(columns=['_priority'])
        after = len(final)
        if before != after:
            log.info(f"  ✓ Removed {before - after} duplicate rows (kept historical over forecast)")
        
        log.info(f"  ✓ Final output: {len(final):,} rows")
        log.info(f"    Historical: {(final['data_type'] == 'historical').sum():,}")
        log.info(f"    Forecast:   {(final['data_type'] == 'forecast').sum():,}")
        
        return final
    
    def save_outputs(self, df: pd.DataFrame, shares: pd.DataFrame, derived: pd.DataFrame):
        """Save all output formats"""
        
        log.info("\nSaving outputs...")
        
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Note: gdhi_per_head is already in df (computed deterministically)
        # derived contains only productivity + income_per_worker (Monte Carlo)
        
        # 1. Long format CSV (base)
        long_path = self.config.output_dir / "itl3_forecast_long.csv"
        df.to_csv(long_path, index=False)
        log.info(f"  ✓ {long_path}")
        
        # 2. Wide format CSV (base)
        wide = df.pivot_table(
            index=['region_code', 'region_name', 'metric'],
            columns='year',
            values='value',
            aggfunc='first'
        ).reset_index()
        wide.columns.name = None
        
        wide_path = self.config.output_dir / "itl3_forecast_wide.csv"
        wide.to_csv(wide_path, index=False)
        log.info(f"  ✓ {wide_path}")
        
        # 3. Derived metrics CSV (productivity, income_per_worker only)
        if not derived.empty:
            derived_path = self.config.output_dir / "itl3_derived.csv"
            derived.to_csv(derived_path, index=False)
            log.info(f"  ✓ {derived_path}")
        
        # 4. Confidence intervals
        ci_data = df[
            (df['data_type'] == 'forecast') &
            (df['ci_lower'].notna())
        ][['region_code', 'metric', 'year', 'value', 'ci_lower', 'ci_upper']].copy()
        
        if not ci_data.empty:
            ci_data['ci_width'] = ci_data['ci_upper'] - ci_data['ci_lower']
            ci_path = self.config.output_dir / "itl3_confidence_intervals.csv"
            ci_data.to_csv(ci_path, index=False)
            log.info(f"  ✓ {ci_path}")
        
        # 5. Shares (for transparency/debugging)
        shares_path = self.config.output_dir / "itl3_shares.csv"
        shares.to_csv(shares_path, index=False)
        log.info(f"  ✓ {shares_path}")
        
        # 6. Metadata
        metadata = {
            'run_timestamp': datetime.now(timezone.utc).isoformat(),
            'version': '5.3',
            'level': 'ITL3',
            'method': 'share_allocation_from_itl2',
            'architecture': {
                'description': 'Share-based allocation - ITL3 inherits ITL2 growth rates',
                'share_lookback_years': self.config.share_lookback_years,
                'additive_metrics': self.config.additive_metrics,
                'rate_metrics': self.config.rate_metrics,
                'derived_metrics': list(self.config.derived_metrics.keys()),
                'base_table_metrics': self.config.additive_metrics + self.config.rate_metrics + ['gdhi_per_head_gbp'],
                'derived_table_metrics': ['productivity_gbp_per_job', 'income_per_worker_gbp']
            },
            'data_summary': {
                'regions': int(df['region_code'].nunique()),
                'metrics': int(df['metric'].nunique()),
                'total_rows': len(df),
                'historical_rows': int((df['data_type'] == 'historical').sum()),
                'forecast_rows': int((df['data_type'] == 'forecast').sum())
            }
        }
        
        meta_path = self.config.output_dir / "itl3_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        log.info(f"  ✓ {meta_path}")
        
        # 7. DuckDB - base and derived tables
        self._write_duckdb(df, derived)
    
    def _write_duckdb(self, df: pd.DataFrame, derived_df: pd.DataFrame):
        """Write to DuckDB gold tables - base and derived"""
        
        log.info("  Writing to DuckDB...")
        
        con = duckdb.connect(str(self.config.duck_path))
        con.execute("CREATE SCHEMA IF NOT EXISTS gold")
        
        # Base table (allocated + rates + gdhi_per_head)
        duck_df = df.copy()
        duck_df['metric_id'] = duck_df['metric']
        duck_df['period'] = duck_df['year']
        duck_df['forecast_run_date'] = datetime.now().date()
        duck_df['forecast_version'] = 'v5.3_share_allocation'
        
        con.register('itl3_df', duck_df)
        
        con.execute("""
            CREATE OR REPLACE TABLE gold.itl3_forecast AS
            SELECT 
                region_code,
                region_name,
                region_level,
                itl2_code,
                metric_id,
                period,
                value,
                unit,
                freq,
                data_type,
                method,
                source,
                ci_lower,
                ci_upper,
                forecast_run_date,
                forecast_version
            FROM itl3_df
        """)
        
        # Views for base table
        con.execute("""
            CREATE OR REPLACE VIEW gold.itl3_forecast_only AS
            SELECT * FROM gold.itl3_forecast
            WHERE data_type = 'forecast'
        """)
        
        con.execute("""
            CREATE OR REPLACE VIEW gold.itl3_latest AS
            SELECT * FROM gold.itl3_forecast
        """)
        
        base_count = con.execute("SELECT COUNT(*) FROM gold.itl3_forecast").fetchone()[0]
        log.info(f"  ✓ gold.itl3_forecast: {base_count:,} rows")
        
        # Derived table (productivity, income_per_worker only)
        if not derived_df.empty:
            derived_duck = derived_df.copy()
            derived_duck['metric_id'] = derived_duck['metric']
            derived_duck['period'] = derived_duck['year']
            derived_duck['forecast_run_date'] = datetime.now().date()
            derived_duck['forecast_version'] = 'v5.3_monte_carlo'
            derived_duck['region_level'] = 'ITL3'
            derived_duck['freq'] = 'A'
            derived_duck['source'] = 'RegionIQ'
            derived_duck['method'] = 'derived_monte_carlo'
            derived_duck['unit'] = derived_duck['metric'].map(self.config.metric_units)
            
            con.register('itl3_derived_df', derived_duck)
            
            con.execute("""
                CREATE OR REPLACE TABLE gold.itl3_derived AS
                SELECT 
                    region_code,
                    region_name,
                    region_level,
                    itl2_code,
                    metric_id,
                    period,
                    value,
                    unit,
                    freq,
                    data_type,
                    method,
                    source,
                    ci_lower,
                    ci_upper,
                    forecast_run_date,
                    forecast_version
                FROM itl3_derived_df
            """)
            
            derived_count = con.execute("SELECT COUNT(*) FROM gold.itl3_derived").fetchone()[0]
            log.info(f"  ✓ gold.itl3_derived: {derived_count:,} rows")
        
        con.close()


# =============================================================================
# Pipeline
# =============================================================================

class ITL3PipelineV5:
    """
    ITL3 V5 Pipeline - Share-based allocation
    
    Steps:
      1. Load ITL3 history
      2. Load ITL2 forecasts (already dampened)
      3. Load ITL3→ITL2 lookup
      4. Compute stable historical shares
      5. Allocate ITL2 forecasts to ITL3 using shares
      6. Apply rate metrics from ITL2
      7. Calculate derived metrics
      8. Combine with history and save
    """
    
    def __init__(self, config: ForecastConfigV5):
        self.config = config
        self.loader = DataLoaderV5(config)
        self.share_calc = ShareCalculator(config)
        self.allocator = ShareAllocator(config)
        self.rate_handler = RateMetricsHandler(config)
        self.derived_calc = DerivedMetricsCalculator(config)
        self.output_builder = OutputBuilder(config)
    
    def run(self) -> pd.DataFrame:
        """Execute the full pipeline"""
        
        start_time = datetime.now()
        
        log.info("")
        log.info("=" * 70)
        log.info(" REGIONIQ — ITL3 FORECASTING ENGINE V5.3")
        log.info(" (Share allocation + deterministic gdhi_per_head)")
        log.info("=" * 70)
        log.info("")
        
        # Connect to DuckDB
        self.loader.connect()
        
        try:
            # 1. Load all data
            log.info("[1/8] Loading data...")
            itl3_history = self.loader.load_itl3_history()
            itl2_forecasts = self.loader.load_itl2_forecasts()
            lookup = self.loader.load_lookup()
            
            # 2. Compute shares
            log.info("\n[2/8] Computing ITL3 shares...")
            shares = self.share_calc.compute_shares(itl3_history, lookup)
            
            # 3. Allocate ITL2 → ITL3
            log.info("\n[3/8] Allocating ITL2 forecasts to ITL3...")
            allocated = self.allocator.allocate(itl2_forecasts, shares)
            
            # 4. Apply rate metrics (forecast where ITL3 history exists)
            log.info("\n[4/8] Processing rate metrics...")
            rates = self.rate_handler.apply_rates(itl2_forecasts, itl3_history, shares, loader=self.loader)
            
            # 4b. Compute gdhi_per_head deterministically (not Monte Carlo)
            log.info("\n[4b/8] Computing gdhi_per_head (deterministic)...")
            gdhi_per_head = self._compute_gdhi_per_head(allocated)
            
            # 5. Calculate derived metrics (Monte Carlo for productivity, income_per_worker only)
            log.info("\n[5/8] Calculating derived metrics...")
            # Combine allocated and rates for derived calculation
            # Use a simple approach: rebuild dataframes with clean columns
            
            common_cols = ['region_code', 'region_name', 'itl2_code', 'year', 
                          'metric', 'value', 'ci_lower', 'ci_upper', 'data_type']
            
            def clean_df(df, cols):
                """Extract only specified columns, handling duplicates"""
                df = df.loc[:, ~df.columns.duplicated()].copy()
                out = pd.DataFrame()
                for c in cols:
                    if c in df.columns:
                        out[c] = df[c].values
                    else:
                        out[c] = np.nan
                return out
            
            alloc_clean = clean_df(allocated, common_cols)
            
            if not rates.empty:
                rates_clean = clean_df(rates, common_cols)
                for_derived = pd.concat([alloc_clean, rates_clean], axis=0, ignore_index=True)
            else:
                for_derived = alloc_clean
            
            derived = self.derived_calc.calculate(for_derived)
            
            # 6. Build final output
            log.info("\n[6/8] Building final output...")
            final = self.output_builder.build_final_output(
                itl3_history, allocated, rates, gdhi_per_head, lookup
            )
            
            # 7. Validate share stability
            log.info("\n[7/8] Validating share stability...")
            self._validate_share_stability(final, shares)
            
            # 8. Save outputs
            log.info("\n[8/8] Saving outputs...")
            # Close read connection before opening write connection
            self.loader.close()
            self.output_builder.save_outputs(final, shares, derived)
            
        except Exception as e:
            self.loader.close()
            raise e
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Summary
        log.info("")
        log.info("=" * 70)
        log.info(" ✅ ITL3 FORECASTING V5.3 COMPLETE")
        log.info("=" * 70)
        log.info(f"Runtime:        {elapsed:.1f}s")
        log.info(f"Regions:        {final['region_code'].nunique()}")
        log.info(f"Metrics:        {final['metric'].nunique()}")
        log.info(f"Total rows:     {len(final):,}")
        log.info(f"  Historical:   {(final['data_type'] == 'historical').sum():,}")
        log.info(f"  Forecast:     {(final['data_type'] == 'forecast').sum():,}")
        log.info("=" * 70)
        
        return final
    
    def _compute_gdhi_per_head(self, allocated: pd.DataFrame) -> pd.DataFrame:
        """
        Compute gdhi_per_head for ALL years where both components exist.
        data_type = 'forecast' if either component is forecast, else 'historical'.
        """
        log.info("  Computing gdhi_per_head from allocated totals...")
        
        # Remove any duplicate columns first
        alloc = allocated.loc[:, ~allocated.columns.duplicated()].copy()
        
        if alloc.empty:
            return pd.DataFrame()
        
        # Extract gdhi and population - include data_type for dynamic determination
        gdhi = alloc[alloc['metric'] == 'gdhi_total_mn_gbp'][
            ['region_code', 'region_name', 'itl2_code', 'year', 'value', 'ci_lower', 'ci_upper', 'data_type']
        ].copy()
        pop = alloc[alloc['metric'] == 'population_total'][
            ['region_code', 'year', 'value', 'ci_lower', 'ci_upper', 'data_type']
        ].copy()
        
        if gdhi.empty or pop.empty:
            log.warning("  ⚠️ Missing gdhi or population for gdhi_per_head calculation")
            return pd.DataFrame()
        
        # Rename before merge
        gdhi.columns = ['region_code', 'region_name', 'itl2_code', 'year', 
                        'value_gdhi', 'ci_lower_gdhi', 'ci_upper_gdhi', 'data_type_gdhi']
        pop.columns = ['region_code', 'year', 'value_pop', 'ci_lower_pop', 'ci_upper_pop', 'data_type_pop']
        
        # Merge on region + year — gives ALL years where BOTH exist
        merged = gdhi.merge(pop, on=['region_code', 'year'], how='inner')
        
        if merged.empty:
            return pd.DataFrame()
        
        # Dynamic data_type: forecast if EITHER component is forecast
        merged['data_type'] = np.where(
            (merged['data_type_gdhi'] == 'forecast') | (merged['data_type_pop'] == 'forecast'),
            'forecast',
            'historical'
        )
        
        # Handle NaN CIs
        merged['ci_lower_gdhi'] = merged['ci_lower_gdhi'].fillna(merged['value_gdhi'] * 0.95)
        merged['ci_upper_gdhi'] = merged['ci_upper_gdhi'].fillna(merged['value_gdhi'] * 1.05)
        merged['ci_lower_pop'] = merged['ci_lower_pop'].fillna(merged['value_pop'] * 0.98)
        merged['ci_upper_pop'] = merged['ci_upper_pop'].fillna(merged['value_pop'] * 1.02)
        
        # Point estimate
        val = (merged['value_gdhi'] / merged['value_pop']) * 1e6
        ci_lo = (merged['ci_lower_gdhi'] / merged['ci_upper_pop']) * 1e6
        ci_hi = (merged['ci_upper_gdhi'] / merged['ci_lower_pop']) * 1e6
        
        # Ensure CI ordering
        ci_lower = np.minimum(ci_lo, ci_hi)
        ci_upper = np.maximum(ci_lo, ci_hi)
        
        # Build output with dynamic data_type
        result = pd.DataFrame({
            'region_code': merged['region_code'].tolist(),
            'region_name': merged['region_name'].tolist(),
            'itl2_code': merged['itl2_code'].tolist(),
            'year': merged['year'].tolist(),
            'metric': ['gdhi_per_head_gbp'] * len(merged),
            'value': val.tolist(),
            'ci_lower': ci_lower.tolist(),
            'ci_upper': ci_upper.tolist(),
            'data_type': merged['data_type'].tolist(),
            'method': ['deterministic_ratio'] * len(merged),
            'source': ['RegionIQ'] * len(merged)
        })
        
        # Log breakdown
        n_hist = (result['data_type'] == 'historical').sum()
        n_fcst = (result['data_type'] == 'forecast').sum()
        log.info(f"  ✓ Computed {len(result):,} gdhi_per_head values ({n_hist:,} historical, {n_fcst:,} forecast)")
        
        return result
    
    def _validate_share_stability(self, final: pd.DataFrame, shares: pd.DataFrame):
        """Verify that ITL3 shares remain stable across forecast horizon"""
        
        log.info("Checking share stability across forecast years...")
        
        for metric in self.config.additive_metrics:
            metric_data = final[
                (final['metric'] == metric) &
                (final['data_type'] == 'forecast')
            ].copy()
            
            if metric_data.empty:
                continue
            
            # Compute shares by year
            for itl2_code in metric_data['itl2_code'].dropna().unique():
                itl2_data = metric_data[metric_data['itl2_code'] == itl2_code]
                
                if itl2_data.empty:
                    continue
                
                # Get shares per year
                shares_by_year = itl2_data.pivot_table(
                    index='region_code',
                    columns='year',
                    values='value',
                    aggfunc='first'
                )
                
                # Normalize to shares
                shares_by_year = shares_by_year.div(shares_by_year.sum(axis=0), axis=1)
                
                # Check stability (should be constant)
                share_std = shares_by_year.std(axis=1).max()
                
                if share_std > 1e-10:
                    log.warning(f"  ⚠️ {metric}/{itl2_code}: share variance detected (std={share_std:.2e})")
                    break
            else:
                continue
            break
        else:
            log.info("  ✓ All shares stable (constant across forecast years)")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run ITL3 V5 pipeline"""
    
    config = ForecastConfigV5()
    pipeline = ITL3PipelineV5(config)
    
    try:
        result = pipeline.run()
        return result
    except Exception as e:
        log.error(f"❌ Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()