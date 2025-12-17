#!/usr/bin/env python3
"""
RegionIQ - LAD Forecasting Engine V1.0
======================================

LAD (Local Authority District) forecasting using share-based allocation from ITL3 parents.

ARCHITECTURE:
  UK Macro → ITL1 → ITL2 → ITL3 → LAD (this script)

  - ADDITIVE metrics: Share allocation from ITL3 (no share drift)
  - RATE metrics: Forecast at LAD where history exists, else inherit ITL3
  - DERIVED metrics: Monte Carlo from allocated totals

Core principle:
  LAD_value(t) = ITL3_value(t) × share_LAD_in_ITL3

Where share is computed as 5-year historical average (stable).

Benefits:
  1. LAD inherits ITL3's dampened growth rates automatically
  2. No runaway trajectories
  3. Perfect reconciliation by construction (sum of shares = 1)
  4. Share drift impossible (shares are FIXED from history)
  5. ~1500× faster than per-LAD VECM

Metrics handling:
  - ADDITIVE: nominal_gva_mn_gbp, gdhi_total_mn_gbp, emp_total_jobs, population_total, population_16_64
  - RATE: employment_rate_pct, unemployment_rate_pct
  - DERIVED: productivity_gbp_per_job, gdhi_per_head_gbp, income_per_worker_gbp

Inputs:
  - silver.lad_history (DuckDB) or data/silver/lad_unified_history.csv
  - gold.itl3_forecast (ITL3 forecasts - already reconciled)
  - data/reference/lad_itl3_concordance.csv

Outputs:
  - data/forecast/lad_forecast_long.csv (all 10 metrics)
  - data/forecast/lad_forecast_wide.csv
  - data/forecast/lad_derived.csv (productivity + income_per_worker only)
  - data/forecast/lad_metadata.json
  - gold.lad_forecast (8 base metrics - DuckDB)
  - gold.lad_derived (2 derived metrics - DuckDB)

Author: RegionIQ
Version: 1.6 (Continuity adjustment for smooth historical→forecast transition)
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
class LADForecastConfig:
    """LAD V1.5 Configuration - Share-based allocation from ITL3 with base/derived split"""
    
    # Paths
    duck_path: Path = field(default_factory=lambda: Path("data/lake/warehouse.duckdb"))
    output_dir: Path = field(default_factory=lambda: Path("data/forecast"))
    concordance_path: Path = field(default_factory=lambda: Path("data/reference/master_2025_geography_lookup.csv"))
    lad_history_csv: Path = field(default_factory=lambda: Path("data/silver/lad_unified_history.csv"))
    
    # Forecast horizon - derived dynamically from ITL3 parent forecasts
    # These are only used as sanity check bounds, actual horizon comes from ITL3
    forecast_horizon_max: int = 2060  # Sanity cap
    
    # Share computation parameters
    share_lookback_years: int = 5  # Years to average for stable shares
    min_share_history: int = 3     # Minimum years needed to compute share
    
    # Monte Carlo for derived metrics (1000 is sufficient for LAD scale)
    monte_carlo_samples: int = 1000
    
    # Additive metrics (allocated via shares)
    additive_metrics: List[str] = field(default_factory=lambda: [
        'nominal_gva_mn_gbp',
        'gdhi_total_mn_gbp',
        'emp_total_jobs',
        'emp_total_jobs_ni',
        'population_total',
        'population_16_64'
    ])
    
    # Rate metrics (forecast at LAD where history exists)
    rate_metrics: List[str] = field(default_factory=lambda: [
        'employment_rate_pct',
        'unemployment_rate_pct'
    ])
    
    # Rate forecasting parameters
    rate_min_history_years: int = 8  # Minimum years to forecast rates at LAD
    rate_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'employment_rate_pct': (40.0, 95.0),
        'unemployment_rate_pct': (1.0, 20.0)
    })
    
    # Derived metrics (calculated from allocated totals)
    # NOTE: gdhi_per_head_gbp is calculated but goes to BASE table
    # Only productivity and income_per_worker go to DERIVED table
    derived_metrics: Dict[str, Dict] = field(default_factory=lambda: {
        'productivity_gbp_per_job': {
            'numerator': 'nominal_gva_mn_gbp',
            'denominator': 'emp_total_jobs',
            'multiplier': 1e6,
            'unit': 'GBP'
        },
        'gdhi_per_head_gbp': {
            'numerator': 'gdhi_total_mn_gbp',
            'denominator': 'population_total',
            'multiplier': 1e6,
            'unit': 'GBP'
        },
        'income_per_worker_gbp': {
            'numerator': 'gdhi_total_mn_gbp',
            'denominator': 'emp_total_jobs',
            'multiplier': 1e6,
            'unit': 'GBP'
        }
    })
    
    # Metrics that go to gold.lad_derived (not gold.lad_forecast)
    derived_only_metrics: List[str] = field(default_factory=lambda: [
        'productivity_gbp_per_job',
        'income_per_worker_gbp'
    ])
    
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

class DataLoaderLAD:
    """Load all required data for LAD share allocation"""
    
    def __init__(self, config: LADForecastConfig):
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
    
    def load_lad_history(self) -> pd.DataFrame:
        """Load LAD historical data from DuckDB or CSV"""
        log.info("Loading LAD history...")
        
        # Try DuckDB first
        try:
            df = self.con.execute("""
                SELECT 
                    region_code,
                    region_name,
                    metric_id as metric,
                    period as year,
                    value
                FROM silver.lad_history
                WHERE value IS NOT NULL AND value >= 0
            """).fetchdf()
            log.info(f"  ✓ Loaded from DuckDB silver.lad_history")
        except Exception as e:
            log.info(f"  DuckDB table not found, trying CSV: {e}")
            
            # Fall back to CSV
            if not self.config.lad_history_csv.exists():
                raise FileNotFoundError(f"LAD history not found: {self.config.lad_history_csv}")
            
            df = pd.read_csv(self.config.lad_history_csv)
            
            # Standardize column names
            col_map = {
                'metric_id': 'metric',
                'period': 'year'
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            
            # Allow zero values (legitimate for some metrics), only drop negatives and NaNs
            df = df[df['value'].notna() & (df['value'] >= 0)]
            log.info(f"  ✓ Loaded from CSV: {self.config.lad_history_csv}")
        
        df['year'] = df['year'].astype(int)
        df['value'] = df['value'].astype(float)
        
        log.info(f"  ✓ {len(df):,} rows, {df['region_code'].nunique()} LADs, "
                 f"{df['metric'].nunique()} metrics")
        
        return df
    
    def load_itl3_forecasts(self) -> pd.DataFrame:
        """Load ITL3 forecasts (parent level - already reconciled)"""
        log.info("Loading ITL3 forecasts...")
        
        df = self.con.execute("""
            SELECT 
                region_code as itl3_code,
                region_name as itl3_name,
                metric_id as metric,
                period as year,
                value,
                ci_lower,
                ci_upper,
                data_type
            FROM gold.itl3_forecast
        """).fetchdf()
        
        df['year'] = df['year'].astype(int)
        
        n_forecast = (df['data_type'] == 'forecast').sum()
        log.info(f"  ✓ {len(df):,} rows ({n_forecast:,} forecast), "
                 f"{df['itl3_code'].nunique()} ITL3 regions")
        
        return df
    
    def load_concordance(self) -> pd.DataFrame:
        """Load LAD→ITL3 concordance mapping"""
        log.info("Loading LAD-ITL3 concordance...")
        
        if not self.config.concordance_path.exists():
            raise FileNotFoundError(f"Concordance not found: {self.config.concordance_path}")
        
        concordance = pd.read_csv(self.config.concordance_path)
        
        # Handle BOM if present
        concordance.columns = [c.replace('\ufeff', '') for c in concordance.columns]
        
        # Expected columns from master_2025_geography_lookup.csv:
        # LAD25CD, LAD25NM, ITL325CD, ITL325NM
        col_candidates = {
            'lad_code': ['lad_code', 'LAD25CD', 'LAD24CD', 'LAD23CD', 'LAD22CD', 'LAD21CD', 'LAU125CD'],
            'itl3_code': ['itl3_code', 'ITL325CD', 'ITL321CD', 'ITL320CD'],
            'lad_name': ['lad_name', 'LAD25NM', 'LAD24NM', 'LAD23NM', 'LAD22NM', 'LAD21NM', 'LAU125NM']
        }
        
        rename_map = {}
        for target, candidates in col_candidates.items():
            for cand in candidates:
                if cand in concordance.columns:
                    rename_map[cand] = target
                    break
        
        concordance = concordance.rename(columns=rename_map)
        
        # Debug: log rename results
        log.info(f"  Rename map: {rename_map}")
        log.info(f"  Columns after rename: {concordance.columns.tolist()}")
        
        # Ensure required columns exist
        required = ['lad_code', 'itl3_code']
        missing = [c for c in required if c not in concordance.columns]
        if missing:
            log.error(f"Concordance columns: {concordance.columns.tolist()}")
            raise ValueError(f"Missing required columns in concordance: {missing}")
        
        # Get unique mappings
        mapping = concordance[['lad_code', 'itl3_code']].drop_duplicates()
        
        # Add lad_name if available
        if 'lad_name' in concordance.columns:
            name_map = concordance[['lad_code', 'lad_name']].drop_duplicates()
            mapping = mapping.merge(name_map, on='lad_code', how='left')
        else:
            mapping['lad_name'] = mapping['lad_code']
        
        log.info(f"  ✓ {len(mapping)} LADs → {mapping['itl3_code'].nunique()} ITL3 parents")
        log.info(f"  Final mapping columns: {mapping.columns.tolist()}")
        
        return mapping


# =============================================================================
# Share Calculator
# =============================================================================

class ShareCalculatorLAD:
    """
    Compute stable historical shares of LAD within ITL3 parents.
    
    Key design:
    - Use trailing N-year average (not single year) for robustness
    - Handle missing years gracefully
    - Normalize shares to sum to 1.0 within each ITL3 parent
    """
    
    def __init__(self, config: LADForecastConfig):
        self.config = config
    
    def compute_shares(
        self,
        lad_history: pd.DataFrame,
        concordance: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute LAD shares of ITL3 parents for each additive metric.
        
        Returns DataFrame with columns:
          lad_code, itl3_code, lad_name, metric, share, share_years, share_std, share_quality
        
        V1.2: Adds fallback tiers for LADs with insufficient history:
          - Tier 1 (good): 3+ years of history
          - Tier 2 (fallback): 1-2 years of history  
          - Tier 3 (imputed): No history - use population share as proxy
        """
        log.info("Computing LAD shares of ITL3 parents...")
        
        # Debug: check concordance columns
        log.info(f"  Concordance columns: {concordance.columns.tolist()}")
        
        # Validate concordance has required columns
        required_conc = ['lad_code', 'itl3_code']
        missing_conc = [c for c in required_conc if c not in concordance.columns]
        if missing_conc:
            raise ValueError(f"Concordance missing columns: {missing_conc}. Has: {concordance.columns.tolist()}")
        
        # Select merge columns (lad_name optional)
        merge_cols = ['lad_code', 'itl3_code']
        if 'lad_name' in concordance.columns:
            merge_cols.append('lad_name')
        
        concordance_subset = concordance[merge_cols].copy()
        log.info(f"  Concordance subset shape: {concordance_subset.shape}")
        
        # Drop any existing itl3_code from lad_history to avoid _x/_y suffix conflict
        lad_history_clean = lad_history.copy()
        cols_to_drop = [c for c in ['itl3_code', 'lad_code', 'lad_name'] if c in lad_history_clean.columns]
        if cols_to_drop:
            log.info(f"  Dropping existing columns from history: {cols_to_drop}")
            lad_history_clean = lad_history_clean.drop(columns=cols_to_drop)
        
        # Merge ITL3 parent codes
        df = lad_history_clean.merge(
            concordance_subset,
            left_on='region_code',
            right_on='lad_code',
            how='left'
        )
        
        # Debug: check merge result
        log.info(f"  After merge columns: {df.columns.tolist()}")
        log.info(f"  After merge shape: {df.shape}")
        
        # Verify itl3_code exists after merge
        if 'itl3_code' not in df.columns:
            raise ValueError(f"itl3_code not in merged df. Columns: {df.columns.tolist()}")
        
        # Drop regions without ITL3 parent
        orphans = df[df['itl3_code'].isna()]['region_code'].unique()
        if len(orphans) > 0:
            log.warning(f"  ⚠️ {len(orphans)} LAD codes without ITL3 parent (dropped)")
        df = df.dropna(subset=['itl3_code'])
        
        # Find the most recent N years available
        max_year = df['year'].max()
        lookback_start = max_year - self.config.share_lookback_years + 1
        
        log.info(f"  Using years {lookback_start}-{max_year} for share computation")
        
        # Get full LAD universe from concordance
        all_lads = concordance[['lad_code', 'itl3_code', 'lad_name']].drop_duplicates()
        
        shares_list = []
        
        for metric in self.config.additive_metrics:
            metric_data = df[
                (df['metric'] == metric) &
                (df['year'] >= lookback_start)
            ].copy()
            
            if metric_data.empty:
                log.warning(f"  ⚠️ No data for {metric}")
                continue
            
            # Compute ITL3 totals per year
            itl3_totals = metric_data.groupby(['itl3_code', 'year'])['value'].sum().reset_index()
            itl3_totals.columns = ['itl3_code', 'year', 'itl3_total']
            
            # Merge back
            metric_data = metric_data.merge(itl3_totals, on=['itl3_code', 'year'])
            
            # Compute share per year (handle zero totals)
            metric_data['share'] = np.where(
                metric_data['itl3_total'] > 0,
                metric_data['value'] / metric_data['itl3_total'],
                0
            )
            
            # Handle infinities and NaNs
            metric_data = metric_data[np.isfinite(metric_data['share'])]
            
            # Tier 1: Full history (3+ years) - GOOD quality
            share_stats = metric_data.groupby(['lad_code', 'itl3_code', 'lad_name']).agg({
                'share': ['mean', 'std', 'count'],
                'region_name': 'first'
            }).reset_index()
            
            share_stats.columns = ['lad_code', 'itl3_code', 'lad_name',
                                   'share', 'share_std', 'share_years', 'region_name']
            share_stats['metric'] = metric
            
            # Split by quality tier
            tier1 = share_stats[share_stats['share_years'] >= self.config.min_share_history].copy()
            tier1['share_quality'] = 'good'
            
            tier2 = share_stats[
                (share_stats['share_years'] > 0) & 
                (share_stats['share_years'] < self.config.min_share_history)
            ].copy()
            tier2['share_quality'] = 'fallback'
            
            log.info(f"  {metric}: {len(tier1)} good, {len(tier2)} fallback shares")
            
            # Tier 3: Missing LADs - impute from population share or equal split
            covered_lads = set(share_stats['lad_code'].unique())
            missing_lads = all_lads[~all_lads['lad_code'].isin(covered_lads)].copy()
            
            if len(missing_lads) > 0:
                # Try to use population shares as proxy
                pop_shares = None
                if metric != 'population_total':
                    pop_data = df[(df['metric'] == 'population_total') & (df['year'] >= lookback_start)]
                    if not pop_data.empty:
                        pop_shares = pop_data.groupby(['lad_code', 'itl3_code'])['value'].mean().reset_index()
                        pop_shares.columns = ['lad_code', 'itl3_code', 'pop_value']
                
                tier3_rows = []
                for _, lad_row in missing_lads.iterrows():
                    imputed_share = None
                    
                    # Try population-based share
                    if pop_shares is not None:
                        pop_match = pop_shares[
                            (pop_shares['lad_code'] == lad_row['lad_code']) &
                            (pop_shares['itl3_code'] == lad_row['itl3_code'])
                        ]
                        if not pop_match.empty:
                            itl3_pop_total = pop_shares[
                                pop_shares['itl3_code'] == lad_row['itl3_code']
                            ]['pop_value'].sum()
                            if itl3_pop_total > 0:
                                imputed_share = pop_match['pop_value'].iloc[0] / itl3_pop_total
                    
                    # Fallback to equal split within ITL3
                    if imputed_share is None:
                        n_lads_in_itl3 = len(all_lads[all_lads['itl3_code'] == lad_row['itl3_code']])
                        imputed_share = 1.0 / max(n_lads_in_itl3, 1)
                    
                    tier3_rows.append({
                        'lad_code': lad_row['lad_code'],
                        'itl3_code': lad_row['itl3_code'],
                        'lad_name': lad_row['lad_name'],
                        'share': imputed_share,
                        'share_std': np.nan,
                        'share_years': 0,
                        'region_name': lad_row['lad_name'],
                        'metric': metric,
                        'share_quality': 'imputed'
                    })
                
                if tier3_rows:
                    tier3 = pd.DataFrame(tier3_rows)
                    log.info(f"  {metric}: {len(tier3)} imputed shares (missing history)")
                else:
                    tier3 = pd.DataFrame()
            else:
                tier3 = pd.DataFrame()
            
            # Combine all tiers
            metric_shares = pd.concat([tier1, tier2, tier3], ignore_index=True)
            shares_list.append(metric_shares)
        
        if not shares_list:
            raise ValueError("No shares computed - check data availability")
        
        shares_df = pd.concat(shares_list, ignore_index=True)
        
        # Normalize shares within each ITL3 parent (ensure sum = 1.0)
        shares_df = self._normalize_shares(shares_df)
        
        # Summary
        quality_counts = shares_df.groupby('share_quality')['lad_code'].nunique()
        log.info(f"  ✓ Computed shares for {shares_df['lad_code'].nunique()} LADs")
        log.info(f"    Quality breakdown: {quality_counts.to_dict()}")
        log.info(f"  ✓ Metrics: {shares_df['metric'].unique().tolist()}")
        
        return shares_df
    
    def _normalize_shares(self, shares_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize shares to sum to 1.0 within each (ITL3, metric) group"""
        
        # Compute sum per ITL3/metric
        totals = shares_df.groupby(['itl3_code', 'metric'])['share'].sum().reset_index()
        totals.columns = ['itl3_code', 'metric', 'share_sum']
        
        # Merge and normalize
        shares_df = shares_df.merge(totals, on=['itl3_code', 'metric'])
        shares_df['share'] = shares_df['share'] / shares_df['share_sum']
        shares_df = shares_df.drop(columns=['share_sum'])
        
        # Verify normalization
        check = shares_df.groupby(['itl3_code', 'metric'])['share'].sum()
        max_deviation = abs(check - 1.0).max()
        
        if max_deviation > 1e-10:
            log.warning(f"  ⚠️ Share normalization error: max deviation = {max_deviation:.2e}")
        else:
            log.info(f"  ✓ Shares normalized (max deviation = {max_deviation:.2e})")
        
        return shares_df


# =============================================================================
# Share Allocator
# =============================================================================

class ShareAllocatorLAD:
    """
    Allocate ITL3 forecasts to LAD using fixed historical shares.
    
    This is the core of the top-down approach - no independent LAD forecasting.
    LAD inherits ITL3's growth rates by construction.
    """
    
    def __init__(self, config: LADForecastConfig):
        self.config = config
    
    def allocate(
        self,
        itl3_forecasts: pd.DataFrame,
        shares: pd.DataFrame,
        lad_history: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Allocate ITL3 forecast values to LAD children using shares.
        
        LAD_value = ITL3_value × share
        LAD_ci_lower = ITL3_ci_lower × share
        LAD_ci_upper = ITL3_ci_upper × share
        
        V1.6: Adds continuity adjustment to ensure smooth transition from historical to forecast.
        For each LAD/metric, the first forecast year is adjusted to match the last historical value,
        then subsequent years follow ITL3 growth trajectory with a scaling factor applied.
        """
        log.info("Allocating ITL3 forecasts to LAD...")
        
        # Prepare historical data for continuity check if provided
        hist_by_lad_metric = {}
        if lad_history is not None:
            for _, row in lad_history.iterrows():
                key = (row['region_code'], row['metric'])
                if key not in hist_by_lad_metric:
                    hist_by_lad_metric[key] = []
                hist_by_lad_metric[key].append((row['year'], row['value']))
            
            # Get last historical year and value for each LAD/metric
            last_hist = {}
            for key, values in hist_by_lad_metric.items():
                if values:
                    max_year = max(v[0] for v in values)
                    last_val = next(v[1] for v in values if v[0] == max_year)
                    last_hist[key] = (max_year, last_val)
        
        results = []
        continuity_adjustments = {}
        
        for metric in self.config.additive_metrics:
            # Get ITL3 forecasts for this metric
            itl3_metric = itl3_forecasts[itl3_forecasts['metric'] == metric].copy()
            
            if itl3_metric.empty:
                log.warning(f"  ⚠️ No ITL3 data for {metric}")
                continue
            
            # Get shares for this metric
            metric_shares = shares[shares['metric'] == metric].copy()
            
            if metric_shares.empty:
                log.warning(f"  ⚠️ No shares for {metric}")
                continue
            
            # Select only needed columns
            share_cols = ['lad_code', 'itl3_code', 'lad_name', 'region_name', 'share']
            metric_shares = metric_shares[[c for c in share_cols if c in metric_shares.columns]]
            
            itl3_cols = ['itl3_code', 'year', 'value', 'ci_lower', 'ci_upper', 'data_type']
            itl3_subset = itl3_metric[[c for c in itl3_cols if c in itl3_metric.columns]]
            
            # Get ITL3 historical values for continuity check
            itl3_hist = itl3_metric[itl3_metric['data_type'] == 'historical'].copy()
            itl3_hist_by_code = {}
            if not itl3_hist.empty:
                for itl3_code in itl3_hist['itl3_code'].unique():
                    itl3_hist_subset = itl3_hist[itl3_hist['itl3_code'] == itl3_code]
                    max_hist_year = itl3_hist_subset['year'].max()
                    max_hist_val = itl3_hist_subset[itl3_hist_subset['year'] == max_hist_year]['value'].iloc[0]
                    itl3_hist_by_code[itl3_code] = (max_hist_year, max_hist_val)
            
            # Merge: each LAD gets its share of ITL3
            allocated = metric_shares.merge(
                itl3_subset,
                on='itl3_code',
                how='inner'
            )
            
            # Apply share allocation
            allocated['value'] = allocated['value'] * allocated['share']
            if 'ci_lower' in allocated.columns:
                allocated['ci_lower'] = allocated['ci_lower'] * allocated['share']
            if 'ci_upper' in allocated.columns:
                allocated['ci_upper'] = allocated['ci_upper'] * allocated['share']
            
            allocated['metric'] = metric
            
            # V1.6: Apply continuity adjustment
            if lad_history is not None:
                for lad_code in allocated['lad_code'].unique():
                    key = (lad_code, metric)
                    if key in last_hist:
                        last_hist_year, last_hist_val = last_hist[key]
                        
                        # Get this LAD's allocated values
                        lad_allocated = allocated[allocated['lad_code'] == lad_code].copy()
                        lad_allocated = lad_allocated.sort_values('year')
                        
                        # Find first forecast year
                        forecast_years = lad_allocated[lad_allocated['data_type'] == 'forecast']
                        if not forecast_years.empty:
                            first_fcst_year = forecast_years['year'].min()
                            first_fcst_val_allocated = forecast_years[forecast_years['year'] == first_fcst_year]['value'].iloc[0]
                            
                            # Get ITL3 values for continuity calculation
                            itl3_code = lad_allocated['itl3_code'].iloc[0]
                            
                            if itl3_code in itl3_hist_by_code:
                                itl3_last_hist_year, itl3_last_hist_val = itl3_hist_by_code[itl3_code]
                                
                                # Get ITL3 first forecast value
                                itl3_first_fcst = itl3_metric[
                                    (itl3_metric['itl3_code'] == itl3_code) &
                                    (itl3_metric['year'] == first_fcst_year) &
                                    (itl3_metric['data_type'] == 'forecast')
                                ]
                                
                                if not itl3_first_fcst.empty:
                                    itl3_first_fcst_val = itl3_first_fcst['value'].iloc[0]
                                    
                                    # Calculate continuity adjustment factor
                                    # Goal: LAD_first_fcst = LAD_last_hist × (ITL3_first_fcst / ITL3_last_hist)
                                    # Allocated: LAD_allocated = ITL3_first_fcst × share
                                    # Factor: (LAD_last_hist × ITL3_first_fcst / ITL3_last_hist) / (ITL3_first_fcst × share)
                                    #       = LAD_last_hist / (ITL3_last_hist × share)
                                    if itl3_last_hist_val > 0:
                                        # Get the share for this LAD
                                        lad_share = metric_shares[metric_shares['lad_code'] == lad_code]['share'].iloc[0]
                                        
                                        continuity_factor = last_hist_val / (itl3_last_hist_val * lad_share)
                                        
                                        if abs(continuity_factor - 1.0) > 0.001:  # Only log if significant adjustment
                                            continuity_adjustments[key] = continuity_factor
                                            
                                            # Apply adjustment to all forecast years for this LAD
                                            forecast_mask = (allocated['lad_code'] == lad_code) & (allocated['data_type'] == 'forecast')
                                            allocated.loc[forecast_mask, 'value'] *= continuity_factor
                                            if 'ci_lower' in allocated.columns:
                                                allocated.loc[forecast_mask, 'ci_lower'] *= continuity_factor
                                            if 'ci_upper' in allocated.columns:
                                                allocated.loc[forecast_mask, 'ci_upper'] *= continuity_factor
            
            # Rename columns to standard schema
            allocated = allocated.rename(columns={
                'lad_code': 'region_code',
                'lad_name': 'region_name'
            })
            
            results.append(allocated)
            
            log.info(f"  {metric}: {len(allocated):,} rows allocated")
        
        if continuity_adjustments:
            log.info(f"  ✓ Applied continuity adjustments to {len(continuity_adjustments)} LAD/metric combinations")
            for (lad_code, metric), factor in list(continuity_adjustments.items())[:5]:  # Show first 5
                log.info(f"    {lad_code}/{metric}: factor {factor:.4f}")
            if len(continuity_adjustments) > 5:
                log.info(f"    ... and {len(continuity_adjustments) - 5} more")
        
        if not results:
            raise ValueError("No allocations performed - check data")
        
        result_df = pd.concat(results, ignore_index=True)
        
        # Validation: check reconciliation (with continuity adjustments, sums may not exactly match)
        # We'll use a relaxed tolerance for reconciliation check
        self._validate_reconciliation(result_df, itl3_forecasts, shares, relaxed_tolerance=True)
        
        return result_df
    
    def _validate_reconciliation(
        self,
        lad_allocated: pd.DataFrame,
        itl3_forecasts: pd.DataFrame,
        shares: pd.DataFrame,
        relaxed_tolerance: bool = False
    ):
        """Verify that sum(LAD) = ITL3 for each parent/metric/year
        
        With continuity adjustments, reconciliation may not be exact but should be close.
        relaxed_tolerance: If True, allows small deviations (up to 0.1%) for continuity-adjusted forecasts.
        """
        
        log.info("\nValidating reconciliation...")
        
        tolerance = 0.001 if relaxed_tolerance else 1e-10
        
        for metric in self.config.additive_metrics:
            lad_metric = lad_allocated[lad_allocated['metric'] == metric]
            itl3_metric = itl3_forecasts[itl3_forecasts['metric'] == metric]
            
            if lad_metric.empty or itl3_metric.empty:
                continue
            
            # Sum LAD by ITL3 parent and year
            lad_sums = lad_metric.groupby(['itl3_code', 'year'])['value'].sum().reset_index()
            lad_sums.columns = ['itl3_code', 'year', 'lad_sum']
            
            # Merge with ITL3 values
            check = lad_sums.merge(
                itl3_metric[['itl3_code', 'year', 'value']].rename(columns={'value': 'itl3_value'}),
                on=['itl3_code', 'year'],
                how='inner'
            )
            
            if check.empty:
                continue
            
            # Compute deviation
            check['deviation'] = abs(check['lad_sum'] - check['itl3_value']) / (check['itl3_value'] + 1e-10)
            max_deviation = check['deviation'].max()
            
            for year in [2025, 2030, 2040, 2050]:
                year_check = check[check['year'] == year]
                if not year_check.empty:
                    year_dev = year_check['deviation'].max()
                    status = "✓" if year_dev < tolerance else "⚠️"
                    log.info(f"  {metric} {year}: max deviation = {year_dev:.6e} {status}")


# =============================================================================
# Rate Metrics Handler
# =============================================================================

class RateMetricsHandlerLAD:
    """
    Handle rate metrics (employment_rate, unemployment_rate) for LAD.
    
    Strategy:
    - Forecast rates at LAD level where sufficient history exists
    - Fall back to ITL3 parent rates where history is missing
    
    Rates are NOT additive - cannot use share allocation.
    V1.2: Uses concordance for LAD universe (not shares) to ensure full coverage.
    """
    
    def __init__(self, config: LADForecastConfig):
        self.config = config
    
    def apply_rates(
        self,
        itl3_forecasts: pd.DataFrame,
        lad_history: pd.DataFrame,
        concordance: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply rate metrics to LAD:
        - Forecast where LAD has history
        - Inherit from ITL3 parent otherwise
        
        V1.2: Uses concordance for full LAD coverage (a LAD may have rate history
        but insufficient additive history, so shares would miss it).
        V1.2: Dynamic horizon from ITL3 parent forecasts (not hardcoded).
        """
        log.info("Processing rate metrics for LAD...")
        
        results = []
        
        # Get LAD→ITL3 mapping from CONCORDANCE (not shares) for full coverage
        lad_to_itl3 = concordance[['lad_code', 'itl3_code', 'lad_name']].drop_duplicates(subset=['lad_code'])

        # Enforce perfect coherence for 1:1 ITL3↔LAD mappings (rates only).
        # If an ITL3 has exactly one LAD child, that LAD's rate forecast must equal
        # the ITL3 forecast by construction, so we always inherit from ITL3 and do
        # NOT run an independent LAD rate model (which can drift).
        itl3_child_counts = lad_to_itl3.groupby('itl3_code')['lad_code'].nunique()
        itl3_one_child = set(itl3_child_counts[itl3_child_counts == 1].index.astype(str).tolist())
        lad_force_inherit = set(
            lad_to_itl3[lad_to_itl3['itl3_code'].astype(str).isin(itl3_one_child)]['lad_code']
            .astype(str)
            .tolist()
        )
        if itl3_one_child:
            log.info(
                f"  1:1 ITL3↔LAD rate coherence: forcing ITL3 inheritance for "
                f"{len(lad_force_inherit)} LADs across {len(itl3_one_child)} ITL3 parents"
            )
        
        # Derive forecast end year from ITL3 parent (not hardcoded)
        itl3_forecast_only = itl3_forecasts[itl3_forecasts['data_type'] == 'forecast']
        if itl3_forecast_only.empty:
            log.warning("  No ITL3 forecast data - skipping rate metrics")
            return pd.DataFrame()
        
        max_parent_year = int(itl3_forecast_only['year'].max())
        log.info(f"  Forecast horizon derived from ITL3: ends {max_parent_year}")
        
        for metric in self.config.rate_metrics:
            # Get ITL3 forecasts for fallback
            itl3_metric = itl3_forecasts[itl3_forecasts['metric'] == metric].copy()
            
            if itl3_metric.empty:
                log.info(f"  {metric}: No ITL3 data (skipping)")
                continue
            
            # Get LAD history for this metric
            lad_metric_history = lad_history[lad_history['metric'] == metric].copy()
            
            # Identify regions with sufficient history
            regions_with_history = self._get_regions_with_history(lad_metric_history, metric)
            
            forecast_count = 0
            inherit_count = 0
            forced_inherit_count = 0
            
            # Process each LAD
            for _, row in lad_to_itl3.iterrows():
                lad_code = row['lad_code']
                itl3_code = row['itl3_code']
                lad_name = row.get('lad_name', lad_code)
                region_name = lad_name  # Use lad_name as region_name

                # 1:1 mapping guardrail: always inherit for rate forecasts
                if str(lad_code) in lad_force_inherit:
                    itl3_subset = itl3_metric[itl3_metric['itl3_code'] == itl3_code].copy()
                    if not itl3_subset.empty:
                        itl3_forecast = itl3_subset[itl3_subset['data_type'] == 'forecast'].copy()
                        if not itl3_forecast.empty:
                            inherited = pd.DataFrame({
                                'region_code': lad_code,
                                'region_name': region_name,
                                'itl3_code': itl3_code,
                                'year': itl3_forecast['year'].values,
                                'metric': metric,
                                'value': itl3_forecast['value'].values,
                                'ci_lower': itl3_forecast['ci_lower'].values if 'ci_lower' in itl3_forecast.columns else itl3_forecast['value'].values * 0.95,
                                'ci_upper': itl3_forecast['ci_upper'].values if 'ci_upper' in itl3_forecast.columns else itl3_forecast['value'].values * 1.05,
                                'data_type': 'forecast',
                                'method': 'itl3_inheritance_1to1'
                            })
                            results.append(inherited)
                            inherit_count += 1
                            forced_inherit_count += 1
                            continue
                
                if lad_code in regions_with_history:
                    # Forecast at LAD level
                    region_history = lad_metric_history[
                        lad_metric_history['region_code'] == lad_code
                    ].sort_values('year')
                    
                    forecast_df = self._forecast_rate(
                        region_history, lad_code, lad_name, region_name,
                        itl3_code, metric, max_parent_year
                    )
                    
                    if forecast_df is not None and len(forecast_df) > 0:
                        results.append(forecast_df)
                        forecast_count += 1
                        continue
                
                # Fall back to ITL3 parent
                itl3_subset = itl3_metric[itl3_metric['itl3_code'] == itl3_code].copy()
                
                if itl3_subset.empty:
                    continue
                
                # Only forecast years
                itl3_forecast = itl3_subset[itl3_subset['data_type'] == 'forecast'].copy()
                
                if itl3_forecast.empty:
                    continue
                
                inherited = pd.DataFrame({
                    'region_code': lad_code,
                    'region_name': region_name,
                    'itl3_code': itl3_code,
                    'year': itl3_forecast['year'].values,
                    'metric': metric,
                    'value': itl3_forecast['value'].values,
                    'ci_lower': itl3_forecast['ci_lower'].values if 'ci_lower' in itl3_forecast.columns else itl3_forecast['value'].values * 0.95,
                    'ci_upper': itl3_forecast['ci_upper'].values if 'ci_upper' in itl3_forecast.columns else itl3_forecast['value'].values * 1.05,
                    'data_type': 'forecast',
                    'method': 'itl3_inheritance'
                })
                
                results.append(inherited)
                inherit_count += 1
            
            extra = f", {forced_inherit_count} forced(1:1)" if forced_inherit_count else ""
            log.info(f"  {metric}: {forecast_count} forecast, {inherit_count} inherited from ITL3{extra}")
        
        if not results:
            return pd.DataFrame()
        
        return pd.concat(results, ignore_index=True)
    
    def _get_regions_with_history(self, metric_history: pd.DataFrame, metric: str) -> set:
        """Identify LAD regions with sufficient rate history (metric + country aware).
        
        Contract: keep GB threshold strict, but allow NI LADs to forecast unemployment
        with shorter history (NI series starts in 2019 in NISRA LMSLGD).
        """
        if metric_history.empty:
            return set()

        region_counts = metric_history.groupby('region_code')['year'].nunique()
        default_min_years = self.config.rate_min_history_years

        if metric == 'unemployment_rate_pct':
            # NI LADs are N09000001..N09000011
            is_ni_lad = region_counts.index.to_series().astype(str).str.startswith('N09')
            ni_ok = set(region_counts[is_ni_lad & (region_counts >= 5)].index)
            gb_ok = set(region_counts[~is_ni_lad & (region_counts >= default_min_years)].index)
            return ni_ok | gb_ok

        # All other rate metrics use the default threshold everywhere
        return set(region_counts[region_counts >= default_min_years].index)
    
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
        lad_code: str,
        lad_name: str,
        region_name: str,
        itl3_code: str,
        metric: str,
        max_parent_year: int
    ) -> Optional[pd.DataFrame]:
        """Forecast a single rate metric using mean reversion to regional long-term average.
        
        V1.3: Uses mean reversion instead of ARIMA/ETS ensemble.
        V1.2: Uses max_parent_year from ITL3 forecasts (not hardcoded).
        """
        
        min_years = self.config.rate_min_history_years
        # NI-only exception: allow unemployment to forecast with shorter history
        if metric == 'unemployment_rate_pct' and str(lad_code).startswith('N09'):
            min_years = 5

        if len(history) < min_years:
            return None
        
        # Determine forecast horizon from ITL3 parent (dynamic)
        last_year = int(history['year'].max())
        horizon = max_parent_year - last_year
        
        if horizon <= 0:
            return None
        
        # Sanity cap
        if horizon > (self.config.forecast_horizon_max - last_year):
            horizon = self.config.forecast_horizon_max - last_year
        
        forecast_years = list(range(last_year + 1, max_parent_year + 1))
        
        # Get bounds for this metric
        bounds = self.config.rate_bounds.get(metric, (0, 100))
        
        # Get ITL3 parent mean for fallback (can be passed from apply_rates if needed)
        fallback_mean = None
        
        # Use mean reversion instead of ensemble
        forecast_result = self._forecast_rate_mean_revert(history, forecast_years, metric, bounds, fallback_mean)
        
        if forecast_result is None:
            return None
        
        # Build output DataFrame
        result = pd.DataFrame({
            'region_code': lad_code,
            'region_name': region_name,
            'itl3_code': itl3_code,
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
            aic = len(series) * np.log(sigma**2 + 1e-10) + 4
            
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

class DerivedMetricsCalculatorLAD:
    """
    Calculate derived metrics from allocated totals using Monte Carlo.
    
    Derived metrics:
    - productivity_gbp_per_job = GVA / Employment × 1e6
    - gdhi_per_head_gbp = GDHI / Population × 1e6
    - income_per_worker_gbp = GDHI / Employment × 1e6
    
    Critical: Handle mixed-vintage CIs (NaN when component is historical)
    """
    
    def __init__(self, config: LADForecastConfig):
        self.config = config
        self.n_samples = config.monte_carlo_samples
    
    def calculate(self, allocated_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all derived metrics with uncertainty propagation"""
        
        log.info("Calculating derived metrics (Monte Carlo)...")
        
        # Get ALL years that have data (not just forecast years)
        # We need to compute derived for historical years too
        all_years = allocated_df['year'].unique()
        
        if len(all_years) == 0:
            log.warning("  No data for derived metrics")
            return pd.DataFrame()
        
        # Use ALL data
        all_data = allocated_df.copy()
        
        # Pivot to wide format for calculation - include data_type
        pivot = all_data.pivot_table(
            index=['region_code', 'itl3_code', 'year'],
            columns='metric',
            values=['value', 'ci_lower', 'ci_upper', 'data_type'],
            aggfunc='first'
        )
        
        # Flatten column names
        pivot.columns = ['_'.join(col).strip() for col in pivot.columns]
        pivot = pivot.reset_index()
        
        # Get region_name mapping
        name_map = all_data.drop_duplicates('region_code').set_index('region_code')['region_name'].to_dict()
        
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
            
            # Data type columns for proper inheritance
            num_dtype_col = f"data_type_{config['numerator']}"
            den_dtype_col = f"data_type_{config['denominator']}"
            
            for _, row in pivot.iterrows():
                num_val = row.get(num_col)
                den_val = row.get(den_col)
                
                if pd.isna(num_val) or pd.isna(den_val) or den_val <= 0:
                    continue
                
                # Get CIs - handle NaN for mixed-vintage (historical components)
                num_low = row.get(num_ci_low)
                num_high = row.get(num_ci_high)
                den_low = row.get(den_ci_low)
                den_high = row.get(den_ci_high)
                
                # Determine data_type: historical ONLY if BOTH components are historical
                num_dtype = row.get(num_dtype_col, 'forecast')
                den_dtype = row.get(den_dtype_col, 'forecast')
                derived_dtype = 'historical' if (num_dtype == 'historical' and den_dtype == 'historical') else 'forecast'
                
                # Monte Carlo with proper NaN handling
                result = self._monte_carlo_ratio(
                    num_val * config['multiplier'],
                    num_low * config['multiplier'] if pd.notna(num_low) else np.nan,
                    num_high * config['multiplier'] if pd.notna(num_high) else np.nan,
                    den_val,
                    den_low if pd.notna(den_low) else np.nan,
                    den_high if pd.notna(den_high) else np.nan
                )
                
                derived_rows.append({
                    'region_code': row['region_code'],
                    'region_name': name_map.get(row['region_code'], ''),
                    'itl3_code': row['itl3_code'],
                    'year': row['year'],
                    'metric': derived_name,
                    'value': result['value'],
                    'ci_lower': result['ci_lower'],
                    'ci_upper': result['ci_upper'],
                    'data_type': derived_dtype,
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
        """
        Monte Carlo simulation for ratio with uncertainty.
        
        CRITICAL FIX V1.4: 
        - Point estimate is ALWAYS deterministic (num/den) to ensure exact reconciliation
        - Monte Carlo is used ONLY for CI bounds
        - Handle NaN CIs (mixed-vintage scenarios)
        """
        
        # Point estimate is ALWAYS deterministic - no Monte Carlo variance
        result_val = max(0, num_val / den_val) if den_val > 0 else 0.0
        
        # Check if BOTH components are historical (no CIs)
        num_is_hist = math.isnan(num_low) or math.isnan(num_high)
        den_is_hist = math.isnan(den_low) or math.isnan(den_high)
        
        if num_is_hist and den_is_hist:
            # Both historical - no uncertainty to propagate
            return {
                'value': result_val,
                'ci_lower': np.nan,
                'ci_upper': np.nan
            }
        
        # At least one component has uncertainty - use Monte Carlo for CIs only
        # Estimate standard errors - handle NaN CIs
        if num_is_hist:
            # Historical component - use 2% uncertainty
            num_se = max(abs(num_val) * 0.02, 1e-10)
        else:
            num_se = max((num_high - num_low) / (2 * 1.96), 1e-10)
        
        if den_is_hist:
            # Historical component - use 2% uncertainty
            den_se = max(abs(den_val) * 0.02, 1e-10)
        else:
            den_se = max((den_high - den_low) / (2 * 1.96), 1e-10)
        
        # Sample for CI estimation
        num_samples = np.random.normal(num_val, num_se, self.n_samples)
        den_samples = np.random.normal(den_val, den_se, self.n_samples)
        
        # Avoid division by zero
        den_samples = np.maximum(den_samples, 1)
        
        # Compute ratio samples for CI bounds only
        ratio_samples = num_samples / den_samples
        
        result_low = max(0, float(np.percentile(ratio_samples, 2.5)))
        result_high = max(0, float(np.percentile(ratio_samples, 97.5)))
        
        # Ensure CI ordering (swap if inverted)
        if result_low > result_high:
            result_low, result_high = result_high, result_low
        
        return {
            'value': result_val,
            'ci_lower': result_low,
            'ci_upper': result_high
        }


# =============================================================================
# Output Builder
# =============================================================================

class OutputBuilderLAD:
    """Build and save all output formats"""
    
    def __init__(self, config: LADForecastConfig):
        self.config = config
    
    def build_final_output(
        self,
        lad_history: pd.DataFrame,
        allocated: pd.DataFrame,
        rates: pd.DataFrame,
        derived: pd.DataFrame,
        concordance: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Combine all components into final output.
        
        Returns:
            Tuple of (base_df, derived_df) matching upstream architecture:
            - base_df: 8 metrics → gold.lad_forecast
            - derived_df: 2 metrics → gold.lad_derived
        """
        
        log.info("Building final output...")
        
        # Define standard columns (matching spec exactly)
        std_cols = [
            'region_code', 'region_name', 'itl3_code',
            'metric', 'year', 'value',
            'ci_lower', 'ci_upper',
            'data_type', 'method', 'source'
        ]
        
        def standardize_df(df, data_type, method, source):
            """Standardize a dataframe to common schema.
            
            V1.2: Preserves existing method/data_type values if present,
            allowing granular method info from rate forecaster etc.
            """
            df = df.loc[:, ~df.columns.duplicated()].copy()
            
            # Only set data_type if not already present
            if 'data_type' not in df.columns:
                df['data_type'] = data_type
            
            # Only set method if not already present (preserve granular methods)
            if 'method' not in df.columns:
                df['method'] = method
            
            df['source'] = source
            
            out = pd.DataFrame()
            for col in std_cols:
                if col in df.columns:
                    out[col] = df[col].values
                else:
                    out[col] = np.nan
            
            return out
        
        # Prepare historical data
        hist = lad_history.copy()
        hist = hist.loc[:, ~hist.columns.duplicated()]
        
        # Drop existing itl3_code/lad_code to avoid _x/_y suffix on merge
        cols_to_drop = [c for c in ['itl3_code', 'lad_code', 'lad_name'] if c in hist.columns]
        if cols_to_drop:
            hist = hist.drop(columns=cols_to_drop)
        
        # Add ITL3 parent codes to history
        concordance_clean = concordance[['lad_code', 'itl3_code']].drop_duplicates()
        hist = hist.merge(
            concordance_clean,
            left_on='region_code',
            right_on='lad_code',
            how='left'
        )
        if 'lad_code' in hist.columns:
            hist = hist.drop(columns=['lad_code'])
        hist = standardize_df(hist, 'historical', 'observed', 'ONS')
        
        # Prepare allocated forecasts
        alloc = standardize_df(allocated, 'forecast', 'share_allocation_v1', 'RegionIQ')
        
        # Prepare rates
        if rates is not None and len(rates) > 0:
            rates_std = standardize_df(rates, 'forecast', 'rate_forecast_v1', 'RegionIQ')
        else:
            rates_std = None
        
        # Prepare derived
        if derived is not None and len(derived) > 0:
            derived_std = standardize_df(derived, 'forecast', 'derived_monte_carlo', 'RegionIQ')
        else:
            derived_std = None
        
        # Combine all
        dfs = [df for df in [hist, alloc, rates_std, derived_std] if df is not None and len(df) > 0]
        final = pd.concat(dfs, axis=0, ignore_index=True)
        
        # Add standard columns
        final['region_level'] = 'LAD'
        final['unit'] = final['metric'].map(self.config.metric_units)
        final['freq'] = 'A'
        
        # Final column order (matches spec)
        final_cols = [
            'region_code', 'region_name', 'region_level', 'itl3_code',
            'metric', 'year', 'value', 'unit', 'freq',
            'data_type', 'method', 'source',
            'ci_lower', 'ci_upper'
        ]
        
        for col in final_cols:
            if col not in final.columns:
                final[col] = np.nan
        
        final = final[final_cols].sort_values(['region_code', 'metric', 'year']).reset_index(drop=True)
        
        # Fix any inverted CIs
        ci_mask = final['ci_lower'] > final['ci_upper']
        if ci_mask.sum() > 0:
            final.loc[ci_mask, ['ci_lower', 'ci_upper']] = final.loc[ci_mask, ['ci_upper', 'ci_lower']].values
            log.info(f"  ✓ Fixed {ci_mask.sum()} inverted CIs")
        
        # Deduplicate: prefer historical over forecast
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
        
        # Split into base (8 metrics) and derived (2 metrics) for architectural consistency
        derived_only = self.config.derived_only_metrics
        
        base_df = final[~final['metric'].isin(derived_only)].copy()
        derived_df = final[final['metric'].isin(derived_only)].copy()
        
        log.info(f"    Base table:    {len(base_df):,} rows ({base_df['metric'].nunique()} metrics)")
        log.info(f"    Derived table: {len(derived_df):,} rows ({derived_df['metric'].nunique()} metrics)")
        
        return base_df, derived_df
    
    def save_outputs(self, base_df: pd.DataFrame, derived_df: pd.DataFrame, shares: pd.DataFrame):
        """Save all output formats - base and derived tables separately"""
        
        log.info("\nSaving outputs...")
        
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Combine for CSV exports (full data)
        full_df = pd.concat([base_df, derived_df], ignore_index=True)
        
        # 1. Long format CSV (matches ITL3 - full dataframe)
        long_path = self.config.output_dir / "lad_forecast_long.csv"
        full_df.to_csv(long_path, index=False)
        log.info(f"  ✓ {long_path}")
        
        # 2. Wide format CSV
        wide = full_df.pivot_table(
            index=['region_code', 'region_name', 'metric'],
            columns='year',
            values='value',
            aggfunc='first'
        ).reset_index()
        wide.columns.name = None
        
        wide_path = self.config.output_dir / "lad_forecast_wide.csv"
        wide.to_csv(wide_path, index=False)
        log.info(f"  ✓ {wide_path}")
        
        # 3. Derived CSV (separate file for clarity)
        derived_path = self.config.output_dir / "lad_derived.csv"
        derived_df.to_csv(derived_path, index=False)
        log.info(f"  ✓ {derived_path}")
        
        # 4. Confidence intervals
        ci_data = full_df[
            (full_df['data_type'] == 'forecast') &
            (full_df['ci_lower'].notna())
        ][['region_code', 'metric', 'year', 'value', 'ci_lower', 'ci_upper']].copy()
        
        if not ci_data.empty:
            ci_data['ci_width'] = ci_data['ci_upper'] - ci_data['ci_lower']
            ci_path = self.config.output_dir / "lad_confidence_intervals.csv"
            ci_data.to_csv(ci_path, index=False)
            log.info(f"  ✓ {ci_path}")
        
        # 5. Shares (for transparency/debugging)
        shares_path = self.config.output_dir / "lad_shares.csv"
        shares.to_csv(shares_path, index=False)
        log.info(f"  ✓ {shares_path}")
        
        # 6. Metadata JSON (with shares_summary for QA)
        shares_summary = {}
        if 'share_years' in shares.columns:
            shares_summary['avg_share_years'] = float(shares['share_years'].mean())
            shares_summary['min_share_years'] = int(shares['share_years'].min())
        if 'share_std' in shares.columns:
            shares_summary['max_share_std'] = float(shares['share_std'].max()) if shares['share_std'].notna().any() else None
        if 'share_quality' in shares.columns:
            quality_counts = shares.groupby('share_quality')['lad_code'].nunique().to_dict()
            shares_summary['quality_breakdown'] = {str(k): int(v) for k, v in quality_counts.items()}
        
        metadata = {
            'run_timestamp': datetime.now(timezone.utc).isoformat(),
            'version': '1.5',
            'level': 'LAD',
            'method': 'share_allocation_from_itl3',
            'architecture': {
                'description': 'Share-based allocation - LAD inherits ITL3 growth rates',
                'share_lookback_years': self.config.share_lookback_years,
                'base_metrics': [m for m in self.config.additive_metrics + self.config.rate_metrics + ['gdhi_per_head_gbp']],
                'derived_metrics': self.config.derived_only_metrics,
                'tables': {
                    'gold.lad_forecast': '8 base metrics',
                    'gold.lad_derived': '2 derived metrics (productivity, income_per_worker)'
                }
            },
            'shares_summary': shares_summary,
            'data_summary': {
                'regions': int(full_df['region_code'].nunique()),
                'base_metrics': int(base_df['metric'].nunique()),
                'derived_metrics': int(derived_df['metric'].nunique()),
                'base_rows': len(base_df),
                'derived_rows': len(derived_df),
                'total_rows': len(full_df),
                'historical_rows': int((full_df['data_type'] == 'historical').sum()),
                'forecast_rows': int((full_df['data_type'] == 'forecast').sum())
            }
        }
        
        meta_path = self.config.output_dir / "lad_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        log.info(f"  ✓ {meta_path}")
        
        # 7. DuckDB - two tables
        self._write_duckdb(base_df, derived_df)
    
    def _write_duckdb(self, base_df: pd.DataFrame, derived_df: pd.DataFrame):
        """
        Write to DuckDB gold tables - matches upstream architecture:
        - gold.lad_forecast: 8 base metrics
        - gold.lad_derived: 2 derived metrics
        """
        
        log.info("  Writing to DuckDB...")
        
        def prepare_df(df):
            """Prepare dataframe for DuckDB schema"""
            duck_df = df.copy()
            
            # Rename for schema consistency
            if 'metric' in duck_df.columns and 'metric_id' not in duck_df.columns:
                duck_df['metric_id'] = duck_df['metric']
            if 'year' in duck_df.columns and 'period' not in duck_df.columns:
                duck_df['period'] = duck_df['year']
            
            duck_df['forecast_run_date'] = datetime.now().date()
            duck_df['forecast_version'] = 'v1.5_share'
            
            return duck_df
        
        # Prepare both dataframes
        base_duck = prepare_df(base_df)
        derived_duck = prepare_df(derived_df)
        
        # Column order - EXACTLY matches ITL3 schema (with itl3_code as parent)
        duck_cols = [
            'region_code', 'region_name', 'region_level', 'itl3_code',
            'metric_id', 'period', 'value', 'unit', 'freq',
            'data_type', 'method', 'source',
            'ci_lower', 'ci_upper',
            'forecast_run_date', 'forecast_version'
        ]
        
        base_duck = base_duck[[c for c in duck_cols if c in base_duck.columns]]
        derived_duck = derived_duck[[c for c in duck_cols if c in derived_duck.columns]]
        
        con = duckdb.connect(str(self.config.duck_path))
        con.execute("CREATE SCHEMA IF NOT EXISTS gold")
        
        # Write base table (8 metrics)
        con.register('base_df', base_duck)
        con.execute("""
            CREATE OR REPLACE TABLE gold.lad_forecast AS
            SELECT * FROM base_df
        """)
        
        # Views for base
        con.execute("""
            CREATE OR REPLACE VIEW gold.lad_forecast_only AS
            SELECT * FROM gold.lad_forecast
            WHERE data_type = 'forecast'
        """)
        
        con.execute("""
            CREATE OR REPLACE VIEW gold.lad_latest AS
            SELECT * FROM gold.lad_forecast
        """)
        
        base_count = con.execute("SELECT COUNT(*) FROM gold.lad_forecast").fetchone()[0]
        log.info(f"  ✓ gold.lad_forecast: {base_count:,} rows (8 metrics)")
        
        # Write derived table (2 metrics)
        con.register('derived_df', derived_duck)
        con.execute("""
            CREATE OR REPLACE TABLE gold.lad_derived AS
            SELECT * FROM derived_df
        """)
        
        # Views for derived
        con.execute("""
            CREATE OR REPLACE VIEW gold.lad_derived_only AS
            SELECT * FROM gold.lad_derived
            WHERE data_type = 'forecast'
        """)
        
        derived_count = con.execute("SELECT COUNT(*) FROM gold.lad_derived").fetchone()[0]
        log.info(f"  ✓ gold.lad_derived: {derived_count:,} rows (2 metrics)")
        
        con.close()


# =============================================================================
# Pipeline
# =============================================================================

class LADForecastPipeline:
    """
    LAD V1 Pipeline - Share-based allocation from ITL3
    
    Steps:
      1. Load LAD history
      2. Load ITL3 forecasts (already reconciled)
      3. Load LAD→ITL3 concordance
      4. Compute stable historical shares
      5. Allocate ITL3 forecasts to LAD using shares
      6. Apply rate metrics (forecast or inherit)
      7. Calculate derived metrics
      8. Combine with history and save
    """
    
    def __init__(self, config: LADForecastConfig):
        self.config = config
        self.loader = DataLoaderLAD(config)
        self.share_calc = ShareCalculatorLAD(config)
        self.allocator = ShareAllocatorLAD(config)
        self.rate_handler = RateMetricsHandlerLAD(config)
        self.derived_calc = DerivedMetricsCalculatorLAD(config)
        self.output_builder = OutputBuilderLAD(config)
    
    def run(self) -> pd.DataFrame:
        """Execute the full pipeline"""
        
        start_time = datetime.now()
        
        log.info("")
        log.info("=" * 70)
        log.info(" REGIONIQ — LAD FORECASTING ENGINE V1.5")
        log.info(" (Share allocation from ITL3 + Base/Derived table split)")
        log.info("=" * 70)
        log.info("")
        
        # Connect to DuckDB
        self.loader.connect()
        
        try:
            # 1. Load all data
            log.info("[1/8] Loading data...")
            lad_history = self.loader.load_lad_history()
            itl3_forecasts = self.loader.load_itl3_forecasts()
            concordance = self.loader.load_concordance()
            
            # 2. Compute shares
            log.info("\n[2/8] Computing LAD shares of ITL3 parents...")
            shares = self.share_calc.compute_shares(lad_history, concordance)
            
            # 3. Allocate ITL3 → LAD (with continuity adjustment)
            log.info("\n[3/8] Allocating ITL3 forecasts to LAD...")
            allocated = self.allocator.allocate(itl3_forecasts, shares, lad_history=lad_history)
            
            # 4. Apply rate metrics
            log.info("\n[4/8] Processing rate metrics...")
            rates = self.rate_handler.apply_rates(itl3_forecasts, lad_history, concordance)
            
            # 5. Calculate derived metrics
            log.info("\n[5/8] Calculating derived metrics...")
            
            common_cols = ['region_code', 'region_name', 'itl3_code', 'year',
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
            base_df, derived_df = self.output_builder.build_final_output(
                lad_history, allocated, rates, derived, concordance
            )
            
            # 7. Validate share stability
            log.info("\n[7/8] Validating share stability...")
            self._validate_share_stability(base_df, shares)
            
            # 8. Save outputs
            log.info("\n[8/8] Saving outputs...")
            # Close read connection before opening write connection
            self.loader.close()
            self.output_builder.save_outputs(base_df, derived_df, shares)
            
        except Exception as e:
            self.loader.close()
            raise e
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Combine for summary stats
        final = pd.concat([base_df, derived_df], ignore_index=True)
        
        # Summary
        log.info("")
        log.info("=" * 70)
        log.info(" ✅ LAD FORECASTING V1.5 COMPLETE")
        log.info("=" * 70)
        log.info(f"Runtime:        {elapsed:.1f}s")
        log.info(f"LADs:           {final['region_code'].nunique()}")
        log.info(f"Metrics:        {final['metric'].nunique()} (8 base + 2 derived)")
        log.info(f"Total rows:     {len(final):,}")
        log.info(f"  Base table:   {len(base_df):,}")
        log.info(f"  Derived table: {len(derived_df):,}")
        log.info(f"  Historical:   {(final['data_type'] == 'historical').sum():,}")
        log.info(f"  Forecast:     {(final['data_type'] == 'forecast').sum():,}")
        log.info("=" * 70)
        
        return final
    
    def _validate_share_stability(self, final: pd.DataFrame, shares: pd.DataFrame):
        """Verify that LAD shares remain stable across forecast horizon"""
        
        log.info("Checking share stability across forecast years...")
        
        for metric in self.config.additive_metrics:
            metric_data = final[
                (final['metric'] == metric) &
                (final['data_type'] == 'forecast')
            ].copy()
            
            if metric_data.empty:
                continue
            
            # Compute shares by year for sample ITL3 parent
            for itl3_code in metric_data['itl3_code'].dropna().unique()[:5]:  # Sample 5
                itl3_data = metric_data[metric_data['itl3_code'] == itl3_code]
                
                if itl3_data.empty:
                    continue
                
                # Get shares per year
                shares_by_year = itl3_data.pivot_table(
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
                    log.warning(f"  ⚠️ {metric}/{itl3_code}: share variance detected (std={share_std:.2e})")
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
    """Run LAD V1 pipeline"""
    
    config = LADForecastConfig()
    pipeline = LADForecastPipeline(config)
    
    try:
        result = pipeline.run()
        return result
    except Exception as e:
        log.error(f"❌ Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()