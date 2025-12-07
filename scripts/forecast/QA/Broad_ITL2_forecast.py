#!/usr/bin/env python3
"""
Region IQ - ITL2 Regional Forecasting Engine V3.5
=============================================================================

V3.5: VAR/VECM VALIDATION - Production-safe thresholds + post-estimation checks
- Raised min_history_years: 5 → 12 (safer middle ground)
- Raised cointegration test: 8 → 15 years (prevent VECM instability)
- Added comprehensive VAR validation:
  * Reject NaN, negative, zero-collapse forecasts
  * Reject explosive growth (>25% annual)
  * Reject unstable confidence intervals (CV > 2.0)
- Falls back to ensemble univariate when VAR validation fails

Why V3.5 exists:
  V3.4's aggressive thresholds (5 years, 8 for VECM) allowed VAR to run but
  produced garbage with COVID-shocked short histories:
    - 14 regions → zero employment by 2040
    - 185 negative employment rates
    - 536 VECM failures
  V3.5 balances VAR usage with result quality.

Architecture:
  1. Load ITL2 history from silver layer (multi-file bottom-up)
  2. Generate ITL2 regional forecasts using VAR (+ univariate fallbacks)
  3. Reconcile ADDITIVE metrics (GVA, GDHI total, employment, population) to ITL1 parents
  4. Recalculate RATIO metrics from reconciled components (e.g. GDHI per head)
  5. Calculate other derived metrics (productivity, employment rate, income per worker)
  6. Output to CSV and DuckDB: gold.itl2_forecast

Inputs:
  - data/silver/itl2_population_history.csv
  - data/silver/itl2_employment_history.csv
  - data/silver/itl2_gva_history.csv
  - data/silver/itl2_gdhi_history.csv
  - gold.itl1_forecast (DuckDB, ITL1 anchors for reconciliation)
  
Outputs:
  - data/forecast/itl2_forecast_v3_long.csv
  - data/forecast/itl2_forecast_v3_wide.csv
  - data/forecast/itl2_confidence_intervals_v3.csv
  - data/forecast/itl2_forecast_quality_v3.csv
  - data/forecast/itl2_metadata_v3.json
  - gold.itl2_forecast (DuckDB table)
  - gold.itl2_forecast_only (DuckDB view)
  - gold.itl2_latest (DuckDB unified history+forecast view)

Expected performance:
  - VAR/VECM usage: 10–20% (down from 30–40% in V3.4, but all valid)
  - GVA–Employment correlation: +0.5 to +0.8 (realistic)
  - Zero invalid forecasts (no zero-employment regions or negative jobs)
  - Reconciliation: 100% success to ITL1 anchors

Author: Region IQ
Version: 3.5 (ITL2, VAR Validated)
License: Proprietary - Core IP
"""

import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import logging
import hashlib
import pickle
from typing import Dict, Tuple, Optional, List, Sequence
from dataclasses import dataclass, field

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('itl2_forecast_engine_v3_5.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# External packages
# ---------------------------------------------------------------------------

try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from scipy import stats
    from scipy.stats import jarque_bera
    HAVE_STATSMODELS = True
except ImportError as e:
    logger.warning(f"Statistical packages not fully available: {e}")
    HAVE_STATSMODELS = False

try:
    import duckdb
    HAVE_DUCKDB = True
except ImportError:
    HAVE_DUCKDB = False


# ===============================
# Configuration
# ===============================

@dataclass
class ForecastConfigITL2:
    """V3.5 configuration for ITL2 with VAR/VECM validation and safer thresholds."""
    
    # Data paths (ITL2)
    silver_dir: Path = Path("data/silver")
    duckdb_path: Path = Path("data/lake/warehouse.duckdb")
    use_duckdb: bool = False  # kept for compatibility, gating is via HAVE_DUCKDB
    
    # Output
    output_dir: Path = Path("data/forecast")
    cache_dir: Path = Path("data/cache")
    output_prefix: str = "itl2"
    
    # Forecast parameters
    target_year: int = 2050
    min_history_years: int = 12  # V3.5: safer middle ground
    confidence_levels: List[float] = field(default_factory=lambda: [0.80, 0.95])
    
    # Model parameters
    max_arima_order: int = 2
    max_var_lags: int = 3
    use_log_transform: Dict[str, bool] = None
    
    # VAR/VECM
    use_var_systems: bool = True
    var_systems: Dict[str, List[str]] = field(default_factory=lambda: {
        'gva_employment': ['nominal_gva_mn_gbp', 'emp_total_jobs']
    })
    var_bootstrap_samples: int = 300
    
    # Top-down reconciliation (to ITL1)
    use_macro_anchoring: bool = True
    reconciliation_method: str = 'proportional'
    
    # Macro anchor mapping: ITL2 metric → ITL1 metric_id (ADDITIVE ONLY)
    macro_anchor_map: Dict[str, str] = field(default_factory=lambda: {
        'nominal_gva_mn_gbp': 'nominal_gva_mn_gbp',
        'gdhi_total_mn_gbp': 'gdhi_total_mn_gbp',
        'emp_total_jobs': 'emp_total_jobs',
        'population_total': 'population_total'
    })
    
    # Metric definitions
    metric_definitions: Dict[str, Dict] = None
    structural_breaks: List[Dict] = None
    
    # Cross-validation
    cv_min_train_size: int = 15
    cv_test_windows: int = 3
    cv_horizon: int = 2
    
    # Performance
    n_bootstrap: int = 200
    cache_enabled: bool = True
    
    # Constraints
    enforce_non_negative: bool = True
    enforce_monotonic_population: bool = True
    growth_rate_cap_percentiles: Tuple[float, float] = (2, 98)
    
    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if self.metric_definitions is None:
            self.metric_definitions = {
                'population_total': {'unit': 'persons', 'transform': 'log', 'monotonic': True},
                'gdhi_total_mn_gbp': {'unit': 'GBP_m', 'transform': 'log', 'monotonic': False},
                'nominal_gva_mn_gbp': {'unit': 'GBP_m', 'transform': 'log', 'monotonic': False},
                'emp_total_jobs': {'unit': 'jobs', 'transform': 'log', 'monotonic': False}
            }
        
        if self.use_log_transform is None:
            self.use_log_transform = {
                metric: info['transform'] == 'log'
                for metric, info in self.metric_definitions.items()
            }
        
        if self.structural_breaks is None:
            self.structural_breaks = [
                {'year': 2008, 'name': 'Financial Crisis', 'type': 'level'},
                {'year': 2009, 'name': 'Crisis Recovery', 'type': 'trend'},
                {'year': 2016, 'name': 'Brexit Vote', 'type': 'trend'},
                {'year': 2020, 'name': 'COVID-19', 'type': 'level'},
                {'year': 2021, 'name': 'COVID Recovery', 'type': 'trend'}
            ]
        
        self._validate()
    
    def _validate(self):
        if not self.silver_dir.exists():
            raise FileNotFoundError(f"Silver dir not found: {self.silver_dir}")
        
        assert self.target_year > 2024
        logger.info(
            f"ITL2 V3.5 configuration validated "
            f"(min_history={self.min_history_years}, VAR validation enabled)"
        )


# ===============================
# ITL1 Anchor Manager
# ===============================

class ITL1AnchorManager:
    """Loads ITL1 forecasts (gold.itl1_forecast) for top-down reconciliation."""
    
    def __init__(self, duckdb_path: Path):
        self.duckdb_path = duckdb_path
        self.anchors = self._load_anchors()
    
    def _load_anchors(self) -> pd.DataFrame:
        """Load ITL1 forecasts from gold.itl1_forecast in DuckDB."""
        if not HAVE_DUCKDB or not self.duckdb_path.exists():
            logger.warning("Cannot load ITL1 anchors - DuckDB unavailable or missing.")
            return pd.DataFrame()
        
        try:
            con = duckdb.connect(str(self.duckdb_path), read_only=True)
            
            anchors = con.execute("""
                SELECT region_code, metric_id, period, value
                FROM gold.itl1_forecast
                WHERE data_type = 'forecast'
            """).fetchdf()
            
            con.close()
            
            if anchors.empty:
                logger.warning("No ITL1 forecasts in gold.itl1_forecast")
                return pd.DataFrame()
            
            # Standardise columns
            if 'period' in anchors.columns and 'year' not in anchors.columns:
                anchors['year'] = pd.to_numeric(anchors['period'], errors='coerce').astype(int)
            if 'metric_id' in anchors.columns and 'metric' not in anchors.columns:
                anchors['metric'] = anchors['metric_id']
            
            logger.info(
                f"✓ Loaded ITL1 anchors: {anchors['metric'].nunique()} metrics, "
                f"{anchors['region_code'].nunique()} regions, "
                f"years {int(anchors['year'].min())}-{int(anchors['year'].max())}"
            )
            
            return anchors
            
        except Exception as e:
            logger.error(f"Failed to load ITL1 anchors: {e}")
            return pd.DataFrame()
    
    def get_parent_value(
        self,
        metric: str,
        year: int,
        parent_region_code: str
    ) -> Optional[float]:
        """Get ITL1 value for a specific metric, year, parent ITL1 code."""
        if self.anchors.empty:
            return None
        
        match = self.anchors[
            (self.anchors['metric'] == metric) &
            (self.anchors['year'] == year) &
            (self.anchors['region_code'] == parent_region_code)
        ]
        
        if match.empty:
            return None
        
        return float(match['value'].iloc[0])
    
    def has_anchors(self) -> bool:
        """Check if anchors are available."""
        return not self.anchors.empty


# ===============================
# Top-Down Reconciler ITL2
# ===============================

class TopDownReconcilerITL2:
    """
    Reconciles ITL2 regional forecasts to ITL1 parent totals.

    Process:
      1. Build ITL2→ITL1 parent mapping from lookup file
      2. Reconcile ADDITIVE metrics (totals sum to ITL1 parent totals)
      3. Recalculate ratio metrics from reconciled components
    """
    
    def __init__(self, config: ForecastConfigITL2, anchor_manager: ITL1AnchorManager):
        self.config = config
        self.anchors = anchor_manager
        self.parent_mapping = self._build_parent_mapping()
    
    def _build_parent_mapping(self) -> pd.DataFrame:
        """Build ITL2→ITL1 parent mapping from geography reference."""
        lookup_path = Path("data/reference/master_2025_geography_lookup.csv")
        
        if not lookup_path.exists():
            logger.warning(f"Lookup file not found: {lookup_path}")
            return pd.DataFrame()
        
        try:
            lookup = pd.read_csv(lookup_path)
            lookup.columns = [col.replace('\ufeff', '') for col in lookup.columns]
            
            # Get ITL2→ITL1 mapping
            mapping = lookup[['ITL225CD', 'ITL125CD']].drop_duplicates()
            mapping.columns = ['itl2_code', 'itl1_code_tlc']
            
            # Map TLC codes to ONS ITL1 codes
            TLC_TO_ONS = {
                'TLC': 'E12000001', 'TLD': 'E12000002', 'TLE': 'E12000003',
                'TLF': 'E12000004', 'TLG': 'E12000005', 'TLH': 'E12000006',
                'TLI': 'E12000007', 'TLJ': 'E12000008', 'TLK': 'E12000009',
                'TLL': 'W92000004', 'TLM': 'S92000003', 'TLN': 'N92000002'
            }
            
            mapping['itl1_code'] = mapping['itl1_code_tlc'].map(TLC_TO_ONS)
            mapping = mapping.drop(columns=['itl1_code_tlc'])
            
            logger.info(
                f"✓ Built ITL2→ITL1 mapping: {len(mapping)} ITL2 regions → "
                f"{mapping['itl1_code'].nunique()} ITL1 parents"
            )
            
            return mapping
            
        except Exception as e:
            logger.error(f"Failed to build parent mapping: {e}")
            return pd.DataFrame()
    
    def reconcile(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        STEP 1: Apply proportional reconciliation to ADDITIVE metrics only
        STEP 2: Recalculate ratio metrics from reconciled components
        """
        if not self.anchors.has_anchors():
            logger.warning("No ITL1 anchors - skipping reconciliation")
            return data
        
        if self.parent_mapping.empty:
            logger.warning("No ITL2→ITL1 mapping - skipping reconciliation")
            return data
        
        # Merge parent ITL1 codes into forecast data
        data_with_parent = data.merge(
            self.parent_mapping,
            left_on='region_code',
            right_on='itl2_code',
            how='left'
        )
        
        if 'itl2_code' in data_with_parent.columns:
            data_with_parent = data_with_parent.drop(columns=['itl2_code'])
        
        # Coverage check
        missing_parents = data_with_parent[
            (data_with_parent['data_type'] == 'forecast') &
            (data_with_parent['itl1_code'].isna())
        ]
        if not missing_parents.empty:
            missing_regions = missing_parents['region_code'].unique()
            logger.warning(
                f"⚠️  {len(missing_regions)} ITL2 regions missing parent mapping: "
                f"{missing_regions[:5].tolist()}"
            )
        
        data = data_with_parent
        
        logger.info("\n" + "=" * 70)
        logger.info("TOP-DOWN RECONCILIATION ITL2 → ITL1")
        logger.info("=" * 70)
        
        reconciliation_log = []
        
        forecast_data = data[data['data_type'] == 'forecast']
        if forecast_data.empty:
            logger.info("No forecast data to reconcile.")
            return data
        
        forecast_years = sorted(forecast_data['year'].unique())
        
        # STEP 1: Reconcile additive metrics to ITL1 anchors
        for metric in forecast_data['metric'].unique():
            itl1_metric = self.config.macro_anchor_map.get(metric)
            if not itl1_metric:
                logger.debug(f"  {metric}: No ITL1 anchor mapping (skipping)")
                continue
            
            logger.info(f"\n  Reconciling {metric} → ITL1 {itl1_metric}")
            
            for parent_region in forecast_data['itl1_code'].dropna().unique():
                for year in forecast_years:
                    year_int = int(year)
                    
                    itl1_value = self.anchors.get_parent_value(
                        itl1_metric, year_int, parent_region
                    )
                    if itl1_value is None:
                        continue
                    
                    mask = (
                        (data['year'] == year_int) &
                        (data['metric'] == metric) &
                        (data['data_type'] == 'forecast') &
                        (data['itl1_code'] == parent_region)
                    )
                    
                    if not mask.any():
                        continue
                    
                    regional_sum_before = data.loc[mask, 'value'].sum()
                    
                    if regional_sum_before <= 0:
                        logger.warning(
                            f"    {parent_region} {year}: Regional sum ≤ 0, skipping"
                        )
                        continue
                    
                    scale_factor = itl1_value / regional_sum_before
                    
                    data.loc[mask, 'value'] *= scale_factor
                    if 'ci_lower' in data.columns:
                        data.loc[mask, 'ci_lower'] *= scale_factor
                    if 'ci_upper' in data.columns:
                        data.loc[mask, 'ci_upper'] *= scale_factor
                    
                    regional_sum_after = data.loc[mask, 'value'].sum()
                    deviation = (
                        abs(regional_sum_after - itl1_value) / itl1_value
                        if itl1_value > 0 else 0
                    )
                    
                    reconciliation_log.append({
                        'year': year_int,
                        'metric': metric,
                        'parent_region_code': parent_region,
                        'itl1_metric': itl1_metric,
                        'itl1_value': itl1_value,
                        'regional_sum_before': regional_sum_before,
                        'regional_sum_after': regional_sum_after,
                        'scale_factor': scale_factor,
                        'deviation_pct': deviation * 100
                    })
                    
                    if year_int in [2025, 2030, 2040, 2050]:
                        logger.info(
                            f"    {parent_region} {year}: "
                            f"SF={scale_factor:.4f} | ITL1={itl1_value:,.0f} | "
                            f"ITL2 sum: {regional_sum_before:,.0f}→{regional_sum_after:,.0f}"
                        )
        
        data.attrs['reconciliation_log'] = reconciliation_log
        logger.info(f"\n✓ Additive reconciliation complete: {len(reconciliation_log)} adjustments")
        
        # STEP 2: Ratio metrics (currently GDHI per head is derived post-forecast)
        logger.info("\n✓ Ratio metrics handled (GDHI per head derived in derived-metrics step)")
        
        return data


# ===============================
# Data Management (ITL2)
# ===============================

class DataManagerITL2:
    """Data management for ITL2 unified silver schema (multi-file)."""
    
    def __init__(self, config: ForecastConfigITL2):
        self.config = config
        
    def load_all_data(self) -> pd.DataFrame:
        """Load ITL2 historical data, apply standardisation and outlier handling."""
        cache_key = self._get_cache_key()
        if self.config.cache_enabled and self._cache_exists(cache_key):
            logger.info("Loading ITL2 data from cache")
            return self._load_from_cache(cache_key)
        
        unified = self._load_from_csv_files()
        unified = self._standardize_columns(unified)
        unified = self._handle_outliers(unified)
        
        if self.config.cache_enabled:
            self._save_to_cache(unified, cache_key)
        
        logger.info(f"Loaded ITL2 data: {len(unified)} rows")
        logger.info(
            f"ITL2 regions: {unified['region_code'].nunique()}, "
            f"metrics: {unified['metric'].nunique()}"
        )
        
        return unified
    
    def _load_from_csv_files(self) -> pd.DataFrame:
        """Load all itl2_*_history.csv files from silver layer."""
        files_to_load = [
            'itl2_population_history.csv',
            'itl2_employment_history.csv',
            'itl2_gva_history.csv',
            'itl2_gdhi_history.csv'
        ]
        
        dfs = []
        for filename in files_to_load:
            filepath = self.config.silver_dir / filename
            
            if filepath.exists():
                logger.info(f"Loading: {filename}")
                df = pd.read_csv(filepath)
                dfs.append(df)
            else:
                logger.warning(f"File not found: {filename}")
        
        if not dfs:
            raise FileNotFoundError(
                f"No ITL2 history files found in {self.config.silver_dir}"
            )
        
        df_all = pd.concat(dfs, ignore_index=True)
        return df_all
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardise column names for forecasting engine compatibility."""
        rename_map = {
            'metric_id': 'metric',
            'period': 'year',
            'region_name': 'region'
        }
        
        for old, new in rename_map.items():
            if old in df.columns and new not in df.columns:
                df[new] = df[old]
        
        df['year'] = pd.to_numeric(df['year'], errors='coerce').astype(int)
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        if 'data_type' not in df.columns:
            df['data_type'] = 'historical'
        
        df = df.dropna(subset=['year', 'value'])
        
        if self.config.enforce_non_negative:
            df = df[df['value'] > 0]
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """IQR-based winsorisation of outliers by (region, metric)."""
        for (region, metric), group in df.groupby(['region_code', 'metric']):
            if len(group) < 5:
                continue
            
            values = group['value'].values
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            
            outliers = (values < lower_bound) | (values > upper_bound)
            if outliers.any():
                logger.info(
                    f"Winsorizing {outliers.sum()} outliers in ITL2 {region}-{metric}"
                )
                df.loc[group[outliers].index, 'value'] = np.clip(
                    values[outliers], lower_bound, upper_bound
                )
        
        return df
    
    def _get_cache_key(self) -> str:
        hasher = hashlib.md5()
        hasher.update(b"itl2_v3.5")  # V3.5-specific cache key
        return hasher.hexdigest()
    
    def _cache_exists(self, key: str) -> bool:
        return (self.config.cache_dir / f"itl2_data_{key}.pkl").exists()
    
    def _load_from_cache(self, key: str) -> pd.DataFrame:
        with open(self.config.cache_dir / f"itl2_data_{key}.pkl", 'rb') as f:
            return pickle.load(f)
    
    def _save_to_cache(self, df: pd.DataFrame, key: str):
        with open(self.config.cache_dir / f"itl2_data_{key}.pkl", 'wb') as f:
            pickle.dump(df, f)


# ===============================
# VAR System Forecaster (V3.5)
# ===============================

class VARSystemForecaster:
    """VAR/VECM forecaster for cross-metric coherence with V3.5 validation."""
    
    def __init__(self, config: ForecastConfigITL2):
        self.config = config
    
    def forecast_system(
        self,
        data: pd.DataFrame,
        region_code: str,
        metrics: List[str],
        horizon: int,
        structural_breaks: Optional[Sequence[Dict]] = None
    ) -> Optional[Dict]:
        """Forecast multiple metrics jointly using VAR/VECM and validate outputs."""
        if not HAVE_STATSMODELS:
            return None
        
        try:
            # Build aligned system dataframe
            series_dict = {}
            for metric in metrics:
                metric_data = data[
                    (data['region_code'] == region_code) &
                    (data['metric'] == metric)
                ].sort_values('year')
                
                if metric_data.empty:
                    return None
                
                values = metric_data['value'].values
                if self.config.use_log_transform.get(metric, False) and (values > 0).all():
                    values = np.log(values)
                
                series_dict[metric] = pd.Series(
                    values, index=metric_data['year'].values.astype(int)
                )
            
            system_df = pd.DataFrame(series_dict).dropna()
            
            # Require sufficient history (V3.5: 12 years by config)
            if len(system_df) < self.config.min_history_years:
                logger.debug(
                    f"  VAR skipped for {region_code}: only {len(system_df)} years "
                    f"(need {self.config.min_history_years})"
                )
                return None
            
            # Cointegration test with safer V3.5 threshold
            coint_result = self._test_cointegration(system_df)
            exog = self._prep_breaks(system_df.index, structural_breaks) if structural_breaks else None
            
            if coint_result['cointegrated'] and coint_result['rank'] > 0:
                model = VECM(
                    system_df,
                    k_ar_diff=min(self.config.max_var_lags, 3),
                    coint_rank=coint_result['rank'],
                    deterministic='ci',
                    exog=exog
                )
                fitted = model.fit()
                method_name = f'VECM(r={coint_result["rank"]})'
                exog_fc = self._extend_exog(exog, system_df.index, horizon) if exog is not None else None
                forecast_values = fitted.predict(steps=horizon, exog_fc=exog_fc)
            else:
                model = VAR(system_df, exog=exog)
                lag_results = model.select_order(maxlags=self.config.max_var_lags)
                try:
                    optimal_lags = lag_results.selected_orders['aic']
                except (AttributeError, KeyError):
                    optimal_lags = lag_results.aic
                optimal_lags = max(1, min(optimal_lags, self.config.max_var_lags))
                
                fitted = model.fit(maxlags=optimal_lags)
                method_name = f'VAR({optimal_lags})'
                exog_fc = self._extend_exog(exog, system_df.index, horizon) if exog is not None else None
                forecast_values = fitted.forecast(
                    system_df.values[-fitted.k_ar:], steps=horizon, exog_future=exog_fc
                )
            
            ci_lower, ci_upper = self._bootstrap_ci(
                fitted, system_df, horizon, exog, exog_fc
            )
            
            last_year = int(system_df.index[-1])
            forecast_years = list(range(last_year + 1, last_year + horizon + 1))
            
            results = {}
            for i, metric in enumerate(metrics):
                vals = forecast_values[:, i]
                ci_l, ci_u = ci_lower[:, i], ci_upper[:, i]
                
                if self.config.use_log_transform.get(metric, False):
                    vals, ci_l, ci_u = np.exp(vals), np.exp(ci_l), np.exp(ci_u)
                
                results[metric] = {
                    'method': method_name,
                    'values': vals,
                    'years': forecast_years,
                    'ci_lower': ci_l,
                    'ci_upper': ci_u,
                    'aic': fitted.aic if hasattr(fitted, 'aic') else None,
                    'system_metrics': metrics,
                    'cointegrated': coint_result['cointegrated']
                }
            
            # ------------------------------------------------------------------
            # V3.5: VAR validation – reject bad systems and fall back to univariate
            # ------------------------------------------------------------------
            validation_failed = False
            
            for metric, fc in results.items():
                # 1) NaNs
                if np.isnan(fc['values']).any():
                    logger.warning(f"VAR produced NaN for {region_code}-{metric}, rejecting")
                    validation_failed = True
                    break
                
                # 2) Negatives for strictly positive metrics
                if metric in ['emp_total_jobs', 'nominal_gva_mn_gbp', 'population_total', 'gdhi_total_mn_gbp']:
                    if (fc['values'] < 0).any():
                        logger.warning(f"VAR produced negative values for {region_code}-{metric}, rejecting")
                        validation_failed = True
                        break
                
                # 3) Zero collapse in long-term tails
                if metric in ['emp_total_jobs', 'nominal_gva_mn_gbp']:
                    long_term = fc['values'][-min(10, len(fc['values'])):]
                    if (long_term < 1e-6).any():
                        logger.warning(f"VAR produced near-zero collapse for {region_code}-{metric}, rejecting")
                        validation_failed = True
                        break
                
                # 4) Explosive growth
                pct_changes = np.diff(fc['values']) / np.clip(fc['values'][:-1], 1e-10, None)
                if np.any(pct_changes > 0.25):
                    logger.warning(f"VAR produced explosive growth for {region_code}-{metric}, rejecting")
                    validation_failed = True
                    break
                
                # 5) Unstable confidence intervals
                ci_width = fc['ci_upper'] - fc['ci_lower']
                cv = ci_width / (2 * np.abs(fc['values']) + 1e-10)
                if np.any(cv > 2.0):
                    logger.warning(f"VAR produced unstable CIs for {region_code}-{metric}, rejecting")
                    validation_failed = True
                    break
            
            if validation_failed:
                logger.info(f"  VAR validation failed for {region_code}, falling back to univariate ensemble")
                return None
            
            return results
        
        except Exception as e:
            logger.error(f"VAR system failed (ITL2 {region_code}): {e}")
            return None
    
    def _test_cointegration(self, data: pd.DataFrame) -> Dict:
        """V3.5: raise threshold to 15 years for VECM reliability."""
        if len(data) < 15:
            return {'cointegrated': False, 'rank': 0}
        try:
            result = coint_johansen(data, det_order=0, k_ar_diff=2)
            rank = int(np.sum(result.lr1 > result.cvt[:, 1]))
            return {'cointegrated': rank > 0, 'rank': rank}
        except Exception:
            return {'cointegrated': False, 'rank': 0}
    
    def _bootstrap_ci(self, fitted, data, horizon, exog, exog_fc, n_boot=300):
        """Bootstrap-based confidence intervals for VAR/VECM."""
        n_vars = data.shape[1]
        boot_fcs = []
        
        residuals = fitted.resid if hasattr(fitted, 'resid') else data.values[fitted.k_ar:] - fitted.fittedvalues
        
        for _ in range(n_boot):
            try:
                boot_resid = residuals[np.random.choice(len(residuals), len(residuals), replace=True)]
                boot_df = pd.DataFrame(
                    fitted.fittedvalues + boot_resid,
                    columns=data.columns,
                    index=data.index[-len(fitted.fittedvalues):]
                )
                
                if hasattr(fitted, 'coint_rank'):
                    boot_model = VECM(
                        boot_df,
                        k_ar_diff=fitted.k_ar_diff,
                        coint_rank=fitted.coint_rank,
                        deterministic='ci',
                        exog=exog
                    ).fit()
                    boot_fc = boot_model.predict(steps=horizon, exog_fc=exog_fc)
                else:
                    boot_model = VAR(boot_df, exog=exog).fit(maxlags=fitted.k_ar)
                    boot_fc = boot_model.forecast(
                        boot_df.values[-fitted.k_ar:], steps=horizon, exog_future=exog_fc
                    )
                
                boot_fcs.append(boot_fc)
            except Exception:
                continue
        
        if len(boot_fcs) < 50:
            resid_std = np.std(residuals, axis=0)
            fc_std = resid_std * np.sqrt(np.arange(1, horizon + 1))[:, None]
            return -fc_std * 1.96, fc_std * 1.96
        
        boot_array = np.array(boot_fcs)
        return np.percentile(boot_array, 2.5, axis=0), np.percentile(boot_array, 97.5, axis=0)
    
    def _prep_breaks(self, index, breaks):
        """Prepare exogenous structural break dummies."""
        if not breaks:
            return None
        dummies = []
        for b in breaks:
            if b.get('type') == 'level':
                dummies.append((index >= b['year']).astype(int))
            else:
                dummies.append(np.maximum(0, index - b['year']))
        return np.column_stack(dummies) if dummies else None
    
    def _extend_exog(self, exog, index, horizon):
        """Extend exogenous variables for forecast horizon."""
        if exog is None:
            return None
        if len(exog.shape) == 1:
            exog = exog.reshape(-1, 1)
        
        extended = []
        for _ in range(1, horizon + 1):
            row = []
            for i in range(exog.shape[1]):
                if np.all(np.isin(exog[:, i], [0, 1])):
                    row.append(exog[-1, i])
                else:
                    row.append(
                        exog[-1, i] + (exog[-1, i] - exog[-2, i] if len(exog) > 1 else 1)
                    )
            extended.append(row)
        return np.array(extended)


# ===============================
# Advanced Forecasting (shared)
# ===============================

class AdvancedForecastingV3:
    """Shared univariate/ensemble forecasting engine for ITL levels."""
    
    def __init__(self, config: ForecastConfigITL2):
        self.config = config
        self.var_forecaster = VARSystemForecaster(config) if config.use_var_systems else None
        
    # --- Univariate orchestration -------------------------------------------------
    
    def forecast_univariate(
        self,
        series: pd.Series,
        horizon: int,
        structural_breaks: Optional[Sequence[Dict]] = None,
        metric_info: Dict = None
    ) -> Dict:
        """Univariate + ensemble forecasting with CV weighting and constraints."""
        models = []
        
        arima_result = self._fit_arima_with_breaks(series, horizon, structural_breaks)
        if arima_result:
            models.append(arima_result)
        
        if HAVE_STATSMODELS and len(series) > 20:
            ets_result = self._fit_ets_auto(series, horizon)
            if ets_result:
                models.append(ets_result)
        
        linear_result = self._fit_linear(series, horizon)
        if linear_result:
            models.append(linear_result)
        
        if not models:
            combined = self._fallback_forecast(series, horizon)
        elif len(models) == 1:
            combined = models[0]
        else:
            cv_errors = self._true_cross_validation(series, models)
            weights = self._calculate_cv_weights(cv_errors)
            combined = self._combine_forecasts_properly(models, weights, series)
        
        if metric_info:
            combined = self._apply_forecast_constraints(combined, series, metric_info)
        
        return combined
    
    # --- ARIMA with structural breaks --------------------------------------------
    
    def _fit_arima_with_breaks(
        self,
        series: pd.Series,
        horizon: int,
        structural_breaks: Optional[Sequence[Dict]] = None
    ) -> Optional[Dict]:
        """ARIMA with structural break dummies and AICc model selection."""
        if not HAVE_STATSMODELS:
            return None
        
        try:
            exog = self._prepare_break_dummies(series, structural_breaks) if structural_breaks else None
            
            best_model = None
            best_aic = np.inf
            best_order = None
            
            adf_p = adfuller(series, autolag='AIC')[1]
            d = 1 if adf_p > 0.05 else 0
            
            for p in range(self.config.max_arima_order + 1):
                for q in range(self.config.max_arima_order + 1):
                    if p == 0 and q == 0 and d == 0:
                        continue
                    
                    try:
                        model = ARIMA(series, order=(p, d, q), exog=exog)
                        fitted = model.fit(method_kwargs={"warn_convergence": False})
                        
                        n = len(series)
                        k = p + q + d + 1
                        aicc = fitted.aic + (2 * k * (k + 1)) / (n - k - 1)
                        
                        if aicc < best_aic:
                            best_aic = aicc
                            best_model = fitted
                            best_order = (p, d, q)
                    except Exception:
                        continue
            
            if best_model is None:
                return None
            
            exog_forecast = (
                self._extend_exog_properly(exog, series.index, horizon)
                if exog is not None else None
            )
            
            forecast_obj = best_model.get_forecast(steps=horizon, exog=exog_forecast)
            forecast = forecast_obj.predicted_mean
            
            residuals = pd.Series(best_model.resid)
            resid_tests = self._test_residuals(residuals)
            
            # Bootstrap CI if residuals non-normal; otherwise use analytic CI
            if not resid_tests.get('normal', True):
                ci = self._bootstrap_confidence_intervals_fixed(
                    best_model, series, horizon, exog, exog_forecast, best_order
                )
            else:
                ci = forecast_obj.conf_int(alpha=0.05)
                if self.config.use_log_transform.get(series.name, False):
                    sigma2 = np.var(residuals)
                    forecast = forecast * np.exp(0.5 * sigma2)
                    ci = ci * np.exp(0.5 * sigma2)
            
            last_year = int(series.index[-1])
            forecast_years = list(range(last_year + 1, last_year + horizon + 1))
            
            return {
                'method': f'ARIMA{best_order}',
                'values': forecast.values,
                'years': forecast_years,
                'ci_lower': ci.iloc[:, 0].values if hasattr(ci, 'iloc') else ci[:, 0],
                'ci_upper': ci.iloc[:, 1].values if hasattr(ci, 'iloc') else ci[:, 1],
                'aic': best_aic,
                'order': best_order,
                'residual_tests': resid_tests,
                'fit_function': lambda s, h: self._arima_predict(s, h, best_order, structural_breaks)
            }
            
        except Exception as e:
            logger.error(f"ARIMA failed: {e}")
            return None
    
    # --- ETS ----------------------------------------------------------------------
    
    def _fit_ets_auto(self, series: pd.Series, horizon: int) -> Optional[Dict]:
        """ETS with simple auto trend/damping selection."""
        if not HAVE_STATSMODELS:
            return None
        
        try:
            best_model = None
            best_aic = np.inf
            best_config = None
            
            for trend in ['add', 'mul', None]:
                for damped in [True, False]:
                    if trend is None and damped:
                        continue
                    
                    try:
                        model = ExponentialSmoothing(
                            series, trend=trend, damped_trend=damped, seasonal=None
                        )
                        fitted = model.fit()
                        
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_model = fitted
                            best_config = f"ETS({trend},{damped})"
                    except Exception:
                        continue
            
            if best_model is None:
                return None
            
            forecast = best_model.forecast(steps=horizon)
            residuals = series - best_model.fittedvalues
            sigma = residuals.std()
            
            t_stat = stats.t.ppf(0.975, len(series) - 1) if len(series) < 30 else 1.96
            ci_lower = forecast - t_stat * sigma * np.sqrt(np.arange(1, horizon + 1))
            ci_upper = forecast + t_stat * sigma * np.sqrt(np.arange(1, horizon + 1))
            
            last_year = int(series.index[-1])
            forecast_years = list(range(last_year + 1, last_year + horizon + 1))
            
            return {
                'method': best_config,
                'values': forecast.values,
                'years': forecast_years,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'aic': best_aic,
                'fit_function': lambda s, h: best_model.forecast(steps=h)
            }
            
        except Exception as e:
            logger.error(f"ETS failed: {e}")
            return None
    
    # --- Linear trend ------------------------------------------------------------
    
    def _fit_linear(self, series: pd.Series, horizon: int) -> Optional[Dict]:
        """Simple linear trend model."""
        try:
            X = np.arange(len(series)).reshape(-1, 1)
            y = series.values
            
            if HAVE_STATSMODELS:
                X_sm = sm.add_constant(X)
                model = sm.OLS(y, X_sm).fit()
                
                X_future = np.arange(len(series), len(series) + horizon).reshape(-1, 1)
                X_future_sm = sm.add_constant(X_future)
                predictions = model.get_prediction(X_future_sm)
                forecast = predictions.predicted_mean
                ci = predictions.conf_int(alpha=0.05)
            else:
                coeffs = np.polyfit(X.flatten(), y, 1)
                X_future = np.arange(len(series), len(series) + horizon)
                forecast = np.polyval(coeffs, X_future)
                
                residuals = y - np.polyval(coeffs, X.flatten())
                sigma = residuals.std()
                ci_width = 1.96 * sigma * np.sqrt(
                    1 + 1/len(series) + (X_future - X.mean())**2 / X.var()
                )
                ci = np.column_stack([forecast - ci_width, forecast + ci_width])
            
            last_year = int(series.index[-1])
            forecast_years = list(range(last_year + 1, last_year + horizon + 1))
            
            return {
                'method': 'Linear',
                'values': forecast,
                'years': forecast_years,
                'ci_lower': ci[:, 0],
                'ci_upper': ci[:, 1],
                'aic': model.aic if HAVE_STATSMODELS else None,
                'fit_function': lambda s, h: self._linear_predict(s, h)
            }
            
        except Exception as e:
            logger.error(f"Linear forecast failed: {e}")
            return None
    
    # --- Cross-validation + ensemble --------------------------------------------
    
    def _true_cross_validation(self, series: pd.Series, models: List[Dict]) -> Dict:
        """Expanding-window CV to estimate each model's relative error."""
        cv_errors = {i: [] for i in range(len(models))}
        
        min_train = max(self.config.cv_min_train_size, len(series) // 2)
        
        for train_end in range(min_train, len(series) - self.config.cv_horizon + 1):
            train = series.iloc[:train_end]
            test = series.iloc[train_end:min(train_end + self.config.cv_horizon, len(series))]
            
            for i, model_info in enumerate(models):
                try:
                    if 'fit_function' in model_info:
                        forecast = model_info['fit_function'](train, len(test))
                        error = np.mean(
                            np.abs(forecast - test.values)
                            / (np.abs(forecast) + np.abs(test.values) + 1e-10)
                        )
                        cv_errors[i].append(error)
                except Exception:
                    cv_errors[i].append(np.inf)
        
        return cv_errors
    
    def _calculate_cv_weights(self, cv_errors: Dict) -> np.ndarray:
        """Calculate ensemble weights from CV errors."""
        mean_errors = []
        for i in sorted(cv_errors.keys()):
            errors = [e for e in cv_errors[i] if e != np.inf]
            if errors:
                mean_errors.append(np.mean(errors))
            else:
                mean_errors.append(np.inf)
        
        if all(e == np.inf for e in mean_errors):
            return np.ones(len(mean_errors)) / len(mean_errors)
        
        weights = 1.0 / (np.array(mean_errors) + 1e-10)
        weights[np.isinf(weights)] = 0
        
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(weights)) / len(weights)
        
        return weights
    
    def _combine_forecasts_properly(
        self,
        models: List[Dict],
        weights: np.ndarray,
        series: pd.Series
    ) -> Dict:
        """Combine forecasts with CV-derived weights and ensemble CIs."""
        combined_values = np.zeros(len(models[0]['values']))
        for model, weight in zip(models, weights):
            combined_values += weight * model['values']
        
        cv_residuals = []
        
        min_train = max(self.config.cv_min_train_size, len(series) // 2)
        for train_end in range(min_train, len(series)):
            train = series.iloc[:train_end]
            test_val = series.iloc[train_end] if train_end < len(series) else None
            
            if test_val is not None:
                ensemble_forecast = 0
                for model, weight in zip(models, weights):
                    if 'fit_function' in model:
                        try:
                            fc = model['fit_function'](train, 1)[0]
                            ensemble_forecast += weight * fc
                        except Exception:
                            pass
                
                if ensemble_forecast > 0:
                    cv_residuals.append(test_val - ensemble_forecast)
        
        if cv_residuals:
            residual_std = np.std(cv_residuals)
            horizon_adjustment = np.sqrt(np.arange(1, len(combined_values) + 1))
            ci_lower = combined_values - 1.96 * residual_std * horizon_adjustment
            ci_upper = combined_values + 1.96 * residual_std * horizon_adjustment
        else:
            ci_lower = np.zeros(len(models[0]['values']))
            ci_upper = np.zeros(len(models[0]['values']))
            for model, weight in zip(models, weights):
                ci_lower += weight * model['ci_lower']
                ci_upper += weight * model['ci_upper']
        
        return {
            'method': f"Ensemble({len(models)})",
            'values': combined_values,
            'years': models[0]['years'],
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'weights': weights.tolist(),
            'component_methods': [m['method'] for m in models],
            'cv_residual_std': np.std(cv_residuals) if cv_residuals else None
        }
    
    # --- Bootstrap CI for ARIMA ---------------------------------------------------
    
    def _bootstrap_confidence_intervals_fixed(
        self,
        model,
        series: pd.Series,
        horizon: int,
        exog: Optional[np.ndarray],
        exog_forecast: Optional[np.ndarray],
        order: Tuple[int, int, int],
        n_bootstrap: int = None
    ) -> np.ndarray:
        """Bootstrap CI for ARIMA with structural breaks."""
        if n_bootstrap is None:
            n_bootstrap = self.config.n_bootstrap
        
        residuals = model.resid
        fitted = model.fittedvalues
        
        bootstrap_forecasts = []
        
        for _ in range(n_bootstrap):
            boot_residuals = np.random.choice(residuals, size=len(residuals), replace=True)
            boot_series = pd.Series(fitted + boot_residuals, index=series.index)
            
            try:
                boot_model = ARIMA(boot_series, order=order, exog=exog)
                boot_fitted = boot_model.fit(disp=False)
                boot_forecast = boot_fitted.forecast(steps=horizon, exog=exog_forecast)
                bootstrap_forecasts.append(boot_forecast.values)
            except Exception:
                continue
        
        if len(bootstrap_forecasts) < 10:
            return model.get_forecast(steps=horizon, exog=exog_forecast).conf_int(alpha=0.05)
        
        bootstrap_array = np.array(bootstrap_forecasts)
        ci_lower = np.percentile(bootstrap_array, 2.5, axis=0)
        ci_upper = np.percentile(bootstrap_array, 97.5, axis=0)
        
        return np.column_stack([ci_lower, ci_upper])
    
    # --- Structural breaks --------------------------------------------------------
    
    def _prepare_break_dummies(
        self,
        series: pd.Series,
        breaks: Optional[Sequence[Dict]]
    ) -> Optional[np.ndarray]:
        """Prepare structural break dummy variables."""
        if not breaks:
            return None
        
        dummies = []
        years = series.index
        
        for break_info in breaks:
            if 'year' in break_info:
                break_year = break_info['year']
                break_type = break_info.get('type', 'level')
                
                if break_year in years or any(y > break_year for y in years):
                    if break_type == 'level':
                        dummy = (years >= break_year).astype(int)
                        dummies.append(dummy)
                    elif break_type == 'trend':
                        time_since_break = np.maximum(0, years - break_year)
                        dummies.append(time_since_break)
        
        return np.column_stack(dummies) if dummies else None
    
    def _extend_exog_properly(
        self,
        exog: np.ndarray,
        historical_index: pd.Index,
        horizon: int
    ) -> np.ndarray:
        """Extend exogenous variables consistently over forecast horizon."""
        if exog is None:
            return None
        
        if len(exog.shape) == 1:
            exog = exog.reshape(-1, 1)
        
        n_vars = exog.shape[1]
        last_year = int(historical_index[-1])
        future_years = range(last_year + 1, last_year + horizon + 1)
        
        extended = []
        for year in future_years:
            row = []
            for i in range(n_vars):
                if np.all(np.isin(exog[:, i], [0, 1])):
                    row.append(exog[-1, i])
                else:
                    increment = exog[-1, i] - exog[-2, i] if len(exog) > 1 else 1
                    row.append(exog[-1, i] + increment * (year - last_year))
            extended.append(row)
        
        result = np.array(extended)
        assert result.shape == (horizon, n_vars)
        
        return result
    
    # --- Constraints --------------------------------------------------------------
    
    def _apply_forecast_constraints(
        self,
        forecast: Dict,
        historical: pd.Series,
        metric_info: Dict
    ) -> Dict:
        """Apply non-negativity, monotonicity, and growth caps."""
        values = forecast['values'].copy()
        
        # Non-negativity
        if self.config.enforce_non_negative:
            values = np.maximum(values, 0)
            forecast['ci_lower'] = np.maximum(forecast['ci_lower'], 0)
        
        # Monotonic population
        if metric_info.get('monotonic', False) and self.config.enforce_monotonic_population:
            for i in range(1, len(values)):
                values[i] = max(values[i], values[i-1])
        
        # Growth rate caps
        if self.config.growth_rate_cap_percentiles:
            historical_growth = historical.pct_change().dropna()
            if len(historical_growth) > 5:
                p_low, p_high = self.config.growth_rate_cap_percentiles
                growth_bounds = np.percentile(historical_growth, [p_low, p_high])
                
                last_value = historical.iloc[-1]
                for i in range(len(values)):
                    prev_value = last_value if i == 0 else values[i-1]
                    growth = (
                        (values[i] - prev_value) / prev_value
                        if prev_value > 0 else 0
                    )
                    
                    if growth < growth_bounds[0]:
                        values[i] = prev_value * (1 + growth_bounds[0])
                    elif growth > growth_bounds[1]:
                        values[i] = prev_value * (1 + growth_bounds[1])
        
        forecast['values'] = values
        forecast['constraints_applied'] = True
        
        return forecast
    
    # --- Residual diagnostics -----------------------------------------------------
    
    def _test_residuals(self, residuals: pd.Series) -> Dict:
        """Basic residual diagnostics (autocorr, normality)."""
        results = {}
        
        if not HAVE_STATSMODELS or len(residuals) < 10:
            return results
        
        clean_resid = residuals.dropna()
        
        try:
            lb = acorr_ljungbox(clean_resid, lags=min(10, len(clean_resid)//3))
            results['ljung_box_p'] = float(lb['lb_pvalue'].min())
            results['no_autocorr'] = results['ljung_box_p'] > 0.05
        except Exception:
            pass
        
        try:
            jb_stat, jb_p = jarque_bera(clean_resid)
            results['jarque_bera_p'] = jb_p
            results['normal'] = jb_p > 0.05
        except Exception:
            pass
        
        return results
    
    # --- Simple helpers -----------------------------------------------------------
    
    def _arima_predict(
        self,
        series: pd.Series,
        horizon: int,
        order: Tuple,
        breaks: Optional[Sequence[Dict]] = None
    ) -> np.ndarray:
        """Helper for CV: refit ARIMA and forecast horizon."""
        try:
            exog = self._prepare_break_dummies(series, breaks) if breaks else None
            model = ARIMA(series, order=order, exog=exog)
            fitted = model.fit(disp=False)
            
            if exog is not None:
                exog_fc = self._extend_exog_properly(exog, series.index, horizon)
            else:
                exog_fc = None
            
            return fitted.forecast(steps=horizon, exog=exog_fc).values
        except Exception:
            return np.repeat(series.iloc[-1], horizon)
    
    def _linear_predict(self, series: pd.Series, horizon: int) -> np.ndarray:
        """Helper for CV: linear trend prediction."""
        try:
            X = np.arange(len(series))
            y = series.values
            coeffs = np.polyfit(X, y, 1)
            X_future = np.arange(len(series), len(series) + horizon)
            return np.polyval(coeffs, X_future)
        except Exception:
            return np.repeat(series.iloc[-1], horizon)
    
    def _fallback_forecast(self, series: pd.Series, horizon: int) -> Dict:
        """Robust fallback forecast (flat + gentle trend)."""
        last_value = series.iloc[-1]
        
        if len(series) > 5:
            recent_trend = (series.iloc[-1] - series.iloc[-5]) / 5
        else:
            recent_trend = 0
        
        values = []
        for h in range(1, horizon + 1):
            value = last_value + recent_trend * h
            values.append(max(value, 0))
        
        if len(series) > 2:
            historical_std = series.std()
            ci_width = 1.96 * historical_std * np.sqrt(np.arange(1, horizon + 1))
        else:
            ci_width = np.array(values) * 0.1
        
        last_year = int(series.index[-1])
        forecast_years = list(range(last_year + 1, last_year + horizon + 1))
        
        return {
            'method': 'Fallback',
            'values': np.array(values),
            'years': forecast_years,
            'ci_lower': np.array(values) - ci_width,
            'ci_upper': np.array(values) + ci_width
        }


# ===============================
# Derived Metrics V3
# ===============================

class DerivedMetricsV3:
    """Calculate derived indicators with Monte Carlo error propagation."""
    
    @staticmethod
    def calculate_with_uncertainty(
        numerator: Tuple[float, float, float],
        denominator: Tuple[float, float, float],
        operation: str = 'divide',
        correlation: float = 0.0,
        n_simulations: int = 10000
    ) -> Tuple[float, float, float]:
        """Monte Carlo propagation with robust handling for ill-conditioned covariance."""
        
        num_val, num_low, num_high = numerator
        den_val, den_low, den_high = denominator
        
        num_se = (num_high - num_low) / (2 * 1.96)
        den_se = (den_high - den_low) / (2 * 1.96)
        
        epsilon = 1e-10
        num_se = max(num_se, epsilon)
        den_se = max(den_se, epsilon)
        
        if operation == 'divide' and abs(den_val) < epsilon:
            return 0, 0, 0
        
        try:
            if abs(correlation) > epsilon:
                mean = [num_val, den_val]
                cov = [
                    [num_se**2 + epsilon, correlation * num_se * den_se],
                    [correlation * num_se * den_se, den_se**2 + epsilon]
                ]
                
                samples = np.random.multivariate_normal(mean, cov, n_simulations)
                num_samples = samples[:, 0]
                den_samples = samples[:, 1]
            else:
                num_samples = np.random.normal(num_val, num_se, n_simulations)
                den_samples = np.random.normal(den_val, den_se, n_simulations)
        except np.linalg.LinAlgError:
            num_samples = np.random.normal(num_val, num_se, n_simulations)
            den_samples = np.random.normal(den_val, den_se, n_simulations)
        
        if operation == 'divide':
            den_samples = np.where(
                np.abs(den_samples) < epsilon,
                epsilon * np.sign(den_samples),
                den_samples
            )
            result_samples = num_samples / den_samples
        elif operation == 'multiply':
            result_samples = num_samples * den_samples
        elif operation == 'add':
            result_samples = num_samples + den_samples
        elif operation == 'subtract':
            result_samples = num_samples - den_samples
        else:
            result_samples = num_samples
        
        mean_result = np.mean(result_samples)
        std_result = np.std(result_samples)
        valid_samples = result_samples[
            np.abs(result_samples - mean_result) < 5 * std_result
        ]
        
        if len(valid_samples) < 100:
            if operation == 'divide':
                result_val = num_val / den_val if den_val != 0 else 0
            elif operation == 'multiply':
                result_val = num_val * den_val
            else:
                result_val = num_val
            
            rel_error = np.sqrt(
                (num_se/abs(num_val))**2 + (den_se/abs(den_val))**2
            ) if num_val != 0 and den_val != 0 else 0.1
            ci_width = 1.96 * abs(result_val) * rel_error
            
            return result_val, result_val - ci_width, result_val + ci_width
        
        result_val = np.median(valid_samples)
        result_low = np.percentile(valid_samples, 2.5)
        result_high = np.percentile(valid_samples, 97.5)
        
        return result_val, result_low, result_high
    
    @staticmethod
    def calculate_all_derived(
        data: pd.DataFrame,
        region_code: str,
        year: int
    ) -> List[Dict]:
        """Calculate all derived ITL2 metrics with uncertainty for a region-year."""
        derived = []
        
        region_year_data = data[
            (data['region_code'] == region_code) & 
            (data['year'] == year)
        ]
        
        if region_year_data.empty:
            return derived
        
        metrics_dict = {}
        for _, row in region_year_data.iterrows():
            metric = row['metric']
            value = row['value']
            ci_lower = row.get('ci_lower', value * 0.95)
            ci_upper = row.get('ci_upper', value * 1.05)
            metrics_dict[metric] = (value, ci_lower, ci_upper)
        
        # Productivity (GVA per worker)
        if 'nominal_gva_mn_gbp' in metrics_dict and 'emp_total_jobs' in metrics_dict:
            gva_pounds = (
                metrics_dict['nominal_gva_mn_gbp'][0] * 1e6,
                metrics_dict['nominal_gva_mn_gbp'][1] * 1e6,
                metrics_dict['nominal_gva_mn_gbp'][2] * 1e6
            )
            
            prod_val, prod_low, prod_high = DerivedMetricsV3.calculate_with_uncertainty(
                gva_pounds,
                metrics_dict['emp_total_jobs'],
                operation='divide',
                correlation=0.8
            )
            
            derived.append({
                'region_code': region_code,
                'year': year,
                'metric': 'productivity_gbp_per_job',
                'value': prod_val,
                'ci_lower': prod_low,
                'ci_upper': prod_high,
                'method': 'derived_monte_carlo',
                'source': 'calculated',
                'data_type': 'forecast' if year > 2023 else 'historical'
            })
        
        # Employment rate
        if 'emp_total_jobs' in metrics_dict and 'population_total' in metrics_dict:
            emp_rate_val, emp_rate_low, emp_rate_high = DerivedMetricsV3.calculate_with_uncertainty(
                metrics_dict['emp_total_jobs'],
                metrics_dict['population_total'],
                operation='divide',
                correlation=0.6
            )
            
            derived.append({
                'region_code': region_code,
                'year': year,
                'metric': 'employment_rate',
                'value': emp_rate_val * 100,
                'ci_lower': emp_rate_low * 100,
                'ci_upper': emp_rate_high * 100,
                'method': 'derived_monte_carlo',
                'source': 'calculated',
                'data_type': 'forecast' if year > 2023 else 'historical'
            })
        
        # Income per worker
        if 'gdhi_total_mn_gbp' in metrics_dict and 'emp_total_jobs' in metrics_dict:
            gdhi_pounds = (
                metrics_dict['gdhi_total_mn_gbp'][0] * 1e6,
                metrics_dict['gdhi_total_mn_gbp'][1] * 1e6,
                metrics_dict['gdhi_total_mn_gbp'][2] * 1e6
            )
            
            income_val, income_low, income_high = DerivedMetricsV3.calculate_with_uncertainty(
                gdhi_pounds,
                metrics_dict['emp_total_jobs'],
                operation='divide',
                correlation=0.7
            )
            
            derived.append({
                'region_code': region_code,
                'year': year,
                'metric': 'income_per_worker_gbp',
                'value': income_val,
                'ci_lower': income_low,
                'ci_upper': income_high,
                'method': 'derived_monte_carlo',
                'source': 'calculated',
                'data_type': 'forecast' if year > 2023 else 'historical'
            })
        
        # GDHI per head
        if 'gdhi_total_mn_gbp' in metrics_dict and 'population_total' in metrics_dict:
            gdhi_pounds = (
                metrics_dict['gdhi_total_mn_gbp'][0] * 1e6,
                metrics_dict['gdhi_total_mn_gbp'][1] * 1e6,
                metrics_dict['gdhi_total_mn_gbp'][2] * 1e6
            )
            
            per_head_val, per_head_low, per_head_high = DerivedMetricsV3.calculate_with_uncertainty(
                gdhi_pounds,
                metrics_dict['population_total'],
                operation='divide',
                correlation=0.9
            )
            
            derived.append({
                'region_code': region_code,
                'year': year,
                'metric': 'gdhi_per_head_gbp',
                'value': per_head_val,
                'ci_lower': per_head_low,
                'ci_upper': per_head_high,
                'method': 'derived_monte_carlo',
                'source': 'calculated',
                'data_type': 'forecast' if year > 2023 else 'historical'
            })
        
        return derived


# ===============================
# Main Pipeline ITL2 V3.5
# ===============================

class InstitutionalForecasterITL2V3:
    """V3.5 production pipeline for ITL2 with VAR/VECM validation and ITL1 anchoring."""
    
    def __init__(self, config: ForecastConfigITL2):
        self.config = config
        self.data_manager = DataManagerITL2(config)
        self.forecaster = AdvancedForecastingV3(config)
        self.derived = DerivedMetricsV3()
        
        if config.use_macro_anchoring and HAVE_DUCKDB:
            self.anchor_manager = ITL1AnchorManager(config.duckdb_path)
            if self.anchor_manager.has_anchors():
                self.reconciler = TopDownReconcilerITL2(config, self.anchor_manager)
                logger.info("✓ ITL1 anchors loaded - ITL2→ITL1 reconciliation enabled")
            else:
                self.reconciler = None
                logger.warning("⚠️ No ITL1 anchors - reconciliation disabled")
        else:
            self.anchor_manager = None
            self.reconciler = None
    
    def run(self) -> pd.DataFrame:
        """Execute full ITL2 pipeline: ingest → forecast → reconcile → derive → persist."""
        logger.info("=" * 60)
        logger.info("ITL2 FORECASTING ENGINE V3.5 (VAR VALIDATED)")
        logger.info("=" * 60)
        
        historical = self.data_manager.load_all_data()
        tasks = self._identify_tasks(historical)
        logger.info(f"ITL2 forecasting tasks: {len(tasks)}")
        
        results = self._run_forecasts_with_systems(historical, tasks)
        
        all_data = pd.concat([historical, pd.DataFrame(results)], ignore_index=True)
        
        # Reconcile to ITL1 anchors
        if self.reconciler:
            all_data = self.reconciler.reconcile(all_data)
        
        # Derived metrics (productivity, employment rate, income per worker, GDHI per head)
        logger.info("Calculating derived indicators (ITL2)...")
        derived = self._calculate_derived(all_data)
        if derived:
            all_data = pd.concat([all_data, pd.DataFrame(derived)], ignore_index=True)
        
        logger.info("Saving ITL2 outputs...")
        self._save_outputs(all_data)
        
        logger.info("✅ ITL2 forecasting V3.5 completed (VAR validated)")
        return all_data
    
    def _identify_tasks(self, data: pd.DataFrame) -> List[Dict]:
        """Identify (region, metric) tasks eligible for forecasting."""
        tasks = []
        for (region_code, region, metric), group in data.groupby(['region_code', 'region', 'metric']):
            if len(group) < self.config.min_history_years:
                continue
            
            last_year = int(group['year'].max())
            horizon = min(self.config.target_year - last_year, 30)
            if horizon <= 0:
                continue
            
            tasks.append({
                'region_code': region_code,
                'region': region,
                'metric': metric,
                'horizon': horizon,
                'last_year': last_year,
                'history_length': len(group),
                'metric_info': self.config.metric_definitions.get(metric, {})
            })
        
        return tasks
    
    def _run_forecasts_with_systems(
        self,
        data: pd.DataFrame,
        tasks: List[Dict]
    ) -> List[Dict]:
        """Run forecasts for all tasks, using VAR systems where valid."""
        results = []
        processed = set()
        
        # Group tasks by region
        tasks_by_region: Dict[str, List[Dict]] = {}
        for task in tasks:
            rc = task['region_code']
            tasks_by_region.setdefault(rc, []).append(task)
        
        for region_code, region_tasks in tasks_by_region.items():
            logger.info(f"\nITL2 Region: {region_code}")
            
            region_metrics = {t['metric'] for t in region_tasks}
            
            # VAR systems first (e.g. GVA + employment)
            for system_name, system_metrics in self.config.var_systems.items():
                available = [m for m in system_metrics if m in region_metrics]
                
                if (
                    len(available) >= 2 and
                    self.config.use_var_systems and
                    self.forecaster.var_forecaster
                ):
                    logger.info(f"  Attempting VAR system: {available}")
                    
                    horizons = [
                        t['horizon'] for t in region_tasks if t['metric'] in available
                    ]
                    common_horizon = min(horizons) if horizons else 0
                    
                    if common_horizon > 0:
                        breaks = [
                            b for b in self.config.structural_breaks
                            if any(
                                t['last_year'] - 20 <= b['year'] <= t['last_year']
                                for t in region_tasks if t['metric'] in available
                            )
                        ]
                        
                        var_results = self.forecaster.var_forecaster.forecast_system(
                            data, region_code, available, common_horizon, breaks
                        )
                        
                        if var_results:
                            for metric, fc in var_results.items():
                                processed.add((region_code, metric))
                                task = next(t for t in region_tasks if t['metric'] == metric)
                                
                                for j, year in enumerate(fc['years']):
                                    results.append({
                                        'region': task['region'],
                                        'region_code': region_code,
                                        'metric': metric,
                                        'year': year,
                                        'value': fc['values'][j],
                                        'ci_lower': fc['ci_lower'][j],
                                        'ci_upper': fc['ci_upper'][j],
                                        'method': fc['method'],
                                        'data_type': 'forecast',
                                        'source': 'model',
                                        'quality_metrics': {
                                            'history_years': task['history_length'],
                                            'method': fc['method'],
                                            'system_metrics': fc.get('system_metrics', [metric]),
                                            'aic': fc.get('aic'),
                                            'cointegrated': fc.get('cointegrated', False)
                                        }
                                    })
                            
                            logger.info(f"  ✓ VAR: {len(var_results)} metrics forecasted")
            
            # Univariate ensemble for remaining metrics
            for task in region_tasks:
                if (task['region_code'], task['metric']) in processed:
                    continue
                
                logger.info(f"  {task['metric']} (univariate)")
                
                series_data = data[
                    (data['region_code'] == task['region_code']) &
                    (data['metric'] == task['metric'])
                ].sort_values('year')
                
                series = pd.Series(
                    series_data['value'].values,
                    index=series_data['year'].values.astype(int),
                    name=task['metric']
                )
                
                use_log = self.config.use_log_transform.get(task['metric'], False)
                working = np.log(series) if use_log and (series > 0).all() else series
                
                breaks = [
                    b for b in self.config.structural_breaks
                    if task['last_year'] - 20 <= b['year'] <= task['last_year']
                ]
                
                try:
                    fc = self.forecaster.forecast_univariate(
                        working, task['horizon'], breaks, task['metric_info']
                    )
                    
                    if use_log:
                        if 'cv_residual_std' in fc and fc['cv_residual_std']:
                            bias_correction = np.exp(0.5 * fc['cv_residual_std']**2)
                        else:
                            bias_correction = 1.0
                        
                        fc['values'] = np.exp(fc['values']) * bias_correction
                        fc['ci_lower'] = np.exp(fc['ci_lower'])
                        fc['ci_upper'] = np.exp(fc['ci_upper'])
                    
                    for j, year in enumerate(fc['years']):
                        results.append({
                            'region': task['region'],
                            'region_code': task['region_code'],
                            'metric': task['metric'],
                            'year': year,
                            'value': fc['values'][j],
                            'ci_lower': fc['ci_lower'][j],
                            'ci_upper': fc['ci_upper'][j],
                            'method': fc['method'],
                            'data_type': 'forecast',
                            'source': 'model',
                            'quality_metrics': {
                                'history_years': task['history_length'],
                                'method': fc['method'],
                                'system_metrics': [task['metric']],
                                'aic': fc.get('aic'),
                                'component_methods': fc.get('component_methods', []),
                                'weights': fc.get('weights', [])
                            }
                        })
                except Exception as e:
                    logger.error(f"Failed ITL2 {task['region_code']}-{task['metric']}: {e}")
        
        return results
    
    def _calculate_derived(self, data: pd.DataFrame) -> List[Dict]:
        """Calculate derived metrics for all region-year combinations."""
        derived = []
        for region_code in data['region_code'].unique():
            region_data = data[data['region_code'] == region_code]
            region = region_data['region'].iloc[0] if 'region' in region_data.columns else region_code
            
            for year in data['year'].unique():
                try:
                    year_results = self.derived.calculate_all_derived(
                        data, region_code, int(year)
                    )
                    for r in year_results:
                        r['region'] = region
                        derived.append(r)
                except Exception as e:
                    logger.error(f"Derived failed ITL2 {region_code}-{year}: {e}")
        return derived
    
    def _save_outputs(self, data: pd.DataFrame):
        """Persist ITL2 outputs to CSV and DuckDB."""
        prefix = self.config.output_prefix
        
        # Long format
        long_path = self.config.output_dir / f"{prefix}_forecast_v3_long.csv"
        data.to_csv(long_path, index=False)
        logger.info(f"✓ Long: {long_path}")
        
        # Wide format
        wide = data.pivot_table(
            index=['region', 'region_code', 'metric'], 
            columns='year', values='value', aggfunc='first'
        ).reset_index()
        wide.columns.name = None
        wide_path = self.config.output_dir / f"{prefix}_forecast_v3_wide.csv"
        wide.to_csv(wide_path, index=False)
        logger.info(f"✓ Wide: {wide_path}")
        
        # Confidence intervals
        if 'ci_lower' in data.columns and 'ci_upper' in data.columns:
            ci_data = data[['region_code', 'metric', 'year', 'value', 'ci_lower', 'ci_upper']].copy()
            ci_data = ci_data.dropna(subset=['ci_lower', 'ci_upper'])
            if not ci_data.empty:
                ci_data['ci_width'] = ci_data['ci_upper'] - ci_data['ci_lower']
                ci_data['cv'] = ci_data['ci_width'] / (2 * ci_data['value'])
                ci_path = self.config.output_dir / f"{prefix}_confidence_intervals_v3.csv"
                ci_data.to_csv(ci_path, index=False)
                logger.info(f"✓ CIs: {ci_path}")
        
        # Quality metrics
        quality_data = []
        for _, row in data[data['data_type'] == 'forecast'].iterrows():
            qm = row.get('quality_metrics', None)
            if isinstance(qm, dict):
                quality_data.append({
                    'region_code': row['region_code'],
                    'metric': row['metric'],
                    'year': row['year'],
                    **qm
                })
        
        if quality_data:
            quality_df = pd.DataFrame(quality_data)
            quality_path = self.config.output_dir / f"{prefix}_forecast_quality_v3.csv"
            quality_df.to_csv(quality_path, index=False)
            logger.info(f"✓ Quality: {quality_path}")
        
        # Metadata
        metadata = {
            'run_timestamp': datetime.now().isoformat(),
            'version': '3.5_itl2_validated',
            'level': 'ITL2',
            'config': {
                'target_year': self.config.target_year,
                'var_enabled': self.config.use_var_systems,
                'min_history_years': self.config.min_history_years,
                'reconciled_to_itl1': self.config.use_macro_anchoring
            },
            'data_summary': {
                'regions': int(data['region_code'].nunique()),
                'metrics': int(data['metric'].nunique()),
                'total_obs': len(data),
                'forecast_obs': int((data['data_type'] == 'forecast').sum())
            },
            'model_usage': data[data['data_type'] == 'forecast']['method'].value_counts().to_dict()
            if 'method' in data.columns else {}
        }
        
        if hasattr(data, 'attrs') and 'reconciliation_log' in data.attrs:
            recon = data.attrs['reconciliation_log']
            if recon:
                metadata['reconciliation_summary'] = {
                    'adjustments': len(recon),
                    'avg_scale_factor': float(np.mean([r['scale_factor'] for r in recon])),
                    'max_deviation_pct': float(max([r['deviation_pct'] for r in recon]))
                }
        
        metadata_path = self.config.output_dir / f"{prefix}_metadata_v3.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"✓ Metadata: {metadata_path}")
        
        # DuckDB gold schema
        if HAVE_DUCKDB:
            try:
                data_copy = data.copy()
                
                # --- Critical column handling: region_name / region_level / metric_id / period ---
                
                # metric_id
                if "metric" in data_copy.columns:
                    if "metric_id" not in data_copy.columns:
                        data_copy["metric_id"] = data_copy["metric"]
                    else:
                        data_copy["metric_id"] = data_copy["metric_id"].fillna(data_copy["metric"])
                
                # period
                if "year" in data_copy.columns:
                    if "period" not in data_copy.columns:
                        data_copy["period"] = data_copy["year"]
                    else:
                        data_copy["period"] = data_copy["period"].fillna(data_copy["year"])
                
                # region_name
                if "region" in data_copy.columns:
                    if "region_name" not in data_copy.columns:
                        data_copy["region_name"] = data_copy["region"]
                    else:
                        mask_null = data_copy["region_name"].isna()
                        data_copy.loc[mask_null, "region_name"] = data_copy.loc[mask_null, "region"]
                else:
                    if "region_name" not in data_copy.columns:
                        logger.warning("⚠️  No 'region' column found - region_name may be incomplete")
                        data_copy["region_name"] = "UNKNOWN"
                
                null_region_names = data_copy["region_name"].isna().sum()
                if null_region_names > 0:
                    logger.error(f"❌ CRITICAL: {null_region_names} rows with NULL region_name after fixing!")
                    data_copy["region_name"] = data_copy["region_name"].fillna("MISSING_REGION_NAME")
                
                # region_level
                if "region_level" not in data_copy.columns:
                    data_copy["region_level"] = "ITL2"
                else:
                    data_copy["region_level"] = data_copy["region_level"].fillna("ITL2")
                data_copy["region_level"] = "ITL2"  # force ITL2 for this pipeline
                
                # unit
                if 'unit' not in data_copy.columns:
                    metric_to_unit = {
                        m: self.config.metric_definitions.get(m, {}).get('unit', 'unknown')
                        for m in data_copy['metric_id'].unique()
                    }
                    data_copy['unit'] = data_copy['metric_id'].map(metric_to_unit)
                
                # freq
                if 'freq' not in data_copy.columns:
                    data_copy['freq'] = 'A'
                
                # metadata
                data_copy["forecast_run_date"] = datetime.now().date()
                data_copy["forecast_version"] = "3.5_itl2_validated"
                
                # Validation
                null_checks = {
                    'region_name': data_copy['region_name'].isna().sum(),
                    'region_level': data_copy['region_level'].isna().sum(),
                    'region_code': data_copy['region_code'].isna().sum(),
                    'metric_id': data_copy['metric_id'].isna().sum()
                }
                
                critical_nulls = {k: v for k, v in null_checks.items() if v > 0}
                if critical_nulls:
                    logger.warning(f"⚠️  NULL values detected before DuckDB write: {critical_nulls}")
                else:
                    logger.info("✓ All critical columns validated (no NULLs) before DuckDB write")
                
                cols = [
                    "region_code", "region_name", "region_level", "metric_id",
                    "period", "value", "unit", "freq", "data_type",
                    "ci_lower", "ci_upper", "forecast_run_date", "forecast_version"
                ]
                data_flat = data_copy[[c for c in cols if c in data_copy.columns]].reset_index(drop=True)
                
                con = duckdb.connect(str(self.config.duckdb_path))
                con.execute("CREATE SCHEMA IF NOT EXISTS gold")
                con.register("forecast_df", data_flat)
                
                con.execute("""
                    CREATE OR REPLACE TABLE gold.itl2_forecast AS
                    SELECT * FROM forecast_df
                """)
                
                con.execute("""
                    CREATE OR REPLACE VIEW gold.itl2_forecast_only AS
                    SELECT * FROM gold.itl2_forecast
                    WHERE data_type = 'forecast'
                """)
                
                # Unified view for history + forecast
                con.execute("""
                    CREATE OR REPLACE VIEW gold.itl2_latest AS
                    SELECT 
                        region_code,
                        region_name,
                        'ITL2' as region_level,
                        metric_id,
                        period,
                        value,
                        unit,
                        freq,
                        'historical' as data_type,
                        CAST(NULL AS DOUBLE) as ci_lower,
                        CAST(NULL AS DOUBLE) as ci_upper,
                        CAST(NULL AS DATE) as forecast_run_date,
                        CAST(NULL AS VARCHAR) as forecast_version
                    FROM silver.itl2_history
                    UNION ALL
                    SELECT 
                        region_code,
                        region_name,
                        region_level,
                        metric_id,
                        period,
                        value,
                        unit,
                        freq,
                        data_type,
                        ci_lower,
                        ci_upper,
                        forecast_run_date,
                        forecast_version
                    FROM gold.itl2_forecast
                    WHERE data_type = 'forecast'
                """)
                
                con.close()
                
                # Verification
                verify_con = duckdb.connect(str(self.config.duckdb_path), read_only=True)
                null_verification = verify_con.execute("""
                    SELECT 
                        COUNT(*) as total_rows,
                        SUM(CASE WHEN region_name IS NULL THEN 1 ELSE 0 END) as null_region_name,
                        SUM(CASE WHEN region_level IS NULL THEN 1 ELSE 0 END) as null_region_level,
                        SUM(CASE WHEN region_code IS NULL THEN 1 ELSE 0 END) as null_region_code
                    FROM gold.itl2_forecast
                """).fetchone()
                verify_con.close()
                
                total, null_name, null_level, null_code = null_verification
                
                logger.info(f"✓ DuckDB: gold.itl2_forecast ({len(data_flat)} rows)")
                logger.info("✓ DuckDB: gold.itl2_latest (unified history+forecast view for Supabase)")
                
                if null_name > 0 or null_level > 0 or null_code > 0:
                    logger.error("❌ CRITICAL: NULLs detected in gold.itl2_forecast!")
                    logger.error(f"   NULL region_name: {null_name}/{total}")
                    logger.error(f"   NULL region_level: {null_level}/{total}")
                    logger.error(f"   NULL region_code: {null_code}/{total}")
                else:
                    logger.info(f"✓ Verified: No NULLs in critical columns ({total:,} rows)")
                
            except Exception as e:
                logger.warning(f"DuckDB save failed: {e}")
        
        logger.info("=" * 70)
        logger.info("✅ ITL2 FORECASTING V3.5 COMPLETED")
        logger.info(f"📊 Total records: {len(data)}")
        logger.info(f"📁 Outputs: {self.config.output_dir}")
        logger.info("=" * 70)


# ===============================
# Entrypoint
# ===============================

def main():
    """Run ITL2 forecasting pipeline V3.5 with VAR validation."""
    try:
        config = ForecastConfigITL2()
        forecaster = InstitutionalForecasterITL2V3(config)
        results = forecaster.run()
        
        logger.info("=" * 70)
        print("\nITL2 V3.5 FORECAST SUMMARY (VAR VALIDATED)")
        print(f"Regions: {results['region_code'].nunique()}")
        print(f"Metrics: {results['metric'].nunique()}")
        print(f"Forecast observations: {len(results[results['data_type'] == 'forecast'])}")
        
        if 'method' in results.columns:
            method_counts = results[results['data_type'] == 'forecast']['method'].value_counts()
            print("\nModel usage:")
            for method, count in method_counts.items():
                pct = 100 * count / len(results[results['data_type'] == 'forecast'])
                print(f"  {method}: {count} ({pct:.1f}%)")
        
        print("\nOutputs:")
        print(f"  - data/forecast/itl2_forecast_v3_long.csv")
        print(f"  - data/forecast/itl2_forecast_v3_wide.csv")
        print(f"  - data/forecast/itl2_confidence_intervals_v3.csv")
        print(f"  - data/forecast/itl2_forecast_quality_v3.csv")
        print(f"  - data/forecast/itl2_metadata_v3.json")
        print(f"  - gold.itl2_forecast (DuckDB table)")
        print(f"  - gold.itl2_forecast_only (DuckDB view)")
        print(f"  - gold.itl2_latest (DuckDB unified view)")
        print("=" * 70)
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
