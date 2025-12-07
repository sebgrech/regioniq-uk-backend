#!/usr/bin/env python3
"""
Region IQ - ITL1 Regional Forecasting Engine V3.2 (With Macro Anchoring)
========================================================================

V3.2: VAR/VECM + Top-Down Reconciliation
- Enterprise-grade regional forecasting with cross-metric coherence
- VAR/VECM for GVA ↔ Employment pairs
- Top-down reconciliation to UK macro totals
- Ensures: Σ(ITL1) = UK for all metrics

Architecture:
  1. Load UK macro forecasts from gold.uk_macro_forecast (run macro V3.2 first)
  2. Generate regional forecasts using VAR + univariate methods
  3. Reconcile regional forecasts to match UK totals (proportional scaling)
  4. Calculate derived metrics
  5. Output to gold.itl1_forecast

Author: Region IQ
Version: 3.2 (Top-Down)
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('forecast_engine_v3.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import statistical packages
try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
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
class ForecastConfig:
    """V3.2 Configuration with macro anchoring"""
    
    # Data paths
    silver_path: Path = Path("data/silver/itl1_unified_history.csv")
    duckdb_path: Path = Path("data/lake/warehouse.duckdb")
    use_duckdb: bool = False
    
    # Output
    output_dir: Path = Path("data/forecast")
    cache_dir: Path = Path("data/cache")
    
    # Forecast parameters
    target_year: int = 2050
    min_history_years: int = 10
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
    
    # NEW: Top-down reconciliation
    use_macro_anchoring: bool = True
    reconciliation_method: str = 'proportional'
    
    # Macro anchor mapping (ITL1 metric → UK metric)
    macro_anchor_map: Dict[str, str] = field(default_factory=lambda: {
        'nominal_gva_mn_gbp': 'uk_nominal_gva_mn_gbp',
        'gdhi_total_mn_gbp': 'uk_gdhi_total_mn_gbp',
        'gdhi_per_head_gbp': 'uk_gdhi_per_head_gbp',
        'emp_total_jobs': 'uk_emp_total_jobs',
        'population_total': 'uk_population_total'
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
                'gdhi_per_head_gbp': {'unit': 'GBP', 'transform': 'log', 'monotonic': False},
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
        if not self.use_duckdb and not self.silver_path.exists():
            raise FileNotFoundError(f"Silver data not found: {self.silver_path}")
        if self.use_duckdb and not self.duckdb_path.exists():
            raise FileNotFoundError(f"DuckDB not found: {self.duckdb_path}")
        assert self.target_year > 2024
        logger.info("ITL1 V3.2 (Top-Down) Configuration validated")


# ===============================
# NEW: Macro Anchor Manager
# ===============================

class MacroAnchorManager:
    """Loads UK macro forecasts for top-down reconciliation"""
    
    def __init__(self, duckdb_path: Path):
        self.duckdb_path = duckdb_path
        self.anchors = self._load_anchors()
    
    def _load_anchors(self) -> pd.DataFrame:
        """Load UK macro forecasts from gold.uk_macro_forecast"""
        
        if not HAVE_DUCKDB or not self.duckdb_path.exists():
            logger.warning("Cannot load macro anchors - DuckDB unavailable")
            return pd.DataFrame()
        
        try:
            con = duckdb.connect(str(self.duckdb_path), read_only=True)
            
            # Load UK forecasts
            anchors = con.execute("""
                SELECT metric_id, period, value
                FROM gold.uk_macro_forecast
                WHERE data_type = 'forecast'
            """).fetchdf()
            
            con.close()
            
            if anchors.empty:
                logger.warning("No UK macro forecasts found in gold.uk_macro_forecast")
                return pd.DataFrame()
            
            # Standardize columns
            if 'period' in anchors.columns and 'year' not in anchors.columns:
                anchors['year'] = pd.to_numeric(anchors['period'], errors='coerce')
            
            logger.info(f"✓ Loaded UK macro anchors: {anchors['metric_id'].nunique()} metrics")
            logger.info(f"  Year range: {anchors['year'].min()}-{anchors['year'].max()}")
            
            return anchors
            
        except Exception as e:
            logger.error(f"Failed to load macro anchors: {e}")
            return pd.DataFrame()
    
    def get_uk_value(self, metric: str, year: int) -> Optional[float]:
        """Get UK forecast value for specific metric and year"""
        if self.anchors.empty:
            return None
        
        match = self.anchors[
            (self.anchors['metric_id'] == metric) &
            (self.anchors['year'] == year)
        ]
        
        if match.empty:
            return None
        
        return match['value'].iloc[0]
    
    def has_anchors(self) -> bool:
        """Check if anchors are available"""
        return not self.anchors.empty


# ===============================
# NEW: Top-Down Reconciler
# ===============================

class TopDownReconciler:
    """
    Reconciles ITL1 regional forecasts to UK macro totals.
    Ensures Σ(ITL1) = UK for each metric and year.
    """
    
    def __init__(self, config: ForecastConfig, macro_manager: MacroAnchorManager):
        self.config = config
        self.macro = macro_manager
    
    def reconcile(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply proportional reconciliation: scale regional forecasts so they sum to UK total.
        
        For each (metric, year):
          scale_factor = UK_total / Σ(ITL1_forecasts)
          ITL1_adjusted = ITL1_forecast × scale_factor
        """
        
        if not self.macro.has_anchors():
            logger.warning("No macro anchors - skipping reconciliation")
            return data
        
        logger.info("\n" + "="*70)
        logger.info("TOP-DOWN RECONCILIATION")
        logger.info("="*70)
        
        reconciliation_log = []
        
        forecast_data = data[data['data_type'] == 'forecast']
        forecast_years = sorted(forecast_data['year'].unique())
        
        for metric in forecast_data['metric'].unique():
            # Get UK metric name
            uk_metric = self.config.macro_anchor_map.get(metric)
            if not uk_metric:
                logger.debug(f"  {metric}: No UK anchor mapping")
                continue
            
            logger.info(f"\n  Reconciling {metric} → {uk_metric}")
            
            for year in forecast_years:
                year_int = int(year)
                
                # Get UK anchor
                uk_value = self.macro.get_uk_value(uk_metric, year_int)
                if uk_value is None:
                    continue
                
                # Get ITL1 regional forecasts for this year
                mask = (
                    (data['year'] == year_int) &
                    (data['metric'] == metric) &
                    (data['data_type'] == 'forecast')
                )
                
                if not mask.any():
                    continue
                
                # Calculate regional sum (before reconciliation)
                regional_sum_before = data.loc[mask, 'value'].sum()
                
                if regional_sum_before <= 0:
                    logger.warning(f"    {year}: Regional sum ≤ 0, skipping")
                    continue
                
                # Calculate scale factor
                scale_factor = uk_value / regional_sum_before
                
                # Apply scaling to forecasts and CIs
                data.loc[mask, 'value'] *= scale_factor
                data.loc[mask, 'ci_lower'] *= scale_factor
                data.loc[mask, 'ci_upper'] *= scale_factor
                
                # Verify
                regional_sum_after = data.loc[mask, 'value'].sum()
                deviation = abs(regional_sum_after - uk_value) / uk_value if uk_value > 0 else 0
                
                reconciliation_log.append({
                    'year': year_int,
                    'metric': metric,
                    'uk_value': uk_value,
                    'regional_sum_before': regional_sum_before,
                    'regional_sum_after': regional_sum_after,
                    'scale_factor': scale_factor,
                    'deviation_pct': deviation * 100
                })
                
                if year_int in [2025, 2030, 2040, 2050]:  # Log key years
                    logger.info(f"    {year}: SF={scale_factor:.4f} | UK={uk_value:,.0f} | Regional: {regional_sum_before:,.0f}→{regional_sum_after:,.0f}")
        
        # Store audit trail
        data.attrs['reconciliation_log'] = reconciliation_log
        
        logger.info(f"\n✓ Reconciliation complete: {len(reconciliation_log)} adjustments")
        
        return data


# ===============================
# Data Management (UNCHANGED)
# ===============================

class DataManagerV3:
    """Data management for unified silver schema"""
    
    def __init__(self, config: ForecastConfig):
        self.config = config
        
    def load_all_data(self) -> pd.DataFrame:
        """Load from unified silver schema"""
        cache_key = self._get_cache_key()
        if self.config.cache_enabled and self._cache_exists(cache_key):
            logger.info("Loading data from cache")
            return self._load_from_cache(cache_key)
        
        if self.config.use_duckdb:
            unified = self._load_from_duckdb()
        else:
            unified = self._load_from_csv()
        
        unified = self._standardize_columns(unified)
        unified = self._handle_outliers(unified)
        
        if self.config.cache_enabled:
            self._save_to_cache(unified, cache_key)
        
        logger.info(f"Loaded {len(unified)} rows")
        logger.info(f"Regions: {unified['region_code'].nunique()}, Metrics: {unified['metric'].nunique()}")
        
        return unified
    
    def _load_from_csv(self) -> pd.DataFrame:
        logger.info(f"Loading from {self.config.silver_path}")
        df = pd.read_csv(self.config.silver_path)
        
        expected_cols = ['region_code', 'region_name', 'metric_id', 'period', 'value']
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            if 'metric' in df.columns and 'metric_id' not in df.columns:
                df['metric_id'] = df['metric']
            if 'year' in df.columns and 'period' not in df.columns:
                df['period'] = df['year']
            
            missing = [c for c in expected_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")
        
        return df
    
    def _load_from_duckdb(self) -> pd.DataFrame:
        try:
            import duckdb
            con = duckdb.connect(str(self.config.duckdb_path), read_only=True)
            df = con.execute("SELECT * FROM silver.itl1_history").fetchdf()
            con.close()
            return df
        except Exception as e:
            logger.error(f"DuckDB load failed: {e}")
            return self._load_from_csv()
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {
            'metric_id': 'metric',
            'period': 'year',
            'region_name': 'region'
        }
        
        for old, new in rename_map.items():
            if old in df.columns and new not in df.columns:
                df[new] = df[old]
        
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        if 'data_type' not in df.columns:
            df['data_type'] = 'historical'
        
        df = df.dropna(subset=['year', 'value'])
        
        if self.config.enforce_non_negative:
            df = df[df['value'] > 0]
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        for (region, metric), group in df.groupby(['region_code', 'metric']):
            if len(group) < 5:
                continue
            
            values = group['value'].values
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            
            outliers = (values < lower_bound) | (values > upper_bound)
            if outliers.any():
                logger.info(f"Winsorizing {outliers.sum()} outliers in {region}-{metric}")
                df.loc[group[outliers].index, 'value'] = np.clip(values[outliers], lower_bound, upper_bound)
        
        return df
    
    def _get_cache_key(self) -> str:
        hasher = hashlib.md5()
        path = self.config.duckdb_path if self.config.use_duckdb else self.config.silver_path
        if path.exists():
            hasher.update(str(path.stat().st_mtime).encode())
        return hasher.hexdigest()
    
    def _cache_exists(self, key: str) -> bool:
        return (self.config.cache_dir / f"data_{key}.pkl").exists()
    
    def _load_from_cache(self, key: str) -> pd.DataFrame:
        with open(self.config.cache_dir / f"data_{key}.pkl", 'rb') as f:
            return pickle.load(f)
    
    def _save_to_cache(self, df: pd.DataFrame, key: str):
        with open(self.config.cache_dir / f"data_{key}.pkl", 'wb') as f:
            pickle.dump(df, f)


# ===============================
# VAR System Forecaster (SAME AS BEFORE)
# ===============================

class VARSystemForecaster:
    """VAR/VECM for cross-metric coherence"""
    
    def __init__(self, config: ForecastConfig):
        self.config = config
    
    def forecast_system(
        self,
        data: pd.DataFrame,
        region_code: str,
        metrics: List[str],
        horizon: int,
        structural_breaks: Optional[Sequence[Dict]] = None
    ) -> Optional[Dict]:
        """Forecast multiple metrics together"""
        if not HAVE_STATSMODELS:
            return None
        
        try:
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
                
                series_dict[metric] = pd.Series(values, index=metric_data['year'].values)
            
            system_df = pd.DataFrame(series_dict).dropna()
            
            if len(system_df) < self.config.min_history_years:
                return None
            
            coint_result = self._test_cointegration(system_df)
            exog = self._prep_breaks(system_df.index, structural_breaks) if structural_breaks else None
            
            if coint_result['cointegrated'] and coint_result['rank'] > 0:
                logger.info(f"  Using VECM (rank={coint_result['rank']})")
                model = VECM(system_df, k_ar_diff=min(self.config.max_var_lags, 3), 
                            coint_rank=coint_result['rank'], deterministic='ci', exog=exog)
                fitted = model.fit()
                method_name = f'VECM(r={coint_result["rank"]})'
                exog_fc = self._extend_exog(exog, system_df.index, horizon) if exog else None
                forecast_values = fitted.predict(steps=horizon, exog_fc=exog_fc)
            else:
                logger.info("  Using VAR")
                model = VAR(system_df, exog=exog)
                lag_results = model.select_order(maxlags=self.config.max_var_lags)
                try:
                    optimal_lags = lag_results.selected_orders['aic']
                except (AttributeError, KeyError):
                    optimal_lags = lag_results.aic
                optimal_lags = max(1, min(optimal_lags, self.config.max_var_lags))
                
                fitted = model.fit(maxlags=optimal_lags)
                method_name = f'VAR({optimal_lags})'
                exog_fc = self._extend_exog(exog, system_df.index, horizon) if exog else None
                forecast_values = fitted.forecast(system_df.values[-fitted.k_ar:], steps=horizon, exog_future=exog_fc)
            
            ci_lower, ci_upper = self._bootstrap_ci(fitted, system_df, horizon, exog, exog_fc)
            
            last_year = system_df.index[-1]
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
            
            return results
        except Exception as e:
            logger.debug(f"VAR failed: {e}")
            return None
    
    def _test_cointegration(self, data: pd.DataFrame) -> Dict:
        if len(data) < 30:
            return {'cointegrated': False, 'rank': 0}
        try:
            result = coint_johansen(data, det_order=0, k_ar_diff=2)
            rank = sum(result.lr1 > result.cvt[:, 1])
            return {'cointegrated': rank > 0, 'rank': rank}
        except:
            return {'cointegrated': False, 'rank': 0}
    
    def _bootstrap_ci(self, fitted, data, horizon, exog, exog_fc, n_boot=300):
        n_vars = data.shape[1]
        boot_fcs = []
        
        residuals = fitted.resid if hasattr(fitted, 'resid') else data.values[fitted.k_ar:] - fitted.fittedvalues
        
        for _ in range(n_boot):
            try:
                boot_resid = residuals[np.random.choice(len(residuals), len(residuals), replace=True)]
                boot_df = pd.DataFrame(fitted.fittedvalues + boot_resid, columns=data.columns, 
                                      index=data.index[-len(fitted.fittedvalues):])
                
                if hasattr(fitted, 'coint_rank'):
                    boot_model = VECM(boot_df, k_ar_diff=fitted.k_ar_diff, coint_rank=fitted.coint_rank, 
                                     deterministic='ci', exog=exog).fit()
                    boot_fc = boot_model.predict(steps=horizon, exog_fc=exog_fc)
                else:
                    boot_model = VAR(boot_df, exog=exog).fit(maxlags=fitted.k_ar)
                    boot_fc = boot_model.forecast(boot_df.values[-fitted.k_ar:], steps=horizon, exog_future=exog_fc)
                
                boot_fcs.append(boot_fc)
            except:
                continue
        
        if len(boot_fcs) < 50:
            resid_std = np.std(residuals, axis=0)
            fc_std = resid_std * np.sqrt(np.arange(1, horizon + 1))[:, None]
            return -fc_std * 1.96, fc_std * 1.96
        
        boot_array = np.array(boot_fcs)
        return np.percentile(boot_array, 2.5, axis=0), np.percentile(boot_array, 97.5, axis=0)
    
    def _prep_breaks(self, index, breaks):
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
        if exog is None:
            return None
        if len(exog.shape) == 1:
            exog = exog.reshape(-1, 1)
        
        extended = []
        for h in range(1, horizon + 1):
            row = []
            for i in range(exog.shape[1]):
                if np.all(np.isin(exog[:, i], [0, 1])):
                    row.append(exog[-1, i])
                else:
                    row.append(exog[-1, i] + (exog[-1, i] - exog[-2, i] if len(exog) > 1 else 1))
            extended.append(row)
        return np.array(extended)


# ===============================
# Advanced Forecasting (SAME AS BEFORE - truncated for brevity)
# ===============================

class AdvancedForecastingV3:
    """V3.2 forecasting with VAR system support"""
    
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.var_forecaster = VARSystemForecaster(config) if config.use_var_systems else None
    
    def forecast_univariate(self, series, horizon, structural_breaks=None, metric_info=None):
        """[Same as before - ARIMA + ETS + Linear + CV]"""
        models = []
        
        arima = self._fit_arima(series, horizon, structural_breaks)
        if arima: models.append(arima)
        
        if HAVE_STATSMODELS and len(series) > 20:
            ets = self._fit_ets(series, horizon)
            if ets: models.append(ets)
        
        linear = self._fit_linear(series, horizon)
        if linear: models.append(linear)
        
        if not models:
            return self._fallback(series, horizon)
        
        if len(models) > 1:
            cv_errors = self._cv(series, models)
            weights = self._calc_weights(cv_errors)
            combined = self._combine(models, weights, series)
        else:
            combined = models[0]
        
        if metric_info:
            combined = self._constrain(combined, series, metric_info)
        
        return combined
    
    # [Include all helper methods from previous V3.2: _fit_arima, _fit_ets, _fit_linear, etc.]
    # [Truncated here for brevity - use exact same methods as ITL1 V3.2 above]


# ===============================
# Derived Metrics (SAME AS BEFORE)
# ===============================

class DerivedMetricsV3:
    """[Same Monte Carlo derived metrics as before]"""
    
    @staticmethod
    def calculate_all_derived(data: pd.DataFrame, region_code: str, year: int) -> List[Dict]:
        """[Same as before - productivity, employment rate, income per worker]"""
        # [Use exact same code from ITL1 V3.2 above]
        pass  # Placeholder - use full implementation from above


# ===============================
# Main Pipeline (ENHANCED WITH RECONCILIATION)
# ===============================

class InstitutionalForecasterV3:
    """V3.2 with top-down macro reconciliation"""
    
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.data_manager = DataManagerV3(config)
        self.forecaster = AdvancedForecastingV3(config)
        self.derived = DerivedMetricsV3()
        
        # NEW: Load macro anchors
        if config.use_macro_anchoring and HAVE_DUCKDB:
            self.macro_manager = MacroAnchorManager(config.duckdb_path)
            if self.macro_manager.has_anchors():
                self.reconciler = TopDownReconciler(config, self.macro_manager)
                logger.info("✓ Macro anchors loaded - reconciliation enabled")
            else:
                self.reconciler = None
                logger.warning("⚠️ No macro anchors - reconciliation disabled")
        else:
            self.macro_manager = None
            self.reconciler = None
    
    def run(self) -> pd.DataFrame:
        """Execute pipeline with reconciliation"""
        logger.info("="*60)
        logger.info("ITL1 FORECASTING ENGINE V3.2 (Top-Down)")
        logger.info("="*60)
        
        historical = self.data_manager.load_all_data()
        tasks = self._identify_tasks(historical)
        logger.info(f"Tasks: {len(tasks)}")
        
        # Generate forecasts (VAR + univariate)
        results = self._run_forecasts_with_systems(historical, tasks)
        
        # Combine historical + forecasts
        all_data = pd.concat([historical, pd.DataFrame(results)], ignore_index=True)
        
        # NEW: Apply top-down reconciliation BEFORE derived metrics
        if self.reconciler:
            all_data = self.reconciler.reconcile(all_data)
        
        # Calculate derived metrics (after reconciliation)
        logger.info("Calculating derived indicators...")
        derived = self._calculate_derived(all_data)
        all_data = pd.concat([all_data, pd.DataFrame(derived)], ignore_index=True)
        
        logger.info("Saving outputs...")
        self._save_outputs(all_data)
        
        logger.info("✅ ITL1 forecasting V3.2 completed")
        return all_data
    
    def _identify_tasks(self, data: pd.DataFrame) -> List[Dict]:
        """[Same as before]"""
        tasks = []
        for (region_code, region, metric), group in data.groupby(['region_code', 'region', 'metric']):
            if len(group) < self.config.min_history_years:
                continue
            
            last_year = group['year'].max()
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
    
    def _run_forecasts_with_systems(self, data: pd.DataFrame, tasks: List[Dict]) -> List[Dict]:
        """[Same as before - VAR systems + univariate]"""
        results = []
        processed = set()
        
        tasks_by_region = {}
        for task in tasks:
            rc = task['region_code']
            if rc not in tasks_by_region:
                tasks_by_region[rc] = []
            tasks_by_region[rc].append(task)
        
        for region_code, region_tasks in tasks_by_region.items():
            logger.info(f"\nRegion: {region_code}")
            
            region_metrics = {t['metric'] for t in region_tasks}
            
            # Try VAR systems
            for system_name, system_metrics in self.config.var_systems.items():
                available = [m for m in system_metrics if m in region_metrics]
                
                if len(available) >= 2 and self.config.use_var_systems:
                    logger.info(f"  VAR system: {available}")
                    
                    horizons = [t['horizon'] for t in region_tasks if t['metric'] in available]
                    common_horizon = min(horizons) if horizons else 0
                    
                    if common_horizon > 0:
                        breaks = [b for b in self.config.structural_breaks 
                                 if any(t['last_year'] - 20 <= b['year'] <= t['last_year'] 
                                       for t in region_tasks if t['metric'] in available)]
                        
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
                                        'source': 'model'
                                    })
            
            # Univariate for remaining
            for task in region_tasks:
                if (task['region_code'], task['metric']) in processed:
                    continue
                
                logger.info(f"  {task['metric']} (univariate)")
                
                series_data = data[
                    (data['region_code'] == task['region_code']) &
                    (data['metric'] == task['metric'])
                ].sort_values('year')
                
                series = pd.Series(series_data['value'].values, index=series_data['year'].values, name=task['metric'])
                
                use_log = self.config.use_log_transform.get(task['metric'], False)
                working = np.log(series) if use_log and (series > 0).all() else series
                
                breaks = [b for b in self.config.structural_breaks 
                         if task['last_year'] - 20 <= b['year'] <= task['last_year']]
                
                try:
                    fc = self.forecaster.forecast_univariate(working, task['horizon'], breaks, task['metric_info'])
                    
                    if use_log:
                        fc['values'] = np.exp(fc['values'])
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
                            'source': 'model'
                        })
                except Exception as e:
                    logger.error(f"Failed {task['region_code']}-{task['metric']}: {e}")
        
        return results
    
    def _calculate_derived(self, data: pd.DataFrame) -> List[Dict]:
        """[Same as before]"""
        derived = []
        for region_code in data['region_code'].unique():
            for year in data['year'].unique():
                try:
                    year_results = self.derived.calculate_all_derived(data, region_code, int(year))
                    for r in year_results:
                        r['region'] = data[data['region_code'] == region_code]['region'].iloc[0]
                        derived.append(r)
                except Exception as e:
                    logger.error(f"Derived failed {region_code}-{year}: {e}")
        return derived
    
    def _save_outputs(self, data: pd.DataFrame):
        """Save outputs (same as before)"""
        
        # Long format
        long_path = self.config.output_dir / "forecast_v3_long.csv"
        data.to_csv(long_path, index=False)
        logger.info(f"✓ Long: {long_path}")
        
        # Wide format
        wide = data.pivot_table(index=['region', 'region_code', 'metric'], 
                               columns='year', values='value', aggfunc='first').reset_index()
        wide.columns.name = None
        wide_path = self.config.output_dir / "forecast_v3_wide.csv"
        wide.to_csv(wide_path, index=False)
        logger.info(f"✓ Wide: {wide_path}")
        
        # CIs
        ci_data = data[['region_code', 'metric', 'year', 'value', 'ci_lower', 'ci_upper']].copy()
        ci_data['ci_width'] = ci_data['ci_upper'] - ci_data['ci_lower']
        ci_path = self.config.output_dir / "confidence_intervals_v3.csv"
        ci_data.to_csv(ci_path, index=False)
        logger.info(f"✓ CIs: {ci_path}")
        
        # Metadata
        metadata = {
            'run_timestamp': datetime.now().isoformat(),
            'version': '3.2_top_down',
            'enhancements': 'VAR/VECM + Top-down reconciliation to UK totals',
            'config': {
                'target_year': self.config.target_year,
                'var_enabled': self.config.use_var_systems,
                'macro_anchoring': self.config.use_macro_anchoring,
                'reconciliation_method': self.config.reconciliation_method
            },
            'data_summary': {
                'regions': data['region_code'].nunique(),
                'metrics': data['metric'].nunique(),
                'total_obs': len(data),
                'forecast_obs': len(data[data['data_type'] == 'forecast'])
            },
            'reconciliation_summary': {}
        }
        
        # Add reconciliation details
        if hasattr(data, 'attrs') and 'reconciliation_log' in data.attrs:
            recon = data.attrs['reconciliation_log']
            if recon:
                metadata['reconciliation_summary'] = {
                    'adjustments': len(recon),
                    'avg_scale_factor': np.mean([r['scale_factor'] for r in recon]),
                    'max_deviation_pct': max([r['deviation_pct'] for r in recon])
                }
        
        metadata_path = self.config.output_dir / "metadata_v3.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"✓ Metadata: {metadata_path}")
        
        # DuckDB
        if HAVE_DUCKDB:
            try:
                data_copy = data.copy()
                
                if "metric" in data_copy.columns and "metric_id" not in data_copy.columns:
                    data_copy["metric_id"] = data_copy["metric"]
                if "year" in data_copy.columns and "period" not in data_copy.columns:
                    data_copy["period"] = data_copy["year"]
                
                data_copy["region_level"] = "ITL1"
                data_copy["forecast_run_date"] = datetime.now().date()
                data_copy["forecast_version"] = "3.2_top_down"
                
                cols = ["region_code", "region_name", "region_level", "metric_id", "period", "value",
                       "unit", "freq", "data_type", "ci_lower", "ci_upper", "forecast_run_date", "forecast_version"]
                data_flat = data_copy[[c for c in cols if c in data_copy.columns]].reset_index(drop=True)
                
                con = duckdb.connect(str(self.config.duckdb_path))
                con.execute("CREATE SCHEMA IF NOT EXISTS gold")
                con.register("itl1_df", data_flat)
                
                con.execute("CREATE OR REPLACE TABLE gold.itl1_forecast AS SELECT * FROM itl1_df")
                con.execute("""
                    CREATE OR REPLACE VIEW gold.itl1_forecast_only AS
                    SELECT * FROM gold.itl1_forecast WHERE data_type = 'forecast'
                """)
                con.execute("""
                    CREATE OR REPLACE VIEW gold.itl1_latest AS
                    SELECT * FROM gold.itl1_forecast
                    WHERE forecast_run_date = (SELECT MAX(forecast_run_date) FROM gold.itl1_forecast)
                """)
                
                con.close()
                logger.info(f"✓ DuckDB: gold.itl1_forecast ({len(data_flat)} rows)")
            except Exception as e:
                logger.warning(f"DuckDB save failed: {e}")


# ===============================
# Entry Point
# ===============================

def main():
    """Run ITL1 forecasting with macro anchoring"""
    
    try:
        config = ForecastConfig()
        
        logger.info("="*70)
        logger.info("ITL1 FORECAST V3.2 - VAR + TOP-DOWN RECONCILIATION")
        logger.info("="*70)
        logger.info(f"  Silver: {config.silver_path}")
        logger.info(f"  Target: {config.target_year}")
        logger.info(f"  VAR: {config.use_var_systems}")
        logger.info(f"  Macro anchoring: {config.use_macro_anchoring}")
        
        # Check for macro anchors
        if config.use_macro_anchoring:
            if not HAVE_DUCKDB:
                logger.warning("⚠️ DuckDB unavailable - macro anchoring disabled")
                config.use_macro_anchoring = False
            elif not config.duckdb_path.exists():
                logger.warning("⚠️ DuckDB not found - macro anchoring disabled")
                config.use_macro_anchoring = False
            else:
                try:
                    temp_mgr = MacroAnchorManager(config.duckdb_path)
                    if not temp_mgr.has_anchors():
                        logger.warning("⚠️ No macro forecasts found - run macro V3.2 first")
                        logger.warning("⚠️ Continuing without reconciliation")
                        config.use_macro_anchoring = False
                except Exception as e:
                    logger.warning(f"⚠️ Could not load macro anchors: {e}")
                    config.use_macro_anchoring = False
        
        forecaster = InstitutionalForecasterV3(config)
        results = forecaster.run()
        
        logger.info("="*70)
        logger.info("✅ ITL1 FORECASTING COMPLETED")
        logger.info(f"📊 Regions: {results['region_code'].nunique()}")
        logger.info(f"📊 Metrics: {results['metric'].nunique()}")
        logger.info(f"📊 Forecasts: {len(results[results['data_type']=='forecast'])}")
        
        if config.use_macro_anchoring and hasattr(results, 'attrs'):
            recon_log = results.attrs.get('reconciliation_log', [])
            logger.info(f"📊 Reconciliation adjustments: {len(recon_log)}")
        
        logger.info("="*70)
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()