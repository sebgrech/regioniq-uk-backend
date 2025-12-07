#!/usr/bin/env python3
"""
Region IQ - UK Macro Forecasting Engine V3.2 (VAR/VECM Fixed)
==============================================================

V3.2: VAR/VECM System Enhancement with Debug Logging
- Enterprise-grade forecasting for UK-level macro indicators
- VAR/VECM for cross-metric coherence (GVA ↔ Employment)
- Structural breaks, bootstrap CIs, growth constraints
- FIXED: VAR silent failure with comprehensive debug logging

Inputs:
    data/silver/uk_macro_history.csv

Outputs:
    data/forecast/uk_macro_forecast_long.csv
    gold.uk_macro_forecast (DuckDB)

Author: Region IQ
Version: 3.2-fixed
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('macro_forecast_v3.log'),
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
    logger.warning(f"Statistical packages not available: {e}")
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
class MacroForecastConfig:
    """Configuration for UK macro forecasting"""
    
    silver_path: Path = Path("data/silver/uk_macro_history.csv")
    duckdb_path: Path = Path("data/lake/warehouse.duckdb")
    use_duckdb: bool = False
    
    output_dir: Path = Path("data/forecast")
    cache_dir: Path = Path("data/cache")
    
    target_year: int = 2050
    min_history_years: int = 10
    confidence_levels: List[float] = field(default_factory=lambda: [0.80, 0.95])
    
    max_arima_order: int = 2
    max_var_lags: int = 3
    use_log_transform: Dict[str, bool] = None
    
    use_var_systems: bool = True
    var_systems: Dict[str, List[str]] = field(default_factory=lambda: {
        'gva_employment': ['uk_nominal_gva_mn_gbp', 'uk_emp_total_jobs']
    })
    var_max_horizon: int = 15  # Cap VAR at 15 years to prevent instability
    var_bootstrap_samples: int = 300
    
    metric_definitions: Dict[str, Dict] = None
    structural_breaks: List[Dict] = None
    
    cv_min_train_size: int = 15
    cv_test_windows: int = 3
    cv_horizon: int = 2
    
    n_bootstrap: int = 200
    cache_enabled: bool = True
    
    enforce_non_negative: bool = True
    enforce_monotonic_population: bool = True
    growth_rate_cap_percentiles: Tuple[float, float] = (2, 98)
    
    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if self.metric_definitions is None:
            self.metric_definitions = {
                'uk_population_total': {'unit': 'persons', 'transform': 'log', 'monotonic': True},
                'uk_gdhi_total_mn_gbp': {'unit': 'GBP_m', 'transform': 'log', 'monotonic': False},
                'uk_gdhi_per_head_gbp': {'unit': 'GBP', 'transform': 'log', 'monotonic': False},
                'uk_nominal_gva_mn_gbp': {'unit': 'GBP_m', 'transform': 'log', 'monotonic': False},
                'uk_emp_total_jobs': {'unit': 'jobs', 'transform': 'log', 'monotonic': False}
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
        assert self.min_history_years >= 5
        assert self.n_bootstrap >= 100
        logger.info("Macro V3.2 Configuration validated")


# ===============================
# Data Management
# ===============================

class MacroDataManager:
    """Data management for UK macro data"""
    
    def __init__(self, config: MacroForecastConfig):
        self.config = config
        
    def load_all_data(self) -> pd.DataFrame:
        """Load UK macro historical data"""
        cache_key = self._get_cache_key()
        if self.config.cache_enabled and self._cache_exists(cache_key):
            logger.info("Loading from cache")
            return self._load_from_cache(cache_key)
        
        if self.config.use_duckdb:
            unified = self._load_from_duckdb()
        else:
            unified = self._load_from_csv()
        
        unified = self._standardize_columns(unified)
        unified = self._handle_outliers(unified)
        
        if self.config.cache_enabled:
            self._save_to_cache(unified, cache_key)
        
        logger.info(f"Loaded {len(unified)} macro observations")
        logger.info(f"Metrics: {unified['metric'].nunique()}")
        
        return unified
    
    def _load_from_csv(self) -> pd.DataFrame:
        """Load from silver CSV"""
        logger.info(f"Loading from {self.config.silver_path}")
        df = pd.read_csv(self.config.silver_path)
        
        # Ensure UK-only
        if 'region_level' in df.columns:
            df = df[df['region_level'] == 'UK']
        
        expected_cols = ['region_code', 'metric_id', 'period', 'value']
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
        """Load from DuckDB"""
        try:
            con = duckdb.connect(str(self.config.duckdb_path), read_only=True)
            df = con.execute("SELECT * FROM silver.uk_macro_history").fetchdf()
            con.close()
            return df
        except Exception as e:
            logger.error(f"DuckDB load failed: {e}")
            return self._load_from_csv()
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names"""
        rename_map = {
            'metric_id': 'metric',
            'period': 'year',
            'region_name': 'region'
        }
        
        for old, new in rename_map.items():
            if old in df.columns and new not in df.columns:
                df[new] = df[old]
        
        # CRITICAL FIX: Convert to regular int, not Int64
        df['year'] = pd.to_numeric(df['year'], errors='coerce').astype(int)
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        if 'data_type' not in df.columns:
            df['data_type'] = 'historical'
        
        df = df.dropna(subset=['year', 'value'])
        
        if self.config.enforce_non_negative:
            df = df[df['value'] > 0]
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers via winsorization"""
        for metric in df['metric'].unique():
            group = df[df['metric'] == metric]
            if len(group) < 5:
                continue
            
            values = group['value'].values
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            
            outliers = (values < lower_bound) | (values > upper_bound)
            if outliers.any():
                logger.info(f"Winsorizing {outliers.sum()} outliers in {metric}")
                df.loc[group[outliers].index, 'value'] = np.clip(values[outliers], lower_bound, upper_bound)
        
        return df
    
    def _get_cache_key(self) -> str:
        """Build cache key from underlying file mtime"""
        hasher = hashlib.md5()
        path = self.config.duckdb_path if self.config.use_duckdb else self.config.silver_path
        if path.exists():
            hasher.update(str(path.stat().st_mtime).encode())
        return hasher.hexdigest()
    
    def _cache_exists(self, key: str) -> bool:
        return (self.config.cache_dir / f"macro_{key}.pkl").exists()
    
    def _load_from_cache(self, key: str) -> pd.DataFrame:
        with open(self.config.cache_dir / f"macro_{key}.pkl", 'rb') as f:
            return pickle.load(f)
    
    def _save_to_cache(self, df: pd.DataFrame, key: str):
        with open(self.config.cache_dir / f"macro_{key}.pkl", 'wb') as f:
            pickle.dump(df, f)


# ===============================
# VAR System Forecaster (FIXED)
# ===============================

class VARSystemForecaster:
    """VAR/VECM for cross-metric coherence with comprehensive debugging"""
    
    def __init__(self, config: MacroForecastConfig):
        self.config = config
    
    def forecast_system(
        self,
        data: pd.DataFrame,
        metrics: List[str],
        horizon: int,
        structural_breaks: Optional[Sequence[Dict]] = None
    ) -> Optional[Dict]:
        """Forecast multiple UK metrics together using VAR/VECM with debug logging"""
        if not HAVE_STATSMODELS:
            print("❌ VAR: statsmodels not available")
            logger.debug("VAR: statsmodels not available")
            return None
        
        try:
            print(f"\n{'='*60}")
            print(f"VAR SYSTEM DEBUG")
            print(f"{'='*60}")
            
            # Build multivariate dataframe
            series_dict = {}
            for metric in metrics:
                metric_data = data[data['metric'] == metric].sort_values('year')
                
                if metric_data.empty:
                    print(f"❌ VAR: No data for {metric}")
                    return None
                
                values = metric_data['value'].values
                print(f"✓ {metric}: {len(values)} values, range {values.min():.0f}-{values.max():.0f}")
                
                if self.config.use_log_transform.get(metric, False) and (values > 0).all():
                    values = np.log(values)
                    print(f"  → Log transformed")
                
                # CRITICAL: Use regular int index
                series_dict[metric] = pd.Series(values, index=metric_data['year'].values.astype(int))
            
            system_df = pd.DataFrame(series_dict).dropna()
            
            print(f"\n✓ System DF built:")
            print(f"  Shape: {system_df.shape}")
            print(f"  Index range: {system_df.index.min()}-{system_df.index.max()}")
            print(f"  Required minimum: {self.config.min_history_years}")
            
            if len(system_df) < self.config.min_history_years:
                print(f"❌ VAR: Insufficient observations ({len(system_df)} < {self.config.min_history_years})")
                logger.debug(f"VAR: Insufficient observations ({len(system_df)})")
                return None
            
            print(f"✓ Sufficient history")
            
            # Test cointegration
            print(f"\n→ Testing cointegration...")
            coint_result = self._test_cointegration(system_df)
            print(f"  Cointegrated: {coint_result['cointegrated']}, Rank: {coint_result['rank']}")
            
            # Prepare break dummies
            print(f"\n→ Preparing structural breaks...")
            exog = self._prepare_break_dummies(system_df.index, structural_breaks) if structural_breaks else None
            if exog is not None:
                print(f"  Exog shape: {exog.shape}")
            else:
                print(f"  No exogenous variables")
            
            # Fit model
            print(f"\n→ Fitting VAR model...")
            if coint_result['cointegrated'] and coint_result['rank'] > 0:
                print(f"  Using VECM (rank={coint_result['rank']})")
                model = VECM(system_df, k_ar_diff=min(self.config.max_var_lags, 3), 
                             coint_rank=coint_result['rank'], deterministic='ci', exog=exog)
                fitted = model.fit()
                method_name = f'VECM(r={coint_result["rank"]})'
                
                exog_fc = self._extend_exog(exog, system_df.index, horizon) if exog is not None else None
                forecast_values = fitted.predict(steps=horizon, exog_fc=exog_fc)
                
            else:
                print(f"  Using VAR")
                model = VAR(system_df, exog=exog)
                
                print(f"  → Selecting optimal lags...")
                lag_results = model.select_order(maxlags=self.config.max_var_lags)
                try:
                    optimal_lags = lag_results.selected_orders['aic']
                except (AttributeError, KeyError):
                    optimal_lags = lag_results.aic
                optimal_lags = max(1, min(optimal_lags, self.config.max_var_lags))
                print(f"  Optimal lags: {optimal_lags}")
                
                print(f"  → Fitting VAR({optimal_lags})...")
                fitted = model.fit(maxlags=optimal_lags)
                method_name = f'VAR({optimal_lags})'
                
                exog_fc = self._extend_exog(exog, system_df.index, horizon) if exog is not None else None
                forecast_values = fitted.forecast(system_df.values[-fitted.k_ar:], steps=horizon, exog_future=exog_fc)
            
            print(f"✓ VAR model fitted successfully: {method_name}")
            print(f"  Forecast shape: {forecast_values.shape}")
            
            # Bootstrap CIs
            print(f"\n→ Computing bootstrap CIs...")
            ci_lower, ci_upper = self._bootstrap_ci(fitted, system_df, horizon, exog, exog_fc)
            print(f"  CI computed")
            
            last_year = int(system_df.index[-1])
            forecast_years = list(range(last_year + 1, last_year + horizon + 1))
            
            # Double-check we never forecast the last historical year
            forecast_years = [y for y in forecast_years if y > last_year]
            h = len(forecast_years)
            if h != horizon:
                logger.warning(f"  VAR: Adjusted horizon from {horizon} to {h} to avoid duplicates")
                forecast_values = forecast_values[:h]
                ci_lower = ci_lower[:h]
                ci_upper = ci_upper[:h]
            
            # Transform back
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
            
            print(f"✓ VAR complete: {len(metrics)} metrics forecasted")
            print(f"{'='*60}\n")
            
            logger.info(f"  VAR: {len(metrics)} metrics forecasted")
            return results
            
        except Exception as e:
            print(f"\n❌ VAR FAILED WITH EXCEPTION:")
            print(f"   {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")
            logger.debug(f"VAR failed: {e}")
            return None
    
    def _test_cointegration(self, data: pd.DataFrame) -> Dict:
        """Test cointegration with error handling"""
        if len(data) < 30:
            return {'cointegrated': False, 'rank': 0}
        try:
            result = coint_johansen(data, det_order=0, k_ar_diff=2)
            rank = sum(result.lr1 > result.cvt[:, 1])
            return {'cointegrated': rank > 0, 'rank': rank}
        except Exception as e:
            print(f"  ⚠ Cointegration test failed: {e}")
            return {'cointegrated': False, 'rank': 0}
    
    def _bootstrap_ci(self, fitted, data, horizon, exog, exog_fc, n_boot=50):
        """Analytical confidence intervals for VAR (simplified for reliability)"""
        # Use analytical CIs instead of bootstrap for VAR (more reliable)
        
        # Get residuals and ensure numpy array
        if hasattr(fitted, 'resid'):
            residuals = fitted.resid
        else:
            residuals = data.values[fitted.k_ar:] - fitted.fittedvalues
        
        # Convert to numpy array if pandas
        if hasattr(residuals, 'values'):
            residuals = residuals.values
        
        # Ensure 2D shape
        residuals = np.atleast_2d(residuals)
        if residuals.shape[0] == 1 and residuals.shape[1] > 1:
            residuals = residuals.T
        
        n_vars = residuals.shape[1]
        
        # Compute residual covariance
        resid_cov = np.cov(residuals.T)
        if n_vars == 1 or resid_cov.ndim == 0:
            resid_cov = np.array([[resid_cov]]) if resid_cov.ndim == 0 else resid_cov.reshape(1, 1)
        
        # Standard errors grow with horizon
        horizon_factors = np.sqrt(np.arange(1, horizon + 1))  # shape (horizon,)
        
        # Create CIs for each variable
        ci_lower = np.zeros((horizon, n_vars))
        ci_upper = np.zeros((horizon, n_vars))
        
        for i in range(n_vars):
            se = np.sqrt(resid_cov[i, i])  # Standard error for variable i
            ci_width = 1.96 * se * horizon_factors  # shape (horizon,)
            ci_lower[:, i] = -ci_width
            ci_upper[:, i] = ci_width
        
        print(f"  ✓ Analytical CIs computed (residual SE: {np.sqrt(np.diag(resid_cov))})")
        
        return ci_lower, ci_upper
    
    def _prepare_break_dummies(self, index, breaks):
        """Prepare structural break dummies"""
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
        """Extend exogenous variables"""
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
                    inc = exog[-1, i] - exog[-2, i] if len(exog) > 1 else 1
                    row.append(exog[-1, i] + inc)
            extended.append(row)
        return np.array(extended)


# ===============================
# Advanced Forecasting - COMPLETE
# ===============================

class AdvancedMacroForecaster:
    """Macro forecasting with VAR support"""
    
    def __init__(self, config: MacroForecastConfig):
        self.config = config
        self.var_forecaster = VARSystemForecaster(config) if config.use_var_systems else None
    
    def forecast_univariate(
        self,
        series: pd.Series,
        horizon: int,
        structural_breaks: Optional[Sequence[Dict]] = None,
        metric_info: Dict = None
    ) -> Dict:
        """Univariate forecasting with ensemble"""
        
        models = []
        
        arima = self._fit_arima_with_breaks(series, horizon, structural_breaks)
        if arima:
            models.append(arima)
        
        if HAVE_STATSMODELS and len(series) > 20:
            ets = self._fit_ets_auto(series, horizon)
            if ets:
                models.append(ets)
        
        linear = self._fit_linear(series, horizon)
        if linear:
            models.append(linear)
        
        if not models:
            return self._fallback_forecast(series, horizon)
        
        if len(models) > 1:
            cv_errors = self._cross_validate(series, models)
            weights = self._calculate_weights(cv_errors)
            combined = self._combine_forecasts(models, weights, series)
        else:
            combined = models[0]
        
        if metric_info:
            combined = self._apply_constraints(combined, series, metric_info)
        
        return combined
    
    def _fit_arima_with_breaks(
        self,
        series: pd.Series,
        horizon: int,
        structural_breaks: Optional[Sequence[Dict]] = None
    ) -> Optional[Dict]:
        """ARIMA with structural breaks"""
        if not HAVE_STATSMODELS:
            return None
        
        try:
            exog = self._prep_breaks(series, structural_breaks) if structural_breaks else None
            
            best_model, best_aic, best_order = None, np.inf, None
            
            adf_p = adfuller(series, autolag='AIC')[1]
            d = 1 if adf_p > 0.05 else 0
            
            for p in range(self.config.max_arima_order + 1):
                for q in range(self.config.max_arima_order + 1):
                    if p == q == d == 0:
                        continue
                    
                    try:
                        model = ARIMA(series, order=(p, d, q), exog=exog)
                        fitted = model.fit(method_kwargs={"warn_convergence": False})
                        
                        n, k = len(series), p + q + d + 1
                        aicc = fitted.aic + (2 * k * (k + 1)) / (n - k - 1)
                        
                        if aicc < best_aic:
                            best_aic, best_model, best_order = aicc, fitted, (p, d, q)
                    except Exception:
                        continue
            
            if best_model is None:
                return None
            
            exog_fc = self._extend_exog(exog, series.index, horizon) if exog is not None else None
            fc_obj = best_model.get_forecast(steps=horizon, exog=exog_fc)
            
            residuals = pd.Series(best_model.resid)
            resid_tests = self._test_residuals(residuals)
            
            if not resid_tests.get('normal', True):
                ci = self._bootstrap_ci_arima(best_model, series, horizon, exog, exog_fc, best_order)
            else:
                ci = fc_obj.conf_int(alpha=0.05)
            
            last_year = int(series.index[-1])
            forecast_years = list(range(last_year + 1, last_year + horizon + 1))
            
            return {
                'method': f'ARIMA{best_order}',
                'values': fc_obj.predicted_mean.values,
                'years': forecast_years,
                'ci_lower': ci.iloc[:, 0].values if hasattr(ci, 'iloc') else ci[:, 0],
                'ci_upper': ci.iloc[:, 1].values if hasattr(ci, 'iloc') else ci[:, 1],
                'aic': best_aic,
                'fit_function': lambda s, h: self._arima_refit(s, h, best_order, structural_breaks)
            }
        except Exception as e:
            logger.debug(f"ARIMA failed: {e}")
            return None
    
    def _fit_ets_auto(self, series: pd.Series, horizon: int) -> Optional[Dict]:
        """ETS with auto trend selection"""
        if not HAVE_STATSMODELS:
            return None
        
        try:
            best_model, best_aic, best_config = None, np.inf, None
            
            for trend in ['add', 'mul', None]:
                for damped in [True, False]:
                    if trend is None and damped:
                        continue
                    
                    try:
                        model = ExponentialSmoothing(
                            series,
                            trend=trend,
                            damped_trend=damped,
                            seasonal=None
                        )
                        fitted = model.fit()
                        
                        if fitted.aic < best_aic:
                            best_aic, best_model, best_config = fitted.aic, fitted, f"ETS({trend},{damped})"
                    except Exception:
                        continue
            
            if best_model is None:
                return None
            
            forecast = best_model.forecast(steps=horizon)
            sigma = (series - best_model.fittedvalues).std()
            t_stat = stats.t.ppf(0.975, len(series) - 1) if len(series) < 30 else 1.96
            
            ci_lower = forecast - t_stat * sigma * np.sqrt(range(1, horizon + 1))
            ci_upper = forecast + t_stat * sigma * np.sqrt(range(1, horizon + 1))
            
            last_year = int(series.index[-1])
            forecast_years = list(range(last_year + 1, last_year + horizon + 1))
            
            return {
                'method': best_config,
                'values': forecast.values,
                'years': forecast_years,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'aic': best_aic,
                'fit_function': lambda s, h: self._ets_refit(s, h, best_model)
            }
        except Exception as e:
            logger.debug(f"ETS failed: {e}")
            return None
    
    def _fit_linear(self, series: pd.Series, horizon: int) -> Optional[Dict]:
        """Linear trend"""
        try:
            X = np.arange(len(series)).reshape(-1, 1)
            y = series.values
            
            if HAVE_STATSMODELS:
                X_sm = sm.add_constant(X)
                model = sm.OLS(y, X_sm).fit()
                
                X_future = sm.add_constant(np.arange(len(series), len(series) + horizon).reshape(-1, 1))
                pred = model.get_prediction(X_future)
                
                forecast = pred.predicted_mean
                ci = pred.conf_int(alpha=0.05)
            else:
                coeffs = np.polyfit(X.flatten(), y, 1)
                X_future = np.arange(len(series), len(series) + horizon)
                forecast = np.polyval(coeffs, X_future)
                
                sigma = (y - np.polyval(coeffs, X.flatten())).std()
                ci_width = 1.96 * sigma
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
                'fit_function': lambda s, h: self._linear_refit(s, h)
            }
        except Exception as e:
            logger.debug(f"Linear failed: {e}")
            return None
    
    def _cross_validate(self, series: pd.Series, models: List[Dict]) -> Dict:
        """Expanding window CV"""
        cv_errors = {i: [] for i in range(len(models))}
        min_train = max(self.config.cv_min_train_size, len(series) // 2)
        
        for train_end in range(min_train, len(series) - self.config.cv_horizon + 1):
            train = series.iloc[:train_end]
            test = series.iloc[train_end:min(train_end + self.config.cv_horizon, len(series))]
            
            for i, model_info in enumerate(models):
                try:
                    if 'fit_function' in model_info:
                        fc = model_info['fit_function'](train, len(test))
                        error = np.mean(
                            np.abs(fc - test.values) /
                            (np.abs(fc) + np.abs(test.values) + 1e-10)
                        )
                        cv_errors[i].append(error)
                except Exception:
                    cv_errors[i].append(np.inf)
        
        return cv_errors
    
    def _calculate_weights(self, cv_errors: Dict) -> np.ndarray:
        """Calculate ensemble weights"""
        mean_errors = []
        for i in sorted(cv_errors.keys()):
            errors = [e for e in cv_errors[i] if e != np.inf]
            mean_errors.append(np.mean(errors) if errors else np.inf)
        
        if all(e == np.inf for e in mean_errors):
            return np.ones(len(mean_errors)) / len(mean_errors)
        
        weights = 1.0 / (np.array(mean_errors) + 1e-10)
        weights[np.isinf(weights)] = 0
        
        return weights / weights.sum() if weights.sum() > 0 else np.ones(len(weights)) / len(weights)
    
    def _combine_forecasts(self, models: List[Dict], weights: np.ndarray, series: pd.Series) -> Dict:
        """Combine forecasts with CV-derived CIs"""
        combined_values = np.zeros(len(models[0]['values']))
        for w, m in zip(weights, models):
            combined_values += w * m['values']
        
        cv_residuals = []
        min_train = max(self.config.cv_min_train_size, len(series) // 2)
        
        for train_end in range(min_train, len(series)):
            train = series.iloc[:train_end]
            test_val = series.iloc[train_end] if train_end < len(series) else None
            
            if test_val is not None:
                ensemble_fc = 0.0
                for w, m in zip(weights, models):
                    if 'fit_function' in m:
                        try:
                            fc = m['fit_function'](train, 1)
                            ensemble_fc += w * fc[0]
                        except Exception:
                            pass
                
                if ensemble_fc > 0:
                    cv_residuals.append(test_val - ensemble_fc)
        
        if cv_residuals:
            resid_std = np.std(cv_residuals)
            horizon_adj = np.sqrt(range(1, len(combined_values) + 1))
            ci_lower = combined_values - 1.96 * resid_std * horizon_adj
            ci_upper = combined_values + 1.96 * resid_std * horizon_adj
        else:
            ci_lower = np.zeros_like(combined_values)
            ci_upper = np.zeros_like(combined_values)
            for w, m in zip(weights, models):
                ci_lower += w * m['ci_lower']
                ci_upper += w * m['ci_upper']
        
        return {
            'method': f"Ensemble({len(models)})",
            'values': combined_values,
            'years': models[0]['years'],
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'weights': weights.tolist(),
            'component_methods': [m['method'] for m in models]
        }
    
    def _apply_constraints(self, forecast: Dict, series: pd.Series, metric_info: Dict) -> Dict:
        """Apply constraints"""
        values = forecast['values'].copy()
        
        if self.config.enforce_non_negative:
            values = np.maximum(values, 0)
            forecast['ci_lower'] = np.maximum(forecast['ci_lower'], 0)
        
        if metric_info.get('monotonic', False) and self.config.enforce_monotonic_population:
            for i in range(1, len(values)):
                values[i] = max(values[i], values[i-1])
        
        if self.config.growth_rate_cap_percentiles:
            hist_growth = series.pct_change().dropna()
            if len(hist_growth) > 5:
                p_low, p_high = self.config.growth_rate_cap_percentiles
                bounds = np.percentile(hist_growth, [p_low, p_high])
                
                last_value = series.iloc[-1]
                for i in range(len(values)):
                    prev = last_value if i == 0 else values[i-1]
                    growth = (values[i] - prev) / prev if prev > 0 else 0
                    
                    if growth < bounds[0]:
                        values[i] = prev * (1 + bounds[0])
                    elif growth > bounds[1]:
                        values[i] = prev * (1 + bounds[1])
        
        forecast['values'] = values
        forecast['constraints_applied'] = True
        return forecast
    
    def _bootstrap_ci_arima(self, model, series, horizon, exog, exog_fc, order, n_boot=None):
        """Bootstrap CIs for ARIMA"""
        if n_boot is None:
            n_boot = self.config.n_bootstrap
        
        boot_fcs = []
        for _ in range(n_boot):
            try:
                boot_resid = np.random.choice(model.resid, size=len(model.resid), replace=True)
                boot_series = pd.Series(model.fittedvalues + boot_resid, index=series.index)
                boot_model = ARIMA(boot_series, order=order, exog=exog).fit(disp=False)
                boot_fc = boot_model.forecast(steps=horizon, exog=exog_fc)
                boot_fcs.append(boot_fc.values)
            except Exception:
                continue
        
        if len(boot_fcs) < 10:
            return model.get_forecast(steps=horizon, exog=exog_fc).conf_int(alpha=0.05)
        
        boot_array = np.array(boot_fcs)
        ci_lower = np.percentile(boot_array, 2.5, axis=0)
        ci_upper = np.percentile(boot_array, 97.5, axis=0)
        return np.column_stack([ci_lower, ci_upper])
    
    def _test_residuals(self, residuals: pd.Series) -> Dict:
        """Test residuals"""
        results = {}
        if not HAVE_STATSMODELS or len(residuals) < 10:
            return results
        
        clean = residuals.dropna()
        
        try:
            lb = acorr_ljungbox(clean, lags=min(10, len(clean)//3))
            results['ljung_box_p'] = float(lb['lb_pvalue'].min())
            results['no_autocorr'] = results['ljung_box_p'] > 0.05
        except Exception:
            pass
        
        try:
            jb_stat, jb_p = jarque_bera(clean)
            results['jarque_bera_p'] = jb_p
            results['normal'] = jb_p > 0.05
        except Exception:
            pass
        
        return results
    
    def _prep_breaks(self, series, breaks):
        """Prepare break dummies"""
        if not breaks:
            return None
        dummies = []
        for b in breaks:
            if 'year' in b:
                if b.get('type') == 'level':
                    dummies.append((series.index >= b['year']).astype(int))
                else:
                    dummies.append(np.maximum(0, series.index - b['year']))
        return np.column_stack(dummies) if dummies else None
    
    def _extend_exog(self, exog, index, horizon):
        """Extend exog variables"""
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
                    inc = exog[-1, i] - exog[-2, i] if len(exog) > 1 else 1
                    row.append(exog[-1, i] + inc)
            extended.append(row)
        return np.array(extended)
    
    def _arima_refit(self, series, horizon, order, breaks=None):
        """Refit for CV"""
        try:
            exog = self._prep_breaks(series, breaks) if breaks else None
            model = ARIMA(series, order=order, exog=exog).fit(disp=False)
            exog_fc = self._extend_exog(exog, series.index, horizon) if exog is not None else None
            return model.forecast(steps=horizon, exog=exog_fc).values
        except Exception:
            return np.repeat(series.iloc[-1], horizon)
    
    def _ets_refit(self, series, horizon, ref_model):
        """Refit ETS for CV"""
        try:
            # Simplified - reuse fitted model to forecast
            return ref_model.forecast(steps=horizon).values
        except Exception:
            return np.repeat(series.iloc[-1], horizon)
    
    def _linear_refit(self, series, horizon):
        """Refit linear for CV"""
        try:
            X = np.arange(len(series))
            coeffs = np.polyfit(X, series.values, 1)
            X_future = np.arange(len(series), len(series) + horizon)
            return np.polyval(coeffs, X_future)
        except Exception:
            return np.repeat(series.iloc[-1], horizon)
    
    def _fallback_forecast(self, series: pd.Series, horizon: int) -> Dict:
        """Fallback using trend + volatility"""
        last_value = series.iloc[-1]
        trend = (series.iloc[-1] - series.iloc[-5]) / 5 if len(series) > 5 else 0
        
        values = np.array([max(last_value + trend * h, 0) for h in range(1, horizon + 1)])
        std = series.std() if len(series) > 2 else last_value * 0.1
        ci_width = 1.96 * std * np.sqrt(range(1, horizon + 1))
        
        last_year = int(series.index[-1])
        forecast_years = list(range(last_year + 1, last_year + horizon + 1))
        
        return {
            'method': 'Fallback',
            'values': values,
            'years': forecast_years,
            'ci_lower': values - ci_width,
            'ci_upper': values + ci_width
        }


# ===============================
# Derived Metrics - COMPLETE
# ===============================

class MacroDerivedMetrics:
    """Calculate UK-level derived indicators"""
    
    @staticmethod
    def calculate_productivity(
        gva: Tuple[float, float, float],
        emp: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """
        GVA per worker with robust error handling and unit correction.
        Returns GBP per job.

        gva is in millions of GBP; emp is in jobs.
        """
        gva_val, gva_low, gva_high = gva
        emp_val, emp_low, emp_high = emp

        # Basic validity
        if gva_val <= 0 or emp_val <= 0:
            return 0.0, 0.0, 0.0

        # 1) Fix suspiciously small GVA units (e.g. billions instead of millions)
        # UK nominal GVA should be > 100_000 mn GBP (i.e. > £100bn)
        if gva_val < 100_000:
            scale = 1_000.0
            gva_val *= scale
            gva_low *= scale
            gva_high *= scale

        # Enforce strictly positive bounds
        gva_low = max(gva_low, 0.0)
        gva_high = max(gva_high, gva_low)
        emp_val = max(emp_val, 1.0)
        emp_low = max(emp_low, 1.0)
        emp_high = max(emp_high, emp_low)

        # Point estimate in GBP/job
        point_estimate = (gva_val * 1_000_000.0) / emp_val

        # Standard errors from CIs
        try:
            gva_se = max((gva_high - gva_low) / (2 * 1.96), 0.0)
            emp_se = max((emp_high - emp_low) / (2 * 1.96), 0.0)
        except Exception:
            gva_se = emp_se = 0.0

        # If SEs look bad, fall back to simple +/-15% band
        if (
            not np.isfinite(gva_se) or not np.isfinite(emp_se) or
            gva_se == 0.0 or emp_se == 0.0
        ):
            se_prod = point_estimate * 0.15
            lower = max(0.0, point_estimate - 1.96 * se_prod)
            upper = point_estimate + 1.96 * se_prod
            return point_estimate, lower, upper

        # 2) Monte-Carlo with clamped positivity
        try:
            n_sim = 5000
            gva_samples = np.random.normal(gva_val * 1_000_000.0, gva_se * 1_000_000.0, n_sim)
            emp_samples = np.maximum(np.random.normal(emp_val, emp_se, n_sim), 1.0)

            prod_samples = np.maximum(gva_samples / emp_samples, 0.0)

            median = float(np.median(prod_samples))
            lower = float(np.percentile(prod_samples, 2.5))
            upper = float(np.percentile(prod_samples, 97.5))

            # Final safety clamp
            lower = max(0.0, min(lower, upper))
            median = max(0.0, median)
            upper = max(median, upper)

            return median, lower, upper

        except Exception:
            # 3) Final safe fallback: delta method around point_estimate
            var_rel = 0.0
            if gva_val > 0:
                var_rel += (gva_se / gva_val) ** 2
            if emp_val > 0:
                var_rel += (emp_se / emp_val) ** 2

            se_prod = point_estimate * np.sqrt(var_rel) if var_rel > 0 else point_estimate * 0.15
            lower = max(0.0, point_estimate - 1.96 * se_prod)
            upper = point_estimate + 1.96 * se_prod
            return point_estimate, lower, upper
    
    @staticmethod
    def calculate_all_derived(data: pd.DataFrame, year: int) -> List[Dict]:
        """Calculate all UK derived metrics for a year"""
        derived = []
        
        year_data = data[data['year'] == year]
        if year_data.empty:
            return derived
        
        metrics = {}
        for _, row in year_data.iterrows():
            metric = row['metric']
            metrics[metric] = (
                row['value'],
                row.get('ci_lower', row['value'] * 0.95),
                row.get('ci_upper', row['value'] * 1.05)
            )
        
        # Productivity: UK GVA per job
        if 'uk_nominal_gva_mn_gbp' in metrics and 'uk_emp_total_jobs' in metrics:
            prod_val, prod_low, prod_high = MacroDerivedMetrics.calculate_productivity(
                metrics['uk_nominal_gva_mn_gbp'],
                metrics['uk_emp_total_jobs']
            )
            
            derived.append({
                'region_code': 'K02000001',
                'region': 'United Kingdom',
                'year': year,
                'metric': 'uk_productivity_gbp_per_job',
                'value': prod_val,
                'ci_lower': prod_low,
                'ci_upper': prod_high,
                'method': 'derived',
                'source': 'calculated',
                'data_type': 'forecast' if year > 2024 else 'historical'
            })
        
        return derived


# ===============================
# Main Pipeline
# ===============================

class MacroForecasterV3:
    """UK Macro forecasting pipeline V3.2"""
    
    def __init__(self, config: MacroForecastConfig):
        self.config = config
        self.data_manager = MacroDataManager(config)
        self.forecaster = AdvancedMacroForecaster(config)
        self.derived = MacroDerivedMetrics()
    
    def run(self) -> pd.DataFrame:
        """Execute macro forecasting pipeline"""
        logger.info("="*60)
        logger.info("UK MACRO FORECASTING ENGINE V3.2 (FIXED)")
        logger.info("="*60)
        
        historical = self.data_manager.load_all_data()
        
        tasks = self._identify_tasks(historical)
        logger.info(f"Identified {len(tasks)} forecasting tasks")
        
        results = self._run_forecasts_with_systems(historical, tasks)
        
        logger.info("Calculating derived indicators...")
        all_data = pd.concat([historical, pd.DataFrame(results)], ignore_index=True)
        derived = self._calculate_derived(all_data)
        all_data = pd.concat([all_data, pd.DataFrame(derived)], ignore_index=True)
        
        # Reconcile per-head GDHI for forecast years
        all_data = self._reconcile_gdhi_per_head(all_data)
        
        # De-duplicate overlapping historical/forecast/derived rows
        all_data = self._deduplicate_records(all_data)
        
        # Ensure core metadata columns exist before saving
        core_cols = [
            "region_code", "region_name", "region_level",
            "metric_id", "period", "unit", "freq",
            "ci_lower", "ci_upper", "method", "data_type"
        ]
        for col in core_cols:
            if col not in all_data.columns:
                all_data[col] = None
        
        logger.info("Saving outputs...")
        self._save_outputs(all_data)
        
        logger.info("✅ Macro forecasting V3.2 completed")
        return all_data
    
    def _identify_tasks(self, data: pd.DataFrame) -> List[Dict]:
        """Identify forecasting tasks"""
        tasks = []
        
        for metric, group in data.groupby('metric'):
            if len(group) < self.config.min_history_years:
                logger.warning(f"Skipping {metric}: insufficient history")
                continue
            
            last_year = int(group['year'].max())
            
            # Start forecast from NEXT year after last historical data
            forecast_start = last_year + 1
            horizon = self.config.target_year - forecast_start + 1
            
            if horizon <= 0:
                continue
            
            metric_info = self.config.metric_definitions.get(metric, {})
            
            tasks.append({
                'metric': metric,
                'horizon': horizon,
                'last_year': last_year,
                'forecast_start': forecast_start,
                'history_length': len(group),
                'metric_info': metric_info
            })
        
        return tasks
    
    def _deduplicate_records(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicates on (region_code, metric, year) with a clear precedence:
        historical > forecast > derived/other.
        """
        data = data.copy()

        # Ensure required columns exist
        if "region_code" not in data.columns:
            data["region_code"] = "K02000001"
        if "metric" not in data.columns and "metric_id" in data.columns:
            data["metric"] = data["metric_id"]

        # Priority order: historical first, then forecast, then anything else
        priority_map = {"historical": 0, "forecast": 1, "derived": 2}
        data["__priority"] = data["data_type"].map(priority_map).fillna(3).astype(int)

        data = data.sort_values(["region_code", "metric", "year", "__priority"])
        data = data.drop_duplicates(subset=["region_code", "metric", "year"], keep="first")
        data = data.drop(columns=["__priority"])

        logger.info(f"  Deduplication complete")
        return data
    
    def _reconcile_gdhi_per_head(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Force forecast uk_gdhi_per_head_gbp to equal:
            uk_gdhi_total_mn_gbp / uk_population_total * 1_000_000
        for all years > last historical year.
        """
        data = data.copy()

        if "metric" not in data.columns:
            return data

        # Last historical year
        hist_mask = data["data_type"] == "historical"
        if not hist_mask.any():
            return data

        last_hist_year = int(data.loc[hist_mask, "year"].max())

        # Pivot to get total GDHI and population by year
        base = data.pivot_table(
            index="year",
            columns="metric",
            values="value",
            aggfunc="first"
        )

        required = {"uk_gdhi_total_mn_gbp", "uk_population_total"}
        if not required.issubset(set(base.columns)):
            return data

        # Calculate derived per-head
        base["gdhi_per_head_calc"] = (
            base["uk_gdhi_total_mn_gbp"] / base["uk_population_total"]
        ) * 1_000_000

        # Map back for forecast years only
        forecast_mask = (
            (data["metric"] == "uk_gdhi_per_head_gbp") &
            (data["year"] > last_hist_year)
        )

        if forecast_mask.any():
            data.loc[forecast_mask, "value"] = data.loc[forecast_mask, "year"].map(
                base["gdhi_per_head_calc"]
            )
            data.loc[forecast_mask, "method"] = "derived_from_total_pop"
            data.loc[forecast_mask, "source"] = "reconciled"
            logger.info(f"  Reconciled {forecast_mask.sum()} GDHI per head forecast values")

        return data
    
    def _run_forecasts_with_systems(self, data: pd.DataFrame, tasks: List[Dict]) -> List[Dict]:
        """Run forecasts with VAR system support"""
        results = []
        processed_metrics = set()
        
        logger.info(f"\n{'='*60}")
        logger.info("Processing UK macro indicators")
        logger.info(f"{'='*60}")
        
        task_metrics = {t['metric'] for t in tasks}
        
        # Try VAR systems
        for system_name, system_metrics in self.config.var_systems.items():
            available = [m for m in system_metrics if m in task_metrics]
            
            if len(available) >= 2 and self.config.use_var_systems:
                logger.info(f"  VAR system: {system_name} ({available})")
                
                horizons = [t['horizon'] for t in tasks if t['metric'] in available]
                common_horizon = min(horizons) if horizons else 0
                
                # CRITICAL: Cap VAR horizon to prevent instability
                var_horizon = min(common_horizon, self.config.var_max_horizon)
                
                if var_horizon > 0:
                    logger.info(f"  VAR horizon: {var_horizon} years (capped from {common_horizon})")
                    
                    # CRITICAL: Remove structural breaks for macro-level VAR
                    # With only 15 observations, breaks cause overparameterization
                    # (24 params / 15 obs = 0.6 obs/param, need 10+)
                    breaks = []
                    logger.info(f"  VAR without structural breaks (macro level)")
                    
                    var_results = self.forecaster.var_forecaster.forecast_system(
                        data, available, var_horizon, breaks
                    )
                    
                    if var_results:
                        for metric, forecast in var_results.items():
                            processed_metrics.add(metric)
                            
                            task = next(t for t in tasks if t['metric'] == metric)
                            
                            # Get historical years for this metric to avoid duplicates
                            hist_years = set(
                                data[data['metric'] == metric]['year'].unique()
                            )
                            
                            # Store VAR results (skip years that exist in historical)
                            for j, year in enumerate(forecast['years']):
                                if year not in hist_years:  # CRITICAL: Avoid duplicates
                                    results.append({
                                        'region_code': 'K02000001',
                                        'region': 'United Kingdom',
                                        'metric': metric,
                                        'year': year,
                                        'value': forecast['values'][j],
                                        'ci_lower': forecast['ci_lower'][j],
                                        'ci_upper': forecast['ci_upper'][j],
                                        'method': forecast['method'],
                                        'data_type': 'forecast',
                                        'source': 'model'
                                    })
                            
                            # If remaining horizon beyond VAR, fill with univariate
                            if task['horizon'] > var_horizon:
                                remaining_horizon = task['horizon'] - var_horizon
                                logger.info(f"    {metric}: Extending {remaining_horizon}y beyond VAR with univariate")
                                
                                # Get series including VAR forecast
                                series_data = data[data['metric'] == metric].sort_values('year')
                                var_forecast_df = pd.DataFrame({
                                    'year': forecast['years'],
                                    'value': forecast['values']
                                })
                                extended_series = pd.concat([
                                    series_data[['year', 'value']],
                                    var_forecast_df
                                ], ignore_index=True)
                                
                                series = pd.Series(
                                    extended_series['value'].values,
                                    index=extended_series['year'].values.astype(int),
                                    name=metric
                                )
                                
                                # Forecast remaining years
                                use_log = self.config.use_log_transform.get(metric, False)
                                if use_log and (series > 0).all():
                                    working_series = np.log(series)
                                else:
                                    working_series = series
                                    use_log = False
                                
                                univariate_fc = self.forecaster.forecast_univariate(
                                    working_series, remaining_horizon, [], task['metric_info']
                                )
                                
                                if use_log:
                                    univariate_fc['values'] = np.exp(univariate_fc['values'])
                                    univariate_fc['ci_lower'] = np.exp(univariate_fc['ci_lower'])
                                    univariate_fc['ci_upper'] = np.exp(univariate_fc['ci_upper'])
                                
                                # Append univariate extension (skip years in historical)
                                for j, year in enumerate(univariate_fc['years']):
                                    if year not in hist_years:  # CRITICAL: Avoid duplicates
                                        results.append({
                                            'region_code': 'K02000001',
                                            'region': 'United Kingdom',
                                            'metric': metric,
                                            'year': year,
                                            'value': univariate_fc['values'][j],
                                            'ci_lower': univariate_fc['ci_lower'][j],
                                            'ci_upper': univariate_fc['ci_upper'][j],
                                            'method': f"{forecast['method']}+{univariate_fc['method']}",
                                            'data_type': 'forecast',
                                            'source': 'model'
                                        })
                        
                        logger.info(f"  ✓ VAR: {len(var_results)} metrics")
        
        # Process remaining metrics univariate
        for task in tasks:
            if task['metric'] in processed_metrics:
                continue
            
            logger.info(f"  Processing {task['metric']} (univariate)...")
            
            series_data = data[data['metric'] == task['metric']].sort_values('year')
            
            # Get historical years to avoid duplicates
            hist_years = set(series_data['year'].unique())
            
            # CRITICAL: Use regular int index
            series = pd.Series(
                series_data['value'].values, 
                index=series_data['year'].values.astype(int), 
                name=task['metric']
            )
            
            use_log = self.config.use_log_transform.get(task['metric'], False)
            if use_log and (series > 0).all():
                working_series = np.log(series)
            else:
                working_series = series
                use_log = False
            
            breaks = [
                b for b in self.config.structural_breaks
                if task['last_year'] - 20 <= b['year'] <= task['last_year']
            ]
            
            try:
                forecast = self.forecaster.forecast_univariate(
                    working_series, task['horizon'], breaks, task['metric_info']
                )
                
                if use_log:
                    forecast['values'] = np.exp(forecast['values'])
                    forecast['ci_lower'] = np.exp(forecast['ci_lower'])
                    forecast['ci_upper'] = np.exp(forecast['ci_upper'])
                
                for j, year in enumerate(forecast['years']):
                    if year not in hist_years:  # CRITICAL: Avoid duplicates
                        results.append({
                            'region_code': 'K02000001',
                            'region': 'United Kingdom',
                            'metric': task['metric'],
                            'year': year,
                            'value': forecast['values'][j],
                            'ci_lower': forecast['ci_lower'][j],
                            'ci_upper': forecast['ci_upper'][j],
                            'method': forecast['method'],
                            'data_type': 'forecast',
                            'source': 'model'
                        })
            except Exception as e:
                logger.error(f"Forecast failed for {task['metric']}: {e}")
        
        return results
    
    def _calculate_derived(self, data: pd.DataFrame) -> List[Dict]:
        """Calculate derived metrics for all years including historical"""
        derived = []
        
        # Get all unique years (both historical and forecast)
        all_years = sorted(data['year'].unique())
        
        for year in all_years:
            try:
                year_int = int(year)
                year_derived = self.derived.calculate_all_derived(data, year_int)
                derived.extend(year_derived)
            except Exception as e:
                logger.error(f"Derived failed for {year}: {e}")
        
        logger.info(f"  Calculated derived metrics for {len(all_years)} years")
        return derived
    
    def _save_outputs(self, data: pd.DataFrame):
        """Save all outputs"""
        data = data.copy()
        
        # CI clean-up before anything else
        if "ci_lower" in data.columns and "ci_upper" in data.columns:
            # 1) Ensure ci_lower <= ci_upper
            swap_mask = (data["ci_lower"].notna() &
                         data["ci_upper"].notna() &
                         (data["ci_lower"] > data["ci_upper"]))
            if swap_mask.any():
                logger.warning(f"Swapping {swap_mask.sum()} CIs where lower > upper")
                tmp = data.loc[swap_mask, "ci_lower"].copy()
                data.loc[swap_mask, "ci_lower"] = data.loc[swap_mask, "ci_upper"]
                data.loc[swap_mask, "ci_upper"] = tmp

            # 2) Cap CV at 0.5 for forecast rows
            with np.errstate(divide="ignore", invalid="ignore"):
                denom = 2.0 * data["value"].abs().replace(0, np.nan)
                cv = (data["ci_upper"] - data["ci_lower"]) / denom

            wide_mask = (
                data["data_type"].eq("forecast") &
                cv.notna() &
                (cv > 0.5)
            )

            if wide_mask.any():
                n_wide = int(wide_mask.sum())
                logger.warning(f"Shrinking {n_wide} very wide CIs (CV > 0.5)")
                scale = 0.5 / cv[wide_mask]

                mid = (data.loc[wide_mask, "ci_upper"] + data.loc[wide_mask, "ci_lower"]) / 2.0
                half_width = (data.loc[wide_mask, "ci_upper"] - data.loc[wide_mask, "ci_lower"]) / 2.0
                half_width = half_width * scale

                data.loc[wide_mask, "ci_lower"] = mid - half_width
                data.loc[wide_mask, "ci_upper"] = mid + half_width
        
        # Long format
        long_path = self.config.output_dir / "uk_macro_forecast_long.csv"
        data.to_csv(long_path, index=False)
        logger.info(f"✓ Long format: {long_path}")
        
        # Wide format
        wide = data.pivot_table(
            index='metric',
            columns='year',
            values='value',
            aggfunc='first'
        ).reset_index()
        wide.columns.name = None
        
        wide_path = self.config.output_dir / "uk_macro_forecast_wide.csv"
        wide.to_csv(wide_path, index=False)
        logger.info(f"✓ Wide format: {wide_path}")
        
        # Confidence intervals - only if columns exist
        if 'ci_lower' in data.columns and 'ci_upper' in data.columns:
            ci_data = data[['metric', 'year', 'value', 'ci_lower', 'ci_upper']].copy()
            ci_data = ci_data.dropna(subset=['ci_lower', 'ci_upper'])
            if not ci_data.empty:
                ci_data['ci_width'] = ci_data['ci_upper'] - ci_data['ci_lower']
                ci_data['cv'] = ci_data['ci_width'] / (2 * ci_data['value'])
                
                ci_path = self.config.output_dir / "uk_macro_confidence_intervals.csv"
                ci_data.to_csv(ci_path, index=False)
                logger.info(f"✓ Confidence intervals: {ci_path}")
        
        # Metadata
        metadata = {
            'run_timestamp': datetime.now().isoformat(),
            'version': '3.2-fixed',
            'level': 'UK_macro',
            'config': {
                'target_year': self.config.target_year,
                'var_enabled': self.config.use_var_systems,
                'var_systems': self.config.var_systems,
                'structural_breaks': self.config.structural_breaks
            },
            'data_summary': {
                'metrics': int(data['metric'].nunique()),
                'total_observations': len(data),
                'historical_obs': len(data[data['data_type'] == 'historical']),
                'forecast_obs': len(data[data['data_type'] == 'forecast'])
            },
            'model_usage': (
                data[data['data_type'] == 'forecast']['method'].value_counts().to_dict()
                if 'method' in data.columns else {}
            )
        }
        
        metadata_path = self.config.output_dir / "uk_macro_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"✓ Metadata: {metadata_path}")
        
        # DuckDB
        if HAVE_DUCKDB:
            try:
                data_copy = data.copy()
                
                # CRITICAL: Fill metric_id/period even if columns exist but contain NULLs
                if "metric" in data_copy.columns:
                    data_copy["metric_id"] = data_copy.get("metric_id", data_copy["metric"]).fillna(
                        data_copy["metric"]
                    )
                if "year" in data_copy.columns:
                    data_copy["period"] = data_copy.get("period", data_copy["year"]).fillna(
                        data_copy["year"]
                    )
                
                if "region" not in data_copy.columns:
                    data_copy["region"] = "United Kingdom"
                
                data_copy["region_level"] = "UK"
                data_copy["forecast_run_date"] = datetime.now().date()
                data_copy["forecast_version"] = "3.2-fixed"
                
                # Add missing columns with defaults
                if 'unit' not in data_copy.columns:
                    data_copy['unit'] = data_copy['metric_id'].map(
                        {
                            m: self.config.metric_definitions.get(m, {}).get('unit', 'unknown')
                            for m in data_copy['metric_id'].unique()
                        }
                    )
                if 'freq' not in data_copy.columns:
                    data_copy['freq'] = 'A'
                if 'region_name' not in data_copy.columns:
                    data_copy['region_name'] = 'United Kingdom'
                
                cols = [
                    "region_code", "region_name", "region_level",
                    "metric_id", "period", "value",
                    "unit", "freq", "data_type",
                    "ci_lower", "ci_upper",
                    "forecast_run_date", "forecast_version"
                ]
                data_flat = data_copy[[c for c in cols if c in data_copy.columns]].reset_index(drop=True)
                
                con = duckdb.connect(str(self.config.duckdb_path))
                con.execute("CREATE SCHEMA IF NOT EXISTS gold")
                con.register("macro_df", data_flat)
                
                con.execute("""
                    CREATE OR REPLACE TABLE gold.uk_macro_forecast AS
                    SELECT * FROM macro_df
                """)
                
                con.execute("""
                    CREATE OR REPLACE VIEW gold.uk_macro_forecast_only AS
                    SELECT * FROM gold.uk_macro_forecast
                    WHERE data_type = 'forecast'
                """)
                
                con.close()
                logger.info(f"✓ DuckDB: gold.uk_macro_forecast ({len(data_flat)} rows)")
                
            except Exception as e:
                logger.warning(f"DuckDB save failed: {e}")


# ===============================
# Entry Point
# ===============================

def main():
    """Run UK macro forecasting V3.2"""
    
    try:
        config = MacroForecastConfig()
        
        logger.info("="*70)
        logger.info("UK MACRO FORECAST V3.2 - VAR ENHANCEMENT (FIXED)")
        logger.info("="*70)
        logger.info(f"  Silver: {config.silver_path}")
        logger.info(f"  Target: {config.target_year}")
        logger.info(f"  VAR: {config.use_var_systems}")
        if config.use_var_systems:
            logger.info(f"  Systems: {list(config.var_systems.keys())}")
        
        forecaster = MacroForecasterV3(config)
        results = forecaster.run()
        
        logger.info("="*70)
        logger.info("✅ UK MACRO FORECASTING COMPLETED")
        logger.info(f"📊 Total records: {len(results)}")
        logger.info(f"📊 Metrics: {results['metric'].nunique()}")
        logger.info(f"📊 Forecasts: {len(results[results['data_type']=='forecast'])}")
        
        if 'method' in results.columns:
            var_count = sum(
                'VAR' in str(m) or 'VECM' in str(m)
                for m in results[results['data_type'] == 'forecast']['method']
            )
            logger.info(f"📊 VAR/VECM forecasts: {var_count}")
        
        logger.info("="*70)
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()