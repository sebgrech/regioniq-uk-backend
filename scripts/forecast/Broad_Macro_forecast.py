#!/usr/bin/env python3
"""
Region IQ - UK Macro Forecasting Engine V3.5 (Blended VAR)
============================================================

V3.5: VAR → Univariate smooth blend + Clean GOLD architecture
- Exponential decay weighting (VAR strong early, fades by ~7 years)
- Vectorised blending of point estimates and CI bands
- Eliminates hard cutoff artifacts at VAR horizon boundary
- CLEAN GOLD: Base metrics in gold.uk_macro_forecast
- DERIVED LAYER: productivity, income_per_worker in gold.uk_macro_derived
- COMPUTED: gdhi_per_head = gdhi_total / population (in BASE, not forecasted)

Metric Categories:
- FORECASTED (7): gva, gdhi_total, emp, pop_total, pop_16_64, emp_rate, unemp_rate
- COMPUTED (1): gdhi_per_head (from gdhi_total / population, stored in BASE)
- DERIVED (2): productivity, income_per_worker (analytics layer, stored in DERIVED)

Changes from V3.4:
- New _blend_var_univariate() method
- VAR and univariate run in parallel for full horizon
- Smooth weight transition instead of abrupt switch
- Removed productivity/income_per_worker from base table
- Added gold.uk_macro_derived for analytics layer
- gdhi_per_head now computed post-forecast (not forecasted directly)

Inputs:
    data/silver/uk_macro_history.csv

Outputs:
    data/forecast/uk_macro_forecast_long.csv   (base metrics)
    data/forecast/uk_macro_derived.csv         (derived metrics)
    data/forecast/uk_macro_forecast_wide.csv
    data/forecast/uk_macro_confidence_intervals.csv
    data/forecast/uk_macro_metadata.json
    gold.uk_macro_forecast (DuckDB - base)
    gold.uk_macro_derived (DuckDB - derived)

Author: Region IQ
Version: 3.5
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
# V3.4 GROWTH CAPS - CRITICAL FIX
# ===============================

GROWTH_CAPS = {
    'uk_nominal_gva_mn_gbp': {'min': -0.03, 'max': 0.045},
    'uk_gdhi_total_mn_gbp': {'min': -0.03, 'max': 0.045},
    'uk_emp_total_jobs': {'min': -0.02, 'max': 0.015},
    'uk_population_total': {'min': 0.0, 'max': 0.008},
    'uk_population_16_64': {'min': -0.005, 'max': 0.006},  # Working age can shrink slightly
}

# Derived metrics (calculated post-forecast, stored in gold.uk_macro_derived)
DERIVED_METRICS = ['uk_productivity_gbp_per_job', 'uk_income_per_worker_gbp']

# Computed metrics (calculated post-forecast from components, stored in BASE table)
# These are standard ONS metrics, not analytics - hence in base not derived
COMPUTED_METRICS = ['uk_gdhi_per_head_gbp']

SANITY_CAPS = {
    'uk_nominal_gva_mn_gbp': 12_000_000,
    'uk_gdhi_total_mn_gbp': 10_000_000,
    'uk_emp_total_jobs': 42_000_000,
    'uk_population_total': 85_000_000,
    'uk_population_16_64': 55_000_000,  # ~65% of total pop
}


@dataclass
class MacroForecastConfig:
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
        'gva_employment': ['uk_nominal_gva_mn_gbp', 'uk_emp_total_jobs'],
        'labour_market_rates': ['uk_employment_rate_pct', 'uk_unemployment_rate_pct']
    })
    var_max_horizon: int = 15  # VAR reliable horizon (blending handles the rest)
    var_blend_decay: float = 0.25  # Exponential decay rate for VAR weight
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
    enforce_growth_caps: bool = True
    
    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if self.metric_definitions is None:
            self.metric_definitions = {
                'uk_population_total': {'unit': 'persons', 'transform': 'log', 'monotonic': True, 'type': 'level'},
                'uk_population_16_64': {'unit': 'persons', 'transform': 'none', 'monotonic': False, 'type': 'level'},  # Working age can decline
                'uk_gdhi_total_mn_gbp': {'unit': 'GBP_m', 'transform': 'log', 'monotonic': False, 'type': 'level'},
                'uk_nominal_gva_mn_gbp': {'unit': 'GBP_m', 'transform': 'log', 'monotonic': False, 'type': 'level'},
                'uk_emp_total_jobs': {'unit': 'jobs', 'transform': 'log', 'monotonic': False, 'type': 'level'},
                'uk_employment_rate_pct': {'unit': 'percent', 'transform': 'none', 'monotonic': False, 'type': 'rate', 'lower_bound': 0.0, 'upper_bound': 100.0},
                'uk_unemployment_rate_pct': {'unit': 'percent', 'transform': 'none', 'monotonic': False, 'type': 'rate', 'lower_bound': 0.0, 'upper_bound': 100.0}
            }
            # Note: uk_gdhi_per_head_gbp is NOT forecasted - it's computed post-forecast from gdhi_total / population
        
        if self.use_log_transform is None:
            self.use_log_transform = {m: info.get('transform', 'none') == 'log' for m, info in self.metric_definitions.items()}
        
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
        logger.info("Macro V3.5 Configuration validated")


class MacroDataManager:
    def __init__(self, config: MacroForecastConfig):
        self.config = config
        
    def load_all_data(self) -> pd.DataFrame:
        cache_key = self._get_cache_key()
        if self.config.cache_enabled and self._cache_exists(cache_key):
            logger.info("Loading from cache")
            return self._load_from_cache(cache_key)
        
        unified = self._load_from_csv() if not self.config.use_duckdb else self._load_from_duckdb()
        unified = self._standardize_columns(unified)
        unified = self._handle_outliers(unified)
        
        if self.config.cache_enabled:
            self._save_to_cache(unified, cache_key)
        
        logger.info(f"Loaded {len(unified)} macro observations, {unified['metric'].nunique()} metrics")
        return unified
    
    def _load_from_csv(self) -> pd.DataFrame:
        logger.info(f"Loading from {self.config.silver_path}")
        df = pd.read_csv(self.config.silver_path)
        if 'region_level' in df.columns:
            df = df[df['region_level'] == 'UK']
        
        if 'metric' in df.columns and 'metric_id' not in df.columns:
            df['metric_id'] = df['metric']
        if 'year' in df.columns and 'period' not in df.columns:
            df['period'] = df['year']
        return df
    
    def _load_from_duckdb(self) -> pd.DataFrame:
        try:
            con = duckdb.connect(str(self.config.duckdb_path), read_only=True)
            df = con.execute("SELECT * FROM silver.uk_macro_history").fetchdf()
            con.close()
            return df
        except Exception as e:
            logger.error(f"DuckDB load failed: {e}")
            return self._load_from_csv()
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for old, new in {'metric_id': 'metric', 'period': 'year', 'region_name': 'region'}.items():
            if old in df.columns and new not in df.columns:
                df[new] = df[old]
        
        df['year'] = pd.to_numeric(df['year'], errors='coerce').astype(int)
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        if 'data_type' not in df.columns:
            df['data_type'] = 'historical'
        df = df.dropna(subset=['year', 'value'])
        if self.config.enforce_non_negative:
            df = df[df['value'] >= 0]
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        for metric in df['metric'].unique():
            group = df[df['metric'] == metric]
            if len(group) < 5:
                continue
            values = group['value'].values
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = (values < lower) | (values > upper)
            if outliers.any():
                df.loc[group[outliers].index, 'value'] = np.clip(values[outliers], lower, upper)
        return df
    
    def _get_cache_key(self) -> str:
        hasher = hashlib.md5()
        path = self.config.duckdb_path if self.config.use_duckdb else self.config.silver_path
        if path.exists():
            hasher.update(str(path.stat().st_mtime).encode())
        hasher.update(b"v3.5")
        return hasher.hexdigest()
    
    def _cache_exists(self, key: str) -> bool:
        return (self.config.cache_dir / f"macro_{key}.pkl").exists()
    
    def _load_from_cache(self, key: str) -> pd.DataFrame:
        with open(self.config.cache_dir / f"macro_{key}.pkl", 'rb') as f:
            return pickle.load(f)
    
    def _save_to_cache(self, df: pd.DataFrame, key: str):
        with open(self.config.cache_dir / f"macro_{key}.pkl", 'wb') as f:
            pickle.dump(df, f)


class VARSystemForecaster:
    def __init__(self, config: MacroForecastConfig):
        self.config = config
    
    def forecast_system(self, data: pd.DataFrame, metrics: List[str], horizon: int, structural_breaks: Optional[Sequence[Dict]] = None) -> Optional[Dict]:
        if not HAVE_STATSMODELS:
            return None
        
        try:
            series_dict = {}
            for metric in metrics:
                metric_data = data[data['metric'] == metric].sort_values('year')
                if metric_data.empty:
                    return None
                values = metric_data['value'].values
                if self.config.use_log_transform.get(metric, False) and (values > 0).all():
                    values = np.log(values)
                series_dict[metric] = pd.Series(values, index=metric_data['year'].values.astype(int))
            
            system_df = pd.DataFrame(series_dict).dropna()
            if len(system_df) < self.config.min_history_years:
                return None
            
            coint_result = self._test_cointegration(system_df)
            exog = self._prepare_break_dummies(system_df.index, structural_breaks) if structural_breaks else None
            
            if coint_result['cointegrated'] and coint_result['rank'] > 0:
                logger.info(f"  VECM (rank={coint_result['rank']})")
                model = VECM(system_df, k_ar_diff=min(self.config.max_var_lags, 3), coint_rank=coint_result['rank'], deterministic='ci', exog=exog)
                fitted = model.fit()
                method_name = f'VECM(r={coint_result["rank"]})'
                exog_fc = self._extend_exog(exog, system_df.index, horizon) if exog is not None else None
                forecast_values = fitted.predict(steps=horizon, exog_fc=exog_fc)
            else:
                logger.info("  VAR")
                model = VAR(system_df, exog=exog)
                lag_results = model.select_order(maxlags=self.config.max_var_lags)
                try:
                    optimal_lags = lag_results.selected_orders['aic']
                except:
                    optimal_lags = lag_results.aic
                optimal_lags = max(1, min(optimal_lags, self.config.max_var_lags))
                fitted = model.fit(maxlags=optimal_lags)
                method_name = f'VAR({optimal_lags})'
                exog_fc = self._extend_exog(exog, system_df.index, horizon) if exog is not None else None
                forecast_values = fitted.forecast(system_df.values[-fitted.k_ar:], steps=horizon, exog_future=exog_fc)
            
            ci_lower, ci_upper = self._compute_ci(fitted, system_df, horizon)
            last_year = int(system_df.index[-1])
            forecast_years = list(range(last_year + 1, last_year + horizon + 1))
            
            results = {}
            for i, metric in enumerate(metrics):
                vals = forecast_values[:, i]
                ci_l, ci_u = ci_lower[:, i], ci_upper[:, i]
                if self.config.use_log_transform.get(metric, False):
                    vals, ci_l, ci_u = np.exp(vals), np.exp(ci_l), np.exp(ci_u)
                results[metric] = {'method': method_name, 'values': vals, 'years': forecast_years, 'ci_lower': ci_l, 'ci_upper': ci_u, 'aic': getattr(fitted, 'aic', None), 'system_metrics': metrics, 'cointegrated': coint_result['cointegrated']}
            
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
    
    def _compute_ci(self, fitted, data, horizon):
        residuals = fitted.resid if hasattr(fitted, 'resid') else data.values[fitted.k_ar:] - fitted.fittedvalues
        if hasattr(residuals, 'values'):
            residuals = residuals.values
        residuals = np.atleast_2d(residuals)
        if residuals.shape[0] == 1:
            residuals = residuals.T
        n_vars = residuals.shape[1]
        resid_cov = np.cov(residuals.T)
        if n_vars == 1 or resid_cov.ndim == 0:
            resid_cov = np.array([[resid_cov]]) if resid_cov.ndim == 0 else resid_cov.reshape(1, 1)
        horizon_factors = np.sqrt(np.arange(1, horizon + 1))
        ci_lower = np.zeros((horizon, n_vars))
        ci_upper = np.zeros((horizon, n_vars))
        for i in range(n_vars):
            se = np.sqrt(resid_cov[i, i])
            ci_width = 1.96 * se * horizon_factors
            ci_lower[:, i] = -ci_width
            ci_upper[:, i] = ci_width
        return ci_lower, ci_upper
    
    def _prepare_break_dummies(self, index, breaks):
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
        for _ in range(horizon):
            row = []
            for i in range(exog.shape[1]):
                if np.all(np.isin(exog[:, i], [0, 1])):
                    row.append(exog[-1, i])
                else:
                    inc = exog[-1, i] - exog[-2, i] if len(exog) > 1 else 1
                    row.append(exog[-1, i] + inc)
            extended.append(row)
        return np.array(extended)


class AdvancedMacroForecaster:
    def __init__(self, config: MacroForecastConfig):
        self.config = config
        self.var_forecaster = VARSystemForecaster(config) if config.use_var_systems else None
    
    def forecast_univariate(self, series: pd.Series, horizon: int, structural_breaks: Optional[Sequence[Dict]] = None, metric_info: Dict = None) -> Dict:
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
            combined = self._fallback_forecast(series, horizon)
        elif len(models) > 1:
            cv_errors = self._cross_validate(series, models)
            weights = self._calculate_weights(cv_errors)
            combined = self._combine_forecasts(models, weights, series)
        else:
            combined = models[0]
        
        if metric_info:
            combined = self._apply_constraints(combined, series, metric_info)
        return combined
    
    def _fit_arima_with_breaks(self, series: pd.Series, horizon: int, structural_breaks: Optional[Sequence[Dict]] = None) -> Optional[Dict]:
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
                    except:
                        continue
            
            if best_model is None:
                return None
            
            exog_fc = self._extend_exog(exog, series.index, horizon) if exog is not None else None
            fc_obj = best_model.get_forecast(steps=horizon, exog=exog_fc)
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
        if not HAVE_STATSMODELS:
            return None
        try:
            best_model, best_aic, best_config = None, np.inf, None
            for trend in ['add', 'mul', None]:
                for damped in [True, False]:
                    if trend is None and damped:
                        continue
                    try:
                        model = ExponentialSmoothing(series, trend=trend, damped_trend=damped, seasonal=None)
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic, best_model, best_config = fitted.aic, fitted, f"ETS({trend},{damped})"
                    except:
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
            
            return {'method': best_config, 'values': forecast.values, 'years': forecast_years, 'ci_lower': ci_lower, 'ci_upper': ci_upper, 'aic': best_aic, 'fit_function': lambda s, h: self._ets_refit(s, h, best_model)}
        except Exception as e:
            logger.debug(f"ETS failed: {e}")
            return None
    
    def _fit_linear(self, series: pd.Series, horizon: int) -> Optional[Dict]:
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
                ci = np.column_stack([forecast - 1.96 * sigma, forecast + 1.96 * sigma])
            
            last_year = int(series.index[-1])
            forecast_years = list(range(last_year + 1, last_year + horizon + 1))
            return {'method': 'Linear', 'values': forecast, 'years': forecast_years, 'ci_lower': ci[:, 0], 'ci_upper': ci[:, 1], 'aic': model.aic if HAVE_STATSMODELS else None, 'fit_function': lambda s, h: self._linear_refit(s, h)}
        except Exception as e:
            logger.debug(f"Linear failed: {e}")
            return None
    
    def _cross_validate(self, series: pd.Series, models: List[Dict]) -> Dict:
        cv_errors = {i: [] for i in range(len(models))}
        min_train = max(self.config.cv_min_train_size, len(series) // 2)
        for train_end in range(min_train, len(series) - self.config.cv_horizon + 1):
            train = series.iloc[:train_end]
            test = series.iloc[train_end:min(train_end + self.config.cv_horizon, len(series))]
            for i, m in enumerate(models):
                try:
                    if 'fit_function' in m:
                        fc = m['fit_function'](train, len(test))
                        error = np.mean(np.abs(fc - test.values) / (np.abs(fc) + np.abs(test.values) + 1e-10))
                        cv_errors[i].append(error)
                except:
                    cv_errors[i].append(np.inf)
        return cv_errors
    
    def _calculate_weights(self, cv_errors: Dict) -> np.ndarray:
        mean_errors = [np.mean([e for e in cv_errors[i] if e != np.inf]) if [e for e in cv_errors[i] if e != np.inf] else np.inf for i in sorted(cv_errors.keys())]
        if all(e == np.inf for e in mean_errors):
            return np.ones(len(mean_errors)) / len(mean_errors)
        weights = 1.0 / (np.array(mean_errors) + 1e-10)
        weights[np.isinf(weights)] = 0
        return weights / weights.sum() if weights.sum() > 0 else np.ones(len(weights)) / len(weights)
    
    def _combine_forecasts(self, models: List[Dict], weights: np.ndarray, series: pd.Series) -> Dict:
        combined_values = sum(w * m['values'] for w, m in zip(weights, models))
        ci_lower = sum(w * m['ci_lower'] for w, m in zip(weights, models))
        ci_upper = sum(w * m['ci_upper'] for w, m in zip(weights, models))
        return {'method': f"Ensemble({len(models)})", 'values': combined_values, 'years': models[0]['years'], 'ci_lower': ci_lower, 'ci_upper': ci_upper, 'weights': weights.tolist(), 'component_methods': [m['method'] for m in models]}
    
    def _apply_constraints(self, forecast: Dict, series: pd.Series, metric_info: Dict) -> Dict:
        """V3.4: Apply constraints with FIXED growth caps in LEVEL space"""
        metric_name = series.name
        is_log = self.config.use_log_transform.get(metric_name, False)
        
        values = forecast['values'].copy()
        ci_lower = forecast['ci_lower'].copy()
        ci_upper = forecast['ci_upper'].copy()
        
        # Convert to level space
        if is_log:
            values_level = np.exp(values)
            ci_lower_level = np.exp(ci_lower)
            ci_upper_level = np.exp(ci_upper)
            last_hist_level = np.exp(series.iloc[-1])
        else:
            values_level = values.copy()
            ci_lower_level = ci_lower.copy()
            ci_upper_level = ci_upper.copy()
            last_hist_level = series.iloc[-1]
        
        # Non-negative
        if self.config.enforce_non_negative:
            values_level = np.maximum(values_level, 0)
            ci_lower_level = np.maximum(ci_lower_level, 0)
        
        # Rate bounds
        lower_bound = metric_info.get('lower_bound')
        upper_bound = metric_info.get('upper_bound')
        if lower_bound is not None:
            values_level = np.maximum(values_level, lower_bound)
            ci_lower_level = np.maximum(ci_lower_level, lower_bound)
        if upper_bound is not None:
            values_level = np.minimum(values_level, upper_bound)
            ci_upper_level = np.minimum(ci_upper_level, upper_bound)
        
        # Monotonic population
        if metric_info.get('monotonic', False) and self.config.enforce_monotonic_population:
            for i in range(1, len(values_level)):
                values_level[i] = max(values_level[i], values_level[i-1])
        
        # V3.4 CRITICAL: Growth caps in LEVEL space
        if self.config.enforce_growth_caps and metric_info.get('type') != 'rate':
            growth_cap = GROWTH_CAPS.get(metric_name)
            if growth_cap:
                min_growth, max_growth = growth_cap['min'], growth_cap['max']
                capped_count = 0
                prev_value = last_hist_level
                
                for i in range(len(values_level)):
                    if prev_value <= 0:
                        prev_value = values_level[i]
                        continue
                    actual_growth = (values_level[i] - prev_value) / prev_value
                    if actual_growth > max_growth:
                        values_level[i] = prev_value * (1 + max_growth)
                        capped_count += 1
                    elif actual_growth < min_growth:
                        values_level[i] = prev_value * (1 + min_growth)
                        capped_count += 1
                    prev_value = values_level[i]
                
                if capped_count > 0:
                    logger.info(f"    {metric_name}: Capped {capped_count} years to [{min_growth*100:.1f}%, {max_growth*100:.1f}%]")
                    # Recalculate CIs
                    hist_vol = series.pct_change().dropna().std()
                    if np.isfinite(hist_vol) and hist_vol > 0:
                        for i in range(len(values_level)):
                            ci_width = 1.96 * values_level[i] * hist_vol * np.sqrt(i + 1)
                            ci_lower_level[i] = max(0, values_level[i] - ci_width)
                            ci_upper_level[i] = values_level[i] + ci_width
        
        # Convert back
        if is_log:
            values_level = np.maximum(values_level, 1e-10)
            ci_lower_level = np.maximum(ci_lower_level, 1e-10)
            ci_upper_level = np.maximum(ci_upper_level, 1e-10)
            forecast['values'] = np.log(values_level)
            forecast['ci_lower'] = np.log(ci_lower_level)
            forecast['ci_upper'] = np.log(ci_upper_level)
        else:
            forecast['values'] = values_level
            forecast['ci_lower'] = ci_lower_level
            forecast['ci_upper'] = ci_upper_level
        
        return forecast
    
    def _prep_breaks(self, series, breaks):
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
        if exog is None:
            return None
        if len(exog.shape) == 1:
            exog = exog.reshape(-1, 1)
        extended = []
        for _ in range(horizon):
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
        try:
            exog = self._prep_breaks(series, breaks) if breaks else None
            model = ARIMA(series, order=order, exog=exog).fit(disp=False)
            exog_fc = self._extend_exog(exog, series.index, horizon) if exog is not None else None
            return model.forecast(steps=horizon, exog=exog_fc).values
        except:
            return np.repeat(series.iloc[-1], horizon)
    
    def _ets_refit(self, series, horizon, ref_model):
        try:
            return ref_model.forecast(steps=horizon).values
        except:
            return np.repeat(series.iloc[-1], horizon)
    
    def _linear_refit(self, series, horizon):
        try:
            X = np.arange(len(series))
            coeffs = np.polyfit(X, series.values, 1)
            return np.polyval(coeffs, np.arange(len(series), len(series) + horizon))
        except:
            return np.repeat(series.iloc[-1], horizon)
    
    def _fallback_forecast(self, series: pd.Series, horizon: int) -> Dict:
        last_value = series.iloc[-1]
        trend = (series.iloc[-1] - series.iloc[-5]) / 5 if len(series) > 5 else 0
        values = np.array([max(last_value + trend * h, 0) for h in range(1, horizon + 1)])
        std = series.std() if len(series) > 2 else abs(last_value) * 0.1
        ci_width = 1.96 * std * np.sqrt(range(1, horizon + 1))
        last_year = int(series.index[-1])
        return {'method': 'Fallback', 'values': values, 'years': list(range(last_year + 1, last_year + horizon + 1)), 'ci_lower': values - ci_width, 'ci_upper': values + ci_width}


class MacroDerivedMetrics:
    """
    Derived metrics are calculated post-forecast and stored in gold.uk_macro_derived.
    These are NOT in the main forecast table - keeps GOLD clean.
    """
    
    @staticmethod
    def calculate_productivity(gva: Tuple[float, float, float], emp: Tuple[float, float, float]) -> Tuple[float, float, float]:
        gva_val, gva_low, gva_high = gva
        emp_val, emp_low, emp_high = emp
        if gva_val <= 0 or emp_val <= 0:
            return 0.0, 0.0, 0.0
        if gva_val < 100_000:
            gva_val, gva_low, gva_high = gva_val * 1000, gva_low * 1000, gva_high * 1000
        point = (gva_val * 1_000_000) / emp_val
        se = point * 0.15
        return point, max(0, point - 1.96 * se), point + 1.96 * se
    
    @staticmethod
    def calculate_income_per_worker(gdhi: Tuple[float, float, float], emp: Tuple[float, float, float]) -> Tuple[float, float, float]:
        gdhi_val, gdhi_low, gdhi_high = gdhi
        emp_val, emp_low, emp_high = emp
        if gdhi_val <= 0 or emp_val <= 0:
            return 0.0, 0.0, 0.0
        if gdhi_val < 100_000:
            gdhi_val, gdhi_low, gdhi_high = gdhi_val * 1000, gdhi_low * 1000, gdhi_high * 1000
        point = (gdhi_val * 1_000_000) / emp_val
        se = point * 0.15
        return point, max(0, point - 1.96 * se), point + 1.96 * se
    
    @staticmethod
    def calculate_all_derived(data: pd.DataFrame, year: int) -> List[Dict]:
        derived = []
        year_data = data[data['year'] == year]
        if year_data.empty:
            return derived
        metrics = {row['metric']: (row['value'], row.get('ci_lower', row['value'] * 0.95), row.get('ci_upper', row['value'] * 1.05)) for _, row in year_data.iterrows()}
        
        # Productivity = GVA / Employment
        if 'uk_nominal_gva_mn_gbp' in metrics and 'uk_emp_total_jobs' in metrics:
            prod_val, prod_low, prod_high = MacroDerivedMetrics.calculate_productivity(metrics['uk_nominal_gva_mn_gbp'], metrics['uk_emp_total_jobs'])
            derived.append({'region_code': 'K02000001', 'region': 'United Kingdom', 'year': year, 'metric': 'uk_productivity_gbp_per_job', 'value': prod_val, 'ci_lower': prod_low, 'ci_upper': prod_high, 'method': 'derived', 'source': 'calculated', 'data_type': 'forecast' if year > 2024 else 'historical', 'formula': 'gva / emp * 1e6'})
        
        # Income per worker = GDHI / Employment
        if 'uk_gdhi_total_mn_gbp' in metrics and 'uk_emp_total_jobs' in metrics:
            inc_val, inc_low, inc_high = MacroDerivedMetrics.calculate_income_per_worker(metrics['uk_gdhi_total_mn_gbp'], metrics['uk_emp_total_jobs'])
            derived.append({'region_code': 'K02000001', 'region': 'United Kingdom', 'year': year, 'metric': 'uk_income_per_worker_gbp', 'value': inc_val, 'ci_lower': inc_low, 'ci_upper': inc_high, 'method': 'derived', 'source': 'calculated', 'data_type': 'forecast' if year > 2024 else 'historical', 'formula': 'gdhi / emp * 1e6'})
        
        return derived


class MacroForecasterV3:
    def __init__(self, config: MacroForecastConfig):
        self.config = config
        self.data_manager = MacroDataManager(config)
        self.forecaster = AdvancedMacroForecaster(config)
        self.derived = MacroDerivedMetrics()
    
    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns:
            Tuple of (base_data, derived_data)
            - base_data: Goes to gold.uk_macro_forecast
            - derived_data: Goes to gold.uk_macro_derived
        """
        logger.info("="*60)
        logger.info("UK MACRO FORECASTING ENGINE V3.5 (Blended VAR)")
        logger.info("="*60)
        
        historical = self.data_manager.load_all_data()
        tasks = self._identify_tasks(historical)
        logger.info(f"Identified {len(tasks)} forecasting tasks")
        
        results = self._run_forecasts_with_systems(historical, tasks)
        base_data = pd.concat([historical, pd.DataFrame(results)], ignore_index=True)
        
        # Compute GDHI per head from components (stays in base - it's a standard ONS metric)
        base_data = self._compute_gdhi_per_head(base_data)
        base_data = self._apply_sanity_caps(base_data)
        base_data = self._deduplicate_records(base_data)
        
        # Calculate derived metrics SEPARATELY
        logger.info("Calculating derived indicators (separate table)...")
        derived_rows = [d for year in base_data['year'].unique() for d in self.derived.calculate_all_derived(base_data, int(year))]
        derived_data = pd.DataFrame(derived_rows) if derived_rows else pd.DataFrame()
        
        for col in ["region_code", "region_name", "region_level", "metric_id", "period", "unit", "freq", "ci_lower", "ci_upper", "method", "data_type"]:
            if col not in base_data.columns:
                base_data[col] = None
        
        self._save_outputs(base_data, derived_data)
        logger.info("✅ Macro forecasting V3.5 completed")
        return base_data, derived_data
    
    def _identify_tasks(self, data: pd.DataFrame) -> List[Dict]:
        tasks = []
        for metric, group in data.groupby('metric'):
            if len(group) < self.config.min_history_years:
                continue
            last_year = int(group['year'].max())
            horizon = self.config.target_year - last_year
            if horizon <= 0:
                continue
            tasks.append({'metric': metric, 'horizon': horizon, 'last_year': last_year, 'forecast_start': last_year + 1, 'history_length': len(group), 'metric_info': self.config.metric_definitions.get(metric, {})})
        return tasks
    
    def _apply_sanity_caps(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        logger.info("\n  Applying sanity caps...")
        for metric, cap in SANITY_CAPS.items():
            mask = (data['metric'] == metric) & (data['value'] > cap)
            if mask.any():
                logger.warning(f"    {metric}: Capping {mask.sum()} values at {cap:,.0f}")
                data.loc[mask, 'value'] = cap
                if 'ci_upper' in data.columns:
                    data.loc[mask, 'ci_upper'] = np.minimum(data.loc[mask, 'ci_upper'], cap * 1.2)
        return data
    
    def _deduplicate_records(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        if "region_code" not in data.columns:
            data["region_code"] = "K02000001"
        if "metric" not in data.columns and "metric_id" in data.columns:
            data["metric"] = data["metric_id"]
        priority_map = {"historical": 0, "forecast": 1, "derived": 2}
        data["__p"] = data["data_type"].map(priority_map).fillna(3).astype(int)
        data = data.sort_values(["region_code", "metric", "year", "__p"]).drop_duplicates(subset=["region_code", "metric", "year"], keep="first").drop(columns=["__p"])
        return data
    
    def _compute_gdhi_per_head(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute uk_gdhi_per_head_gbp from components for ALL years.
        
        This is a COMPUTED metric (not forecasted):
        - gdhi_per_head = gdhi_total / population × 1,000,000
        - Stored in BASE table (it's a standard ONS metric, not analytics)
        - Computed identically in macro and ITL1 for semantic consistency
        """
        data = data.copy()
        if "metric" not in data.columns:
            return data
        
        # Check we have the components
        base = data.pivot_table(index="year", columns="metric", values="value", aggfunc="first")
        if not {"uk_gdhi_total_mn_gbp", "uk_population_total"}.issubset(set(base.columns)):
            logger.warning("  Cannot compute GDHI per head: missing components")
            return data
        
        # Calculate for ALL years
        base["gdhi_per_head_calc"] = (base["uk_gdhi_total_mn_gbp"] / base["uk_population_total"]) * 1_000_000
        
        # Get historical cutoff for data_type tagging
        hist_mask = data["data_type"] == "historical"
        last_hist_year = int(data.loc[hist_mask, "year"].max()) if hist_mask.any() else 2024
        
        # Remove any existing gdhi_per_head rows (we're replacing them all)
        data = data[data['metric'] != 'uk_gdhi_per_head_gbp']
        
        # Create rows for ALL years
        new_rows = []
        for year in sorted(base.index):
            if pd.isna(base.loc[year, "gdhi_per_head_calc"]):
                continue
            
            per_head_val = base.loc[year, "gdhi_per_head_calc"]
            data_type = "historical" if year <= last_hist_year else "forecast"
            
            # Estimate CI (propagate ~10% relative uncertainty, growing with horizon)
            years_ahead = max(0, year - last_hist_year)
            ci_width = per_head_val * 0.10 * (1 + 0.02 * years_ahead)
            ci_lower = max(0, per_head_val - 1.96 * ci_width)
            ci_upper = per_head_val + 1.96 * ci_width
            
            new_rows.append({
                'region_code': 'K02000001',
                'region': 'United Kingdom',
                'metric': 'uk_gdhi_per_head_gbp',
                'year': year,
                'value': per_head_val,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'method': 'computed_from_components',
                'data_type': data_type,
                'source': 'gdhi_total / population'
            })
        
        if new_rows:
            data = pd.concat([data, pd.DataFrame(new_rows)], ignore_index=True)
            logger.info(f"  Computed {len(new_rows)} GDHI per head values from components")
        
        return data
    
    def _blend_var_univariate(
        self,
        var_forecast: Dict,
        uni_forecast: Dict,
        full_horizon: int,
        var_method: str,
        uni_method: str
    ) -> Dict:
        """
        V3.5: Smooth exponential blend from VAR → Univariate
        
        VAR weight decays as exp(-0.25 * h), so:
        - h=1: VAR ~78%, Uni ~22%
        - h=5: VAR ~29%, Uni ~71%
        - h=10: VAR ~8%, Uni ~92%
        - h=15: VAR ~2%, Uni ~98%
        """
        decay = self.config.var_blend_decay
        
        # Get VAR arrays (may be shorter than full horizon)
        var_vals = np.asarray(var_forecast['values'])
        var_ci_lo = np.asarray(var_forecast['ci_lower'])
        var_ci_hi = np.asarray(var_forecast['ci_upper'])
        var_len = len(var_vals)
        
        # Get univariate arrays (full horizon)
        uni_vals = np.asarray(uni_forecast['values'])
        uni_ci_lo = np.asarray(uni_forecast['ci_lower'])
        uni_ci_hi = np.asarray(uni_forecast['ci_upper'])
        
        # Extend VAR if shorter than full horizon (carry forward last value with decay)
        if var_len < full_horizon:
            # Extend VAR by linear extrapolation from last few points
            if var_len >= 3:
                # Use last 3 points to estimate trend
                trend = (var_vals[-1] - var_vals[-3]) / 2
            else:
                trend = 0
            
            extension_len = full_horizon - var_len
            var_extension = var_vals[-1] + trend * np.arange(1, extension_len + 1)
            var_vals = np.concatenate([var_vals, var_extension])
            
            # Extend CIs with growing uncertainty
            last_ci_width = (var_ci_hi[-1] - var_ci_lo[-1]) / 2
            ext_factors = np.sqrt(np.arange(var_len + 1, full_horizon + 1))
            var_ci_lo = np.concatenate([var_ci_lo, var_extension - last_ci_width * ext_factors / np.sqrt(var_len)])
            var_ci_hi = np.concatenate([var_ci_hi, var_extension + last_ci_width * ext_factors / np.sqrt(var_len)])
        
        # Compute weights: h = 1, 2, 3, ... (1-indexed horizon)
        h = np.arange(1, full_horizon + 1)
        w_var = np.exp(-decay * h)
        w_uni = 1 - w_var
        
        # Normalise (defensive, should already sum to 1)
        w_total = w_var + w_uni
        w_var = w_var / w_total
        w_uni = w_uni / w_total
        
        # Handle NaN protection
        var_vals = np.nan_to_num(var_vals, nan=uni_vals)
        var_ci_lo = np.nan_to_num(var_ci_lo, nan=uni_ci_lo)
        var_ci_hi = np.nan_to_num(var_ci_hi, nan=uni_ci_hi)
        
        # Blend
        blended_values = w_var * var_vals + w_uni * uni_vals
        blended_ci_lower = w_var * var_ci_lo + w_uni * uni_ci_lo
        blended_ci_upper = w_var * var_ci_hi + w_uni * uni_ci_hi
        
        # Log blend info
        logger.info(f"    Blend weights: h=1 VAR={w_var[0]:.0%}, h=5 VAR={w_var[4]:.0%}, h=10 VAR={w_var[9]:.0%}")
        
        return {
            'values': blended_values,
            'ci_lower': blended_ci_lower,
            'ci_upper': blended_ci_upper,
            'years': uni_forecast['years'],  # Use univariate years (full horizon)
            'method': f"{var_method}+{uni_method}+blend",
            'var_weight_decay': decay,
            'var_horizon_used': var_len
        }
    
    def _run_forecasts_with_systems(self, data: pd.DataFrame, tasks: List[Dict]) -> List[Dict]:
        results = []
        processed = set()
        task_metrics = {t['metric'] for t in tasks}
        
        for system_name, system_metrics in self.config.var_systems.items():
            available = [m for m in system_metrics if m in task_metrics]
            if len(available) >= 2 and self.config.use_var_systems:
                logger.info(f"\n  VAR system: {system_name} ({available})")
                
                # Get full horizon and VAR horizon
                full_horizons = {t['metric']: t['horizon'] for t in tasks if t['metric'] in available}
                max_full_horizon = max(full_horizons.values())
                var_horizon = min(max_full_horizon, self.config.var_max_horizon)
                
                if var_horizon > 0:
                    logger.info(f"    VAR horizon: {var_horizon}y, Full horizon: {max_full_horizon}y")
                    
                    # Run VAR for var_horizon
                    var_results = self.forecaster.var_forecaster.forecast_system(data, available, var_horizon, [])
                    
                    if var_results:
                        for metric in available:
                            processed.add(metric)
                            task = next(t for t in tasks if t['metric'] == metric)
                            full_horizon = task['horizon']
                            hist_years = set(data[data['metric'] == metric]['year'].unique())
                            
                            var_forecast = var_results[metric]
                            var_method = var_forecast['method']
                            
                            # Apply growth caps to VAR first
                            var_forecast = self._apply_growth_caps_to_var(
                                var_forecast, 
                                data[data['metric'] == metric].sort_values('year'), 
                                metric, 
                                task['metric_info']
                            )
                            
                            # Run univariate for FULL horizon
                            series_data = data[data['metric'] == metric].sort_values('year')
                            series = pd.Series(
                                series_data['value'].values, 
                                index=series_data['year'].values.astype(int), 
                                name=metric
                            )
                            use_log = self.config.use_log_transform.get(metric, False)
                            working = np.log(series) if use_log and (series > 0).all() else series
                            breaks = [b for b in self.config.structural_breaks if task['last_year'] - 20 <= b['year'] <= task['last_year']]
                            
                            uni_forecast = self.forecaster.forecast_univariate(working, full_horizon, breaks, task['metric_info'])
                            uni_method = uni_forecast['method']
                            
                            # Convert univariate to level space if needed
                            uni_vals = np.asarray(uni_forecast['values'])
                            uni_ci_lo = np.asarray(uni_forecast['ci_lower'])
                            uni_ci_hi = np.asarray(uni_forecast['ci_upper'])
                            if use_log:
                                uni_vals = np.exp(uni_vals)
                                uni_ci_lo = np.exp(uni_ci_lo)
                                uni_ci_hi = np.exp(uni_ci_hi)
                            
                            uni_forecast_level = {
                                'values': uni_vals,
                                'ci_lower': uni_ci_lo,
                                'ci_upper': uni_ci_hi,
                                'years': uni_forecast['years'],
                                'method': uni_method
                            }
                            
                            # Blend VAR and univariate
                            blended = self._blend_var_univariate(
                                var_forecast,
                                uni_forecast_level,
                                full_horizon,
                                var_method,
                                uni_method
                            )
                            
                            # Emit results
                            for j, year in enumerate(blended['years']):
                                if year not in hist_years:
                                    results.append({
                                        'region_code': 'K02000001',
                                        'region': 'United Kingdom',
                                        'metric': metric,
                                        'year': year,
                                        'value': blended['values'][j],
                                        'ci_lower': blended['ci_lower'][j],
                                        'ci_upper': blended['ci_upper'][j],
                                        'method': blended['method'],
                                        'data_type': 'forecast',
                                        'source': 'model'
                                    })
                        
                        logger.info(f"    ✓ VAR+Blend completed: {system_name}")
        
        # Univariate for remaining metrics
        for task in tasks:
            if task['metric'] in processed:
                continue
            logger.info(f"\n  Processing {task['metric']} (univariate)...")
            series_data = data[data['metric'] == task['metric']].sort_values('year')
            hist_years = set(series_data['year'].unique())
            series = pd.Series(series_data['value'].values, index=series_data['year'].values.astype(int), name=task['metric'])
            use_log = self.config.use_log_transform.get(task['metric'], False)
            working = np.log(series) if use_log and (series > 0).all() else series
            breaks = [b for b in self.config.structural_breaks if task['last_year'] - 20 <= b['year'] <= task['last_year']]
            
            try:
                forecast = self.forecaster.forecast_univariate(working, task['horizon'], breaks, task['metric_info'])
                fc_vals = np.asarray(forecast['values'])
                fc_ci_lo = np.asarray(forecast['ci_lower'])
                fc_ci_hi = np.asarray(forecast['ci_upper'])
                if use_log:
                    fc_vals, fc_ci_lo, fc_ci_hi = np.exp(fc_vals), np.exp(fc_ci_lo), np.exp(fc_ci_hi)
                for j, year in enumerate(forecast['years']):
                    if year not in hist_years:
                        results.append({
                            'region_code': 'K02000001',
                            'region': 'United Kingdom',
                            'metric': task['metric'],
                            'year': year,
                            'value': fc_vals[j],
                            'ci_lower': fc_ci_lo[j],
                            'ci_upper': fc_ci_hi[j],
                            'method': forecast['method'],
                            'data_type': 'forecast',
                            'source': 'model'
                        })
            except Exception as e:
                logger.error(f"Forecast failed for {task['metric']}: {e}")
        
        return results
    
    def _apply_growth_caps_to_var(self, forecast: Dict, historical: pd.DataFrame, metric: str, metric_info: Dict) -> Dict:
        if metric_info.get('type') == 'rate':
            return forecast
        growth_cap = GROWTH_CAPS.get(metric)
        if not growth_cap:
            return forecast
        
        values = np.asarray(forecast['values']).copy()
        ci_lower = np.asarray(forecast['ci_lower']).copy()
        ci_upper = np.asarray(forecast['ci_upper']).copy()
        
        min_g, max_g = growth_cap['min'], growth_cap['max']
        last_hist = historical['value'].iloc[-1]
        prev = last_hist
        capped = 0
        
        for i in range(len(values)):
            if prev <= 0:
                prev = values[i]
                continue
            g = (values[i] - prev) / prev
            if g > max_g:
                values[i] = prev * (1 + max_g)
                capped += 1
            elif g < min_g:
                values[i] = prev * (1 + min_g)
                capped += 1
            prev = values[i]
        
        if capped > 0:
            logger.info(f"    {metric}: Capped {capped} VAR years to [{min_g*100:.1f}%, {max_g*100:.1f}%]")
            hist_vol = historical['value'].pct_change().dropna().std()
            if np.isfinite(hist_vol) and hist_vol > 0:
                for i in range(len(values)):
                    ci_w = 1.96 * values[i] * hist_vol * np.sqrt(i + 1)
                    ci_lower[i] = max(0, values[i] - ci_w)
                    ci_upper[i] = values[i] + ci_w
        
        forecast['values'] = values
        forecast['ci_lower'] = ci_lower
        forecast['ci_upper'] = ci_upper
        return forecast
    
    def _save_outputs(self, base_data: pd.DataFrame, derived_data: pd.DataFrame):
        """
        Save outputs:
        - Base metrics → gold.uk_macro_forecast
        - Derived metrics → gold.uk_macro_derived
        """
        data = base_data.copy()
        
        # CI cleanup
        if "ci_lower" in data.columns and "ci_upper" in data.columns:
            swap = (data["ci_lower"].notna() & data["ci_upper"].notna() & (data["ci_lower"] > data["ci_upper"]))
            if swap.any():
                tmp = data.loc[swap, "ci_lower"].copy()
                data.loc[swap, "ci_lower"] = data.loc[swap, "ci_upper"]
                data.loc[swap, "ci_upper"] = tmp
        
        # Long (base only)
        data.to_csv(self.config.output_dir / "uk_macro_forecast_long.csv", index=False)
        logger.info(f"✓ Long: {self.config.output_dir / 'uk_macro_forecast_long.csv'}")
        
        # Derived metrics CSV
        if not derived_data.empty:
            derived_data.to_csv(self.config.output_dir / "uk_macro_derived.csv", index=False)
            logger.info(f"✓ Derived: {self.config.output_dir / 'uk_macro_derived.csv'}")
        
        # Wide (base only)
        wide = data.pivot_table(index='metric', columns='year', values='value', aggfunc='first').reset_index()
        wide.columns.name = None
        wide.to_csv(self.config.output_dir / "uk_macro_forecast_wide.csv", index=False)
        logger.info(f"✓ Wide: {self.config.output_dir / 'uk_macro_forecast_wide.csv'}")
        
        # CIs (base only)
        if 'ci_lower' in data.columns and 'ci_upper' in data.columns:
            ci = data[['metric', 'year', 'value', 'ci_lower', 'ci_upper']].dropna(subset=['ci_lower', 'ci_upper'])
            if not ci.empty:
                ci['ci_width'] = ci['ci_upper'] - ci['ci_lower']
                ci.to_csv(self.config.output_dir / "uk_macro_confidence_intervals.csv", index=False)
                logger.info(f"✓ CIs: {self.config.output_dir / 'uk_macro_confidence_intervals.csv'}")
        
        # Metadata
        meta = {
            'run_timestamp': datetime.now().isoformat(),
            'version': '3.5',
            'architecture': {
                'base_table': 'gold.uk_macro_forecast',
                'derived_table': 'gold.uk_macro_derived',
                'forecasted_metrics': [m for m in self.config.metric_definitions.keys()],
                'computed_metrics': COMPUTED_METRICS,  # gdhi_per_head (in BASE)
                'derived_metrics': DERIVED_METRICS     # productivity, income_per_worker (in DERIVED)
            },
            'blend_config': {
                'var_max_horizon': self.config.var_max_horizon,
                'blend_decay': self.config.var_blend_decay,
                'description': 'Exponential decay blend: VAR strong early, fades to univariate'
            },
            'growth_caps': GROWTH_CAPS,
            'sanity_caps': SANITY_CAPS,
            'config': {
                'target_year': self.config.target_year,
                'var_enabled': self.config.use_var_systems
            },
            'data_summary': {
                'base_metrics': int(data['metric'].nunique()),
                'derived_metrics': int(derived_data['metric'].nunique()) if not derived_data.empty else 0,
                'base_forecast_obs': len(data[data['data_type'] == 'forecast']),
                'derived_obs': len(derived_data)
            }
        }
        with open(self.config.output_dir / "uk_macro_metadata.json", 'w') as f:
            json.dump(meta, f, indent=2, default=str)
        logger.info(f"✓ Metadata: {self.config.output_dir / 'uk_macro_metadata.json'}")
        
        # DuckDB - TWO TABLES
        if HAVE_DUCKDB:
            try:
                con = duckdb.connect(str(self.config.duckdb_path))
                con.execute("CREATE SCHEMA IF NOT EXISTS gold")
                
                # === BASE METRICS TABLE ===
                dc = data.copy()
                dc["metric_id"] = dc.get("metric_id", dc["metric"]).fillna(dc["metric"])
                dc["period"] = dc.get("period", dc["year"]).fillna(dc["year"])
                dc["region_level"] = "UK"
                dc["region_name"] = "United Kingdom"
                dc["region_code"] = dc.get("region_code", "K02000001").fillna("K02000001")
                dc["forecast_run_date"] = datetime.now().date()
                dc["forecast_version"] = "3.5"
                if 'unit' not in dc.columns:
                    dc['unit'] = dc['metric_id'].map({m: self.config.metric_definitions.get(m, {}).get('unit', 'unknown') for m in dc['metric_id'].unique()})
                if 'freq' not in dc.columns:
                    dc['freq'] = 'A'
                
                cols = ["region_code", "region_name", "region_level", "metric_id", "period", "value", "unit", "freq", "data_type", "ci_lower", "ci_upper", "forecast_run_date", "forecast_version"]
                flat = dc[[c for c in cols if c in dc.columns]].reset_index(drop=True)
                
                con.register("macro_df", flat)
                con.execute("CREATE OR REPLACE TABLE gold.uk_macro_forecast AS SELECT * FROM macro_df")
                con.execute("CREATE OR REPLACE VIEW gold.uk_macro_forecast_only AS SELECT * FROM gold.uk_macro_forecast WHERE data_type = 'forecast'")
                logger.info(f"✓ DuckDB: gold.uk_macro_forecast ({len(flat)} rows)")
                
                # === DERIVED METRICS TABLE ===
                if not derived_data.empty:
                    dd = derived_data.copy()
                    dd["metric_id"] = dd["metric"]
                    dd["period"] = dd["year"]
                    dd["region_level"] = "UK"
                    dd["region_name"] = "United Kingdom"
                    dd["region_code"] = dd.get("region_code", "K02000001").fillna("K02000001")
                    dd["forecast_run_date"] = datetime.now().date()
                    dd["forecast_version"] = "3.5"
                    dd["unit"] = "GBP"
                    dd["freq"] = "A"
                    
                    derived_cols = ["region_code", "region_name", "region_level", "metric_id", "period", "value", "unit", "freq", "data_type", "ci_lower", "ci_upper", "method", "formula", "forecast_run_date", "forecast_version"]
                    derived_flat = dd[[c for c in derived_cols if c in dd.columns]].reset_index(drop=True)
                    
                    con.register("derived_df", derived_flat)
                    con.execute("CREATE OR REPLACE TABLE gold.uk_macro_derived AS SELECT * FROM derived_df")
                    logger.info(f"✓ DuckDB: gold.uk_macro_derived ({len(derived_flat)} rows)")
                
                con.close()
            except Exception as e:
                logger.warning(f"DuckDB save failed: {e}")


def main():
    try:
        config = MacroForecastConfig()
        logger.info("="*70)
        logger.info("UK MACRO FORECAST V3.5 - BLENDED VAR")
        logger.info("="*70)
        logger.info(f"  Silver: {config.silver_path}")
        logger.info(f"  Target: {config.target_year}")
        logger.info(f"  VAR horizon: {config.var_max_horizon}y (blended to full horizon)")
        logger.info(f"  Blend decay: {config.var_blend_decay} (VAR ~{np.exp(-config.var_blend_decay*5)*100:.0f}% at h=5)")
        logger.info(f"  Growth caps: {list(GROWTH_CAPS.keys())}")
        logger.info(f"  Computed (in base): {COMPUTED_METRICS}")
        logger.info(f"  Derived (separate table): {DERIVED_METRICS}")
        
        forecaster = MacroForecasterV3(config)
        base_data, derived_data = forecaster.run()
        
        logger.info("="*70)
        logger.info("✅ UK MACRO V3.5 COMPLETED")
        logger.info(f"📊 Forecasted metrics: {len(config.metric_definitions)}")
        logger.info(f"📊 Computed metrics: {len(COMPUTED_METRICS)} (gdhi_per_head)")
        logger.info(f"📊 Derived metrics: {derived_data['metric'].nunique() if not derived_data.empty else 0}")
        logger.info(f"📊 Total base metrics: {base_data['metric'].nunique()}")
        logger.info(f"📊 Base forecasts: {len(base_data[base_data['data_type']=='forecast'])}")
        
        logger.info("\n📊 2050 SUMMARY (base):")
        for m in ['uk_nominal_gva_mn_gbp', 'uk_gdhi_total_mn_gbp', 'uk_gdhi_per_head_gbp', 'uk_emp_total_jobs', 'uk_population_total', 'uk_population_16_64']:
            v = base_data[(base_data['metric'] == m) & (base_data['year'] == 2050)]['value']
            if not v.empty:
                logger.info(f"    {m}: {v.iloc[0]:,.0f}")
        
        logger.info("\n📊 2050 SUMMARY (derived):")
        if not derived_data.empty:
            for m in DERIVED_METRICS:
                v = derived_data[(derived_data['metric'] == m) & (derived_data['year'] == 2050)]['value']
                if not v.empty:
                    logger.info(f"    {m}: {v.iloc[0]:,.0f}")
        
        logger.info("="*70)
        return base_data, derived_data
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()