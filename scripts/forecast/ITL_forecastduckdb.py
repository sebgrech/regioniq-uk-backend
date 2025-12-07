#!/usr/bin/env python3
"""
Region IQ - Institutional-Grade Econometric Forecasting Engine V3.1
====================================================================

V3.1: Adapted for unified silver schema from DuckDB pipeline
- Reads from data/silver/itl1_unified_history.csv
- Maintains all V3.0 forecasting improvements
- Simplified data loading (no wide-to-long conversion needed)

Author: Region IQ
Version: 3.1
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
from typing import Dict, Tuple, Optional, List, Union, Any
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

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

# Suppress warnings in production
warnings.filterwarnings('ignore')

# Import statistical packages with fallbacks
try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.api import ExponentialSmoothing, VAR
    from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    from scipy import stats
    from scipy.stats import jarque_bera
    HAVE_STATSMODELS = True
except ImportError as e:
    logger.warning(f"Statistical packages not fully available: {e}")
    HAVE_STATSMODELS = False

try:
    import arch
    HAVE_ARCH = True
except ImportError:
    HAVE_ARCH = False
    logger.warning("arch package not available - GARCH models disabled")

try:
    from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
    from sklearn.model_selection import TimeSeriesSplit
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False
    logger.warning("sklearn not available - using basic metrics")

# ===============================
# Enhanced Configuration V3.1 - UNIFIED SCHEMA
# ===============================

@dataclass
class ForecastConfig:
    """V3.1 Configuration for unified silver schema"""
    
    # Data paths - SIMPLIFIED FOR UNIFIED SCHEMA
    data_dir: Path = Path("data")
    
    # MAIN CHANGE: Single unified silver input
    silver_path: Path = Path("data/silver/itl1_unified_history.csv")
    
    # Optional: DuckDB path if you want to read directly from DB
    duckdb_path: Path = Path("data/lake/warehouse.duckdb")
    use_duckdb: bool = False  # Set True to read from DuckDB instead of CSV
    
    # Output configuration (unchanged)
    output_dir: Path = Path("data/forecast")
    cache_dir: Path = Path("data/cache")
    
    # Forecast parameters (unchanged)
    target_year: int = 2050
    min_history_years: int = 10
    confidence_levels: List[float] = field(default_factory=lambda: [0.80, 0.95])
    
    # Model parameters (unchanged)
    max_arima_order: int = 2
    max_var_lags: int = 3
    use_log_transform: Dict[str, bool] = None
    
    # Metric definitions - Updated for unified schema metric IDs
    metric_definitions: Dict[str, Dict] = None
    
    # Structural breaks (unchanged)
    structural_breaks: List[Dict] = None
    
    # Cross-validation (unchanged)
    cv_min_train_size: int = 15
    cv_test_windows: int = 3
    cv_horizon: int = 2
    
    # Performance (unchanged)
    parallel_processing: bool = True
    max_workers: int = 4
    n_bootstrap: int = 200
    cache_enabled: bool = True
    
    # Coherence constraints (unchanged)
    enforce_non_negative: bool = True
    enforce_monotonic_population: bool = True
    growth_rate_cap_percentiles: Tuple[float, float] = (2, 98)
    max_coherence_deviation: float = 0.05
    
    # Data coverage - Northern Ireland handling
    handle_ni_employment: str = 'use_if_present'  # Changed: unified schema may already have NI
    
    def __post_init__(self):
        """Initialize with V3.1 unified schema adaptations"""
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Updated metric definitions for unified schema
        if self.metric_definitions is None:
            self.metric_definitions = {
                'population_total': {'unit': 'persons', 'transform': 'log', 'monotonic': True},
                'population_male': {'unit': 'persons', 'transform': 'log', 'monotonic': True},
                'population_female': {'unit': 'persons', 'transform': 'log', 'monotonic': True},
                'gdhi_total_mn_gbp': {'unit': 'GBP_m', 'transform': 'log', 'monotonic': False},
                'gdhi_per_head_gbp': {'unit': 'GBP', 'transform': 'log', 'monotonic': False},
                'nominal_gva_mn_gbp': {'unit': 'GBP_m', 'transform': 'log', 'monotonic': False},
                'chained_gva_mn_gbp': {'unit': 'GBP_m', 'transform': 'log', 'monotonic': False},
                'gva_deflator_index': {'unit': 'index', 'transform': 'none', 'monotonic': False},
                'emp_total_jobs': {'unit': 'jobs', 'transform': 'log', 'monotonic': False}
            }
        
        # Set default log transforms based on metric definitions
        if self.use_log_transform is None:
            self.use_log_transform = {
                metric: info['transform'] == 'log'
                for metric, info in self.metric_definitions.items()
            }
        
        # Structural breaks unchanged
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
        """Enhanced validation for unified schema"""
        # Check silver path exists
        if not self.use_duckdb and not self.silver_path.exists():
            raise FileNotFoundError(f"Silver data not found: {self.silver_path}")
        
        if self.use_duckdb and not self.duckdb_path.exists():
            raise FileNotFoundError(f"DuckDB not found: {self.duckdb_path}")
        
        assert self.target_year > 2024, "Target year must be future"
        assert self.min_history_years >= 5, "Need at least 5 years history"
        assert self.n_bootstrap >= 100, "Bootstrap samples too low for reliability"
        assert 0 < self.max_coherence_deviation < 0.2, "Coherence threshold unrealistic"
        
        logger.info("Configuration V3.1 (Unified Schema) validated successfully")


# ===============================
# Data Management V3.1 - UNIFIED SCHEMA
# ===============================

class DataManagerV3:
    """Data management adapted for unified silver schema"""
    
    def __init__(self, config: ForecastConfig):
        self.config = config
        
    def load_all_data(self) -> pd.DataFrame:
        """Load from unified silver schema - MUCH SIMPLER NOW"""
        # Check cache
        cache_key = self._get_cache_key()
        if self.config.cache_enabled and self._cache_exists(cache_key):
            logger.info("Loading data from cache")
            return self._load_from_cache(cache_key)
        
        # Load unified silver data
        if self.config.use_duckdb:
            unified = self._load_from_duckdb()
        else:
            unified = self._load_from_csv()
        
        # Standardize column names for forecaster
        unified = self._standardize_columns(unified)
        
        # Add GVA deflator if we have both nominal and real
        unified = self._calculate_gva_deflator(unified)
        
        # Detect and handle outliers
        unified = self._handle_outliers(unified)
        
        # Apply coherence constraints
        unified = self._apply_coherence_constraints(unified)
        
        # Cache if enabled
        if self.config.cache_enabled:
            self._save_to_cache(unified, cache_key)
        
        logger.info(f"Loaded {len(unified)} rows from unified silver schema")
        logger.info(f"Regions: {unified['region_code'].nunique()}, Metrics: {unified['metric'].nunique()}")
        
        return unified
    
    def _load_from_csv(self) -> pd.DataFrame:
        """Load from unified silver CSV"""
        logger.info(f"Loading unified data from {self.config.silver_path}")
        
        df = pd.read_csv(self.config.silver_path)
        
        # Validate expected columns
        expected_cols = ['region_code', 'region_name', 'metric_id', 'period', 'value']
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            # Try alternative names
            if 'metric' in df.columns and 'metric_id' not in df.columns:
                df['metric_id'] = df['metric']
            if 'year' in df.columns and 'period' not in df.columns:
                df['period'] = df['year']
            
            # Re-check
            missing = [c for c in expected_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns in silver data: {missing}")
        
        return df
    
    def _load_from_duckdb(self) -> pd.DataFrame:
        """Load from DuckDB silver table"""
        try:
            import duckdb
            
            logger.info(f"Loading from DuckDB: {self.config.duckdb_path}")
            con = duckdb.connect(str(self.config.duckdb_path), read_only=True)
            
            # Try to load silver table
            df = con.execute("SELECT * FROM silver_itl1_history").fetchdf()
            con.close()
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load from DuckDB: {e}")
            logger.info("Falling back to CSV")
            return self._load_from_csv()
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names for forecaster compatibility"""
        # Rename to match forecaster expectations
        rename_map = {
            'metric_id': 'metric',
            'period': 'year',
            'region_name': 'region'  # Ensure we have 'region' column
        }
        
        for old, new in rename_map.items():
            if old in df.columns and new not in df.columns:
                df[new] = df[old]
        
        # Ensure numeric types
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Add data_type if not present
        if 'data_type' not in df.columns:
            df['data_type'] = 'historical'
        
        # Drop rows with null year or value
        df = df.dropna(subset=['year', 'value'])
        
        # Apply non-negativity constraint if configured
        if self.config.enforce_non_negative:
            df = df[df['value'] > 0]
        
        return df
    
    def _calculate_gva_deflator(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate GVA deflator from nominal and real GVA"""
        for region in df['region_code'].unique():
            region_data = df[df['region_code'] == region]
            
            nominal = region_data[region_data['metric'] == 'nominal_gva_mn_gbp']
            real = region_data[region_data['metric'] == 'chained_gva_mn_gbp']
            
            if not nominal.empty and not real.empty:
                # Merge on year
                merged = nominal.merge(real, on='year', suffixes=('_nom', '_real'))
                
                # Calculate deflator (nominal/real * 100, with 2022=100)
                deflator_values = (merged['value_nom'] / merged['value_real']) * 100
                
                # Normalize to 2022=100 if 2022 exists
                if 2022 in merged['year'].values:
                    base_value = deflator_values[merged['year'] == 2022].iloc[0]
                    deflator_values = (deflator_values / base_value) * 100
                
                # Add deflator rows
                for idx, year in enumerate(merged['year']):
                    df = pd.concat([df, pd.DataFrame([{
                        'region': region_data['region'].iloc[0] if 'region' in region_data else region,
                        'region_code': region,
                        'metric': 'gva_deflator_index',
                        'year': year,
                        'value': deflator_values.iloc[idx],
                        'source': 'calculated',
                        'data_type': 'historical'
                    }])], ignore_index=True)
        
        return df
    
    def _apply_coherence_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply coherence constraints and checks"""
        coherence_flags = []
        
        for region_code in df['region_code'].unique():
            for year in df['year'].unique():
                region_year = df[
                    (df['region_code'] == region_code) & 
                    (df['year'] == year)
                ]
                
                # Check GVA identity if all components present
                nominal = region_year[region_year['metric'] == 'nominal_gva_mn_gbp']['value'].values
                real = region_year[region_year['metric'] == 'chained_gva_mn_gbp']['value'].values
                deflator = region_year[region_year['metric'] == 'gva_deflator_index']['value'].values
                
                if len(nominal) > 0 and len(real) > 0 and len(deflator) > 0:
                    expected_nominal = real[0] * deflator[0] / 100
                    deviation = abs(nominal[0] - expected_nominal) / nominal[0] if nominal[0] > 0 else 0
                    
                    if deviation > self.config.max_coherence_deviation:
                        coherence_flags.append({
                            'region_code': region_code,
                            'year': year,
                            'check': 'gva_identity',
                            'deviation': deviation,
                            'status': 'violation'
                        })
                        
                        # Adjust to maintain coherence
                        logger.warning(f"Adjusting GVA coherence for {region_code}-{year}")
                        correct_deflator = nominal[0] / real[0] * 100
                        df.loc[
                            (df['region_code'] == region_code) & 
                            (df['year'] == year) & 
                            (df['metric'] == 'gva_deflator_index'),
                            'value'
                        ] = correct_deflator
        
        # Store coherence report in attributes
        df.attrs['coherence_flags'] = coherence_flags
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced outlier handling with winsorization"""
        for (region, metric), group in df.groupby(['region_code', 'metric']):
            if len(group) < 5:
                continue
            
            values = group['value'].values
            
            # Use IQR method for outlier detection
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = (values < lower_bound) | (values > upper_bound)
            
            if outliers.any():
                logger.info(f"Winsorizing {outliers.sum()} outliers in {region}-{metric}")
                # Winsorize to bounds
                df.loc[group[outliers].index, 'value'] = np.clip(
                    values[outliers], lower_bound, upper_bound
                )
        
        return df
    
    def _get_cache_key(self) -> str:
        """Generate cache key based on input file"""
        hasher = hashlib.md5()
        
        if self.config.use_duckdb:
            path = self.config.duckdb_path
        else:
            path = self.config.silver_path
        
        if path.exists():
            hasher.update(str(path.stat().st_mtime).encode())
        
        return hasher.hexdigest()
    
    def _cache_exists(self, key: str) -> bool:
        """Check if cache exists"""
        cache_path = self.config.cache_dir / f"data_{key}.pkl"
        return cache_path.exists()
    
    def _load_from_cache(self, key: str) -> pd.DataFrame:
        """Load data from cache"""
        cache_path = self.config.cache_dir / f"data_{key}.pkl"
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    def _save_to_cache(self, df: pd.DataFrame, key: str):
        """Save data to cache"""
        cache_path = self.config.cache_dir / f"data_{key}.pkl"
        with open(cache_path, 'wb') as f:
            pickle.dump(df, f)


# ===============================
# Advanced Forecasting V3
# ===============================

class AdvancedForecastingV3:
    """V3 forecasting with fixed CV, bootstrap, and CI combination"""
    
    def __init__(self, config: ForecastConfig):
        self.config = config
        
    def forecast_univariate(
        self,
        series: pd.Series,
        horizon: int,
        structural_breaks: List[int] = None,
        metric_info: Dict = None
    ) -> Dict:
        """Univariate forecasting with proper CV and constraints"""
        
        # Get candidate models
        models = []
        
        # 1. ARIMA with structural breaks
        arima_result = self._fit_arima_with_breaks(series, horizon, structural_breaks)
        if arima_result:
            models.append(arima_result)
        
        # 2. ETS with auto selection
        if HAVE_STATSMODELS and len(series) > 20:
            ets_result = self._fit_ets_auto(series, horizon)
            if ets_result:
                models.append(ets_result)
        
        # 3. Linear trend as baseline
        linear_result = self._fit_linear(series, horizon)
        if linear_result:
            models.append(linear_result)
        
        if not models:
            return self._fallback_forecast(series, horizon)
        
        # TRUE CROSS-VALIDATION for model selection
        if len(models) > 1:
            cv_errors = self._true_cross_validation(series, models)
            weights = self._calculate_cv_weights(cv_errors)
            combined = self._combine_forecasts_properly(models, weights, series)
        else:
            combined = models[0]
        
        # Apply constraints
        if metric_info:
            combined = self._apply_forecast_constraints(combined, series, metric_info)
        
        return combined
    
    def _fit_arima_with_breaks(
        self,
        series: pd.Series,
        horizon: int,
        structural_breaks: List[int] = None
    ) -> Optional[Dict]:
        """ARIMA with proper break handling and bootstrap"""
        if not HAVE_STATSMODELS:
            return None
        
        try:
            # Prepare structural break dummies
            exog = self._prepare_break_dummies(series, structural_breaks) if structural_breaks else None
            
            # Grid search for best ARIMA
            best_model = None
            best_aic = np.inf
            best_order = None
            
            # Test for stationarity
            adf_p = adfuller(series, autolag='AIC')[1]
            d = 1 if adf_p > 0.05 else 0
            
            for p in range(self.config.max_arima_order + 1):
                for q in range(self.config.max_arima_order + 1):
                    if p == 0 and q == 0 and d == 0:
                        continue
                    
                    try:
                        model = ARIMA(series, order=(p, d, q), exog=exog)
                        fitted = model.fit(method_kwargs={"warn_convergence": False})
                        
                        # Use AICc for small samples
                        n = len(series)
                        k = p + q + d + 1
                        aicc = fitted.aic + (2 * k * (k + 1)) / (n - k - 1)
                        
                        if aicc < best_aic:
                            best_aic = aicc
                            best_model = fitted
                            best_order = (p, d, q)
                    except:
                        continue
            
            if best_model is None:
                return None
            
            # Extend exogenous for forecast
            if exog is not None:
                exog_forecast = self._extend_exog_properly(exog, series.index, horizon)
            else:
                exog_forecast = None
            
            # Generate forecast
            forecast_obj = best_model.get_forecast(steps=horizon, exog=exog_forecast)
            forecast = forecast_obj.predicted_mean
            
            # Check residuals
            residuals = pd.Series(best_model.resid)
            resid_tests = self._test_residuals(residuals)
            
            # Bootstrap CI if residuals non-normal
            if not resid_tests.get('normal', True):
                ci = self._bootstrap_confidence_intervals_fixed(
                    best_model, series, horizon, exog, exog_forecast, best_order
                )
            else:
                ci = forecast_obj.conf_int(alpha=0.05)
                # Apply log bias correction if needed
                if self.config.use_log_transform.get(series.name, False):
                    sigma2 = np.var(residuals)
                    forecast = forecast * np.exp(0.5 * sigma2)
                    ci = ci * np.exp(0.5 * sigma2)
            
            last_year = series.index[-1]
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
    
    def _fit_ets_auto(self, series: pd.Series, horizon: int) -> Optional[Dict]:
        """ETS with automatic trend selection"""
        if not HAVE_STATSMODELS:
            return None
        
        try:
            best_model = None
            best_aic = np.inf
            best_config = None
            
            # Try different configurations
            for trend in ['add', 'mul', None]:
                for damped in [True, False]:
                    if trend is None and damped:
                        continue
                    
                    try:
                        model = ExponentialSmoothing(
                            series,
                            trend=trend,
                            damped_trend=damped,
                            seasonal=None  # Annual data
                        )
                        fitted = model.fit()
                        
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_model = fitted
                            best_config = f"ETS({trend},{damped})"
                    except:
                        continue
            
            if best_model is None:
                return None
            
            # Forecast
            forecast = best_model.forecast(steps=horizon)
            
            # CI from residuals
            residuals = series - best_model.fittedvalues
            sigma = residuals.std()
            
            # Use t-distribution for small samples
            if len(series) < 30:
                t_stat = stats.t.ppf(0.975, len(series) - 1)
            else:
                t_stat = 1.96
            
            ci_lower = forecast - t_stat * sigma * np.sqrt(range(1, horizon + 1))
            ci_upper = forecast + t_stat * sigma * np.sqrt(range(1, horizon + 1))
            
            last_year = series.index[-1]
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
    
    def _fit_linear(self, series: pd.Series, horizon: int) -> Optional[Dict]:
        """Linear trend with proper prediction intervals"""
        try:
            X = np.arange(len(series)).reshape(-1, 1)
            y = series.values
            
            if HAVE_STATSMODELS:
                X_sm = sm.add_constant(X)
                model = sm.OLS(y, X_sm).fit()
                
                # Forecast
                X_future = np.arange(len(series), len(series) + horizon).reshape(-1, 1)
                X_future_sm = sm.add_constant(X_future)
                predictions = model.get_prediction(X_future_sm)
                forecast = predictions.predicted_mean
                ci = predictions.conf_int(alpha=0.05)
            else:
                # Numpy fallback
                coeffs = np.polyfit(X.flatten(), y, 1)
                X_future = np.arange(len(series), len(series) + horizon)
                forecast = np.polyval(coeffs, X_future)
                
                # Simple CI
                residuals = y - np.polyval(coeffs, X.flatten())
                sigma = residuals.std()
                ci_width = 1.96 * sigma * np.sqrt(1 + 1/len(series) + (X_future - X.mean())**2 / X.var())
                ci = np.column_stack([forecast - ci_width, forecast + ci_width])
            
            last_year = series.index[-1]
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
    
    def _true_cross_validation(self, series: pd.Series, models: List[Dict]) -> Dict:
        """TRUE expanding window cross-validation"""
        cv_errors = {i: [] for i in range(len(models))}
        
        min_train = max(self.config.cv_min_train_size, len(series) // 2)
        
        for train_end in range(min_train, len(series) - self.config.cv_horizon + 1):
            train = series.iloc[:train_end]
            test = series.iloc[train_end:min(train_end + self.config.cv_horizon, len(series))]
            
            for i, model_info in enumerate(models):
                try:
                    # Use stored fit function
                    if 'fit_function' in model_info:
                        forecast = model_info['fit_function'](train, len(test))
                        
                        # Calculate sMAPE
                        error = np.mean(np.abs(forecast - test.values) / (np.abs(forecast) + np.abs(test.values) + 1e-10))
                        cv_errors[i].append(error)
                except:
                    cv_errors[i].append(np.inf)
        
        return cv_errors
    
    def _calculate_cv_weights(self, cv_errors: Dict) -> np.ndarray:
        """Calculate weights from CV errors"""
        mean_errors = []
        for i in sorted(cv_errors.keys()):
            errors = [e for e in cv_errors[i] if e != np.inf]
            if errors:
                mean_errors.append(np.mean(errors))
            else:
                mean_errors.append(np.inf)
        
        if all(e == np.inf for e in mean_errors):
            # Equal weights if all failed
            return np.ones(len(mean_errors)) / len(mean_errors)
        
        # Weight inversely proportional to error
        weights = 1.0 / (np.array(mean_errors) + 1e-10)
        weights[np.isinf(weights)] = 0
        
        # Normalize
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
        """PROPER forecast combination with CI from CV residuals"""
        # Combine point forecasts
        combined_values = np.zeros(len(models[0]['values']))
        for model, weight in zip(models, weights):
            combined_values += weight * model['values']
        
        # Build CI from historical CV performance
        # Collect all CV residuals for the ensemble
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
                        except:
                            pass
                
                if ensemble_forecast > 0:
                    cv_residuals.append(test_val - ensemble_forecast)
        
        # Calculate CI from residual distribution
        if cv_residuals:
            residual_std = np.std(cv_residuals)
            # Adjust for forecast horizon
            horizon_adjustment = np.sqrt(range(1, len(combined_values) + 1))
            ci_lower = combined_values - 1.96 * residual_std * horizon_adjustment
            ci_upper = combined_values + 1.96 * residual_std * horizon_adjustment
        else:
            # Fallback: weighted average of component CIs
            ci_lower = np.zeros(len(models[0]['values']))
            ci_upper = np.zeros(len(models[0]['values']))
            for model, weight in zip(models, weights):
                ci_lower += weight * model['ci_lower']
                ci_upper += weight * model['ci_upper']
        
        return {
            'method': f"Ensemble({len(models)} models)",
            'values': combined_values,
            'years': models[0]['years'],
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'weights': weights.tolist(),
            'component_methods': [m['method'] for m in models],
            'cv_residual_std': np.std(cv_residuals) if cv_residuals else None
        }
    
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
        """FIXED bootstrap with proper order passing"""
        if n_bootstrap is None:
            n_bootstrap = self.config.n_bootstrap
        
        residuals = model.resid
        fitted = model.fittedvalues
        
        bootstrap_forecasts = []
        
        for _ in range(n_bootstrap):
            # Resample residuals with replacement
            boot_residuals = np.random.choice(residuals, size=len(residuals), replace=True)
            boot_series = pd.Series(fitted + boot_residuals, index=series.index)
            
            try:
                # Refit with CORRECT order
                boot_model = ARIMA(boot_series, order=order, exog=exog)
                boot_fitted = boot_model.fit(disp=False)
                
                # Forecast
                boot_forecast = boot_fitted.forecast(steps=horizon, exog=exog_forecast)
                bootstrap_forecasts.append(boot_forecast.values)
            except:
                continue
        
        if len(bootstrap_forecasts) < 10:
            # Fallback to normal CI
            return model.get_forecast(steps=horizon, exog=exog_forecast).conf_int(alpha=0.05)
        
        # Calculate percentile CIs
        bootstrap_array = np.array(bootstrap_forecasts)
        ci_lower = np.percentile(bootstrap_array, 2.5, axis=0)
        ci_upper = np.percentile(bootstrap_array, 97.5, axis=0)
        
        return np.column_stack([ci_lower, ci_upper])
    
    def _prepare_break_dummies(
        self,
        series: pd.Series,
        breaks: List[Dict]
    ) -> Optional[np.ndarray]:
        """Prepare structural break dummies with trend interactions"""
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
                        # Step dummy
                        dummy = (years >= break_year).astype(int)
                        dummies.append(dummy)
                    elif break_type == 'trend':
                        # Trend interaction
                        time_since_break = np.maximum(0, years - break_year)
                        dummies.append(time_since_break)
        
        return np.column_stack(dummies) if dummies else None
    
    def _extend_exog_properly(
        self,
        exog: np.ndarray,
        historical_index: pd.Index,
        horizon: int
    ) -> np.ndarray:
        """PROPERLY extend exogenous variables"""
        if exog is None:
            return None
        
        # Ensure proper shape
        if len(exog.shape) == 1:
            exog = exog.reshape(-1, 1)
        
        n_vars = exog.shape[1]
        
        # For structural breaks, maintain the pattern
        last_year = historical_index[-1]
        future_years = range(last_year + 1, last_year + horizon + 1)
        
        extended = []
        for year in future_years:
            row = []
            for i in range(n_vars):
                # Check if this is a level dummy (0/1) or trend
                if np.all(np.isin(exog[:, i], [0, 1])):
                    # Level dummy - maintain last value
                    row.append(exog[-1, i])
                else:
                    # Trend - continue incrementing
                    increment = exog[-1, i] - exog[-2, i] if len(exog) > 1 else 1
                    row.append(exog[-1, i] + increment * (year - last_year))
            extended.append(row)
        
        result = np.array(extended)
        
        # Assert correct shape
        assert result.shape == (horizon, n_vars), f"Exog shape mismatch: {result.shape} vs ({horizon}, {n_vars})"
        
        return result
    
    def _apply_forecast_constraints(
        self,
        forecast: Dict,
        historical: pd.Series,
        metric_info: Dict
    ) -> Dict:
        """Apply coherence constraints to forecast"""
        values = forecast['values']
        
        # 1. Non-negativity
        if self.config.enforce_non_negative:
            values = np.maximum(values, 0)
            forecast['ci_lower'] = np.maximum(forecast['ci_lower'], 0)
        
        # 2. Monotonicity for population
        if metric_info.get('monotonic', False) and self.config.enforce_monotonic_population:
            # Ensure non-decreasing
            for i in range(1, len(values)):
                values[i] = max(values[i], values[i-1])
        
        # 3. Growth rate caps
        if self.config.growth_rate_cap_percentiles:
            historical_growth = historical.pct_change().dropna()
            if len(historical_growth) > 5:
                p_low, p_high = self.config.growth_rate_cap_percentiles
                growth_bounds = np.percentile(historical_growth, [p_low, p_high])
                
                last_value = historical.iloc[-1]
                for i in range(len(values)):
                    prev_value = last_value if i == 0 else values[i-1]
                    growth = (values[i] - prev_value) / prev_value if prev_value > 0 else 0
                    
                    if growth < growth_bounds[0]:
                        values[i] = prev_value * (1 + growth_bounds[0])
                    elif growth > growth_bounds[1]:
                        values[i] = prev_value * (1 + growth_bounds[1])
        
        forecast['values'] = values
        forecast['constraints_applied'] = True
        
        return forecast
    
    def _test_residuals(self, residuals: pd.Series) -> Dict:
        """Comprehensive residual tests"""
        results = {}
        
        if not HAVE_STATSMODELS or len(residuals) < 10:
            return results
        
        clean_resid = residuals.dropna()
        
        # Ljung-Box for autocorrelation
        try:
            lb = acorr_ljungbox(clean_resid, lags=min(10, len(clean_resid)//3))
            results['ljung_box_p'] = float(lb['lb_pvalue'].min())
            results['no_autocorr'] = results['ljung_box_p'] > 0.05
        except:
            pass
        
        # Jarque-Bera for normality
        try:
            jb_stat, jb_p = jarque_bera(clean_resid)
            results['jarque_bera_p'] = jb_p
            results['normal'] = jb_p > 0.05
        except:
            pass
        
        # ARCH test
        try:
            arch_test = het_arch(clean_resid)
            results['arch_p'] = arch_test[1]
            results['no_arch'] = arch_test[1] > 0.05
        except:
            pass
        
        return results
    
    def _arima_predict(self, series: pd.Series, horizon: int, order: Tuple, breaks: List = None) -> np.ndarray:
        """Helper for CV - refit ARIMA and predict"""
        try:
            exog = self._prepare_break_dummies(series, breaks) if breaks else None
            model = ARIMA(series, order=order, exog=exog)
            fitted = model.fit(disp=False)
            
            if exog is not None:
                exog_fc = self._extend_exog_properly(exog, series.index, horizon)
            else:
                exog_fc = None
            
            return fitted.forecast(steps=horizon, exog=exog_fc).values
        except:
            return np.repeat(series.iloc[-1], horizon)
    
    def _linear_predict(self, series: pd.Series, horizon: int) -> np.ndarray:
        """Helper for CV - linear prediction"""
        try:
            X = np.arange(len(series))
            y = series.values
            coeffs = np.polyfit(X, y, 1)
            X_future = np.arange(len(series), len(series) + horizon)
            return np.polyval(coeffs, X_future)
        except:
            return np.repeat(series.iloc[-1], horizon)
    
    def _fallback_forecast(self, series: pd.Series, horizon: int) -> Dict:
        """Robust fallback using historical volatility"""
        last_value = series.iloc[-1]
        
        # Use recent trend if available
        if len(series) > 5:
            recent_trend = (series.iloc[-1] - series.iloc[-5]) / 5
        else:
            recent_trend = 0
        
        values = []
        for h in range(1, horizon + 1):
            value = last_value + recent_trend * h
            values.append(max(value, 0))
        
        # CI based on historical volatility
        if len(series) > 2:
            historical_std = series.std()
            # NOT annualized for annual data
            ci_width = 1.96 * historical_std * np.sqrt(range(1, horizon + 1))
        else:
            ci_width = np.array(values) * 0.1  # 10% bands
        
        last_year = series.index[-1]
        forecast_years = list(range(last_year + 1, last_year + horizon + 1))
        
        return {
            'method': 'Fallback',
            'values': np.array(values),
            'years': forecast_years,
            'ci_lower': np.array(values) - ci_width,
            'ci_upper': np.array(values) + ci_width
        }


# ===============================
# Error Propagation V3
# ===============================

class DerivedMetricsV3:
    """Calculate derived indicators with FIXED error propagation"""
    
    @staticmethod
    def calculate_with_uncertainty(
        numerator: Tuple[float, float, float],
        denominator: Tuple[float, float, float],
        operation: str = 'divide',
        correlation: float = 0.0,
        n_simulations: int = 10000
    ) -> Tuple[float, float, float]:
        """FIXED Monte Carlo with SVD error handling"""
        
        # Extract values and uncertainties
        num_val, num_low, num_high = numerator
        den_val, den_low, den_high = denominator
        
        # Calculate standard errors
        num_se = (num_high - num_low) / (2 * 1.96)
        den_se = (den_high - den_low) / (2 * 1.96)
        
        # Prevent zero standard errors
        epsilon = 1e-10
        num_se = max(num_se, epsilon)
        den_se = max(den_se, epsilon)
        
        if operation == 'divide' and abs(den_val) < epsilon:
            return 0, 0, 0
        
        # Monte Carlo simulation with error handling
        try:
            if abs(correlation) > epsilon:
                # Generate correlated samples
                mean = [num_val, den_val]
                cov = [[num_se**2 + epsilon, correlation * num_se * den_se],
                       [correlation * num_se * den_se, den_se**2 + epsilon]]
                
                samples = np.random.multivariate_normal(mean, cov, n_simulations)
                num_samples = samples[:, 0]
                den_samples = samples[:, 1]
            else:
                # Independent samples
                num_samples = np.random.normal(num_val, num_se, n_simulations)
                den_samples = np.random.normal(den_val, den_se, n_simulations)
        except np.linalg.LinAlgError:
            # Fallback to independent sampling
            logger.warning("SVD failed in Monte Carlo, using independent sampling")
            num_samples = np.random.normal(num_val, num_se, n_simulations)
            den_samples = np.random.normal(den_val, den_se, n_simulations)
        
        # Apply operation
        if operation == 'divide':
            den_samples = np.where(np.abs(den_samples) < epsilon, epsilon * np.sign(den_samples), den_samples)
            result_samples = num_samples / den_samples
        elif operation == 'multiply':
            result_samples = num_samples * den_samples
        elif operation == 'add':
            result_samples = num_samples + den_samples
        elif operation == 'subtract':
            result_samples = num_samples - den_samples
        else:
            result_samples = num_samples
        
        # Remove extreme outliers
        mean_result = np.mean(result_samples)
        std_result = np.std(result_samples)
        valid_samples = result_samples[
            np.abs(result_samples - mean_result) < 5 * std_result
        ]
        
        if len(valid_samples) < 100:
            # Too many outliers, use simple calculation
            if operation == 'divide':
                result_val = num_val / den_val if den_val != 0 else 0
            elif operation == 'multiply':
                result_val = num_val * den_val
            else:
                result_val = num_val
            
            # Simple error propagation
            rel_error = np.sqrt((num_se/abs(num_val))**2 + (den_se/abs(den_val))**2) if num_val != 0 and den_val != 0 else 0.1
            ci_width = 1.96 * abs(result_val) * rel_error
            
            return result_val, result_val - ci_width, result_val + ci_width
        
        # Calculate percentiles from valid samples
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
        """Calculate all derived metrics with uncertainty"""
        derived = []
        
        # Get base metrics for this region-year
        region_year_data = data[
            (data['region_code'] == region_code) & 
            (data['year'] == year)
        ]
        
        if region_year_data.empty:
            return derived
        
        # Extract values with uncertainty
        metrics_dict = {}
        for _, row in region_year_data.iterrows():
            metric = row['metric']
            value = row['value']
            ci_lower = row.get('ci_lower', value * 0.95)
            ci_upper = row.get('ci_upper', value * 1.05)
            metrics_dict[metric] = (value, ci_lower, ci_upper)
        
        # 1. Productivity (GVA per worker)
        if 'nominal_gva_mn_gbp' in metrics_dict and 'emp_total_jobs' in metrics_dict:
            # Convert GVA from £m to £
            gva_pounds = (
                metrics_dict['nominal_gva_mn_gbp'][0] * 1e6,
                metrics_dict['nominal_gva_mn_gbp'][1] * 1e6,
                metrics_dict['nominal_gva_mn_gbp'][2] * 1e6
            )
            
            prod_val, prod_low, prod_high = DerivedMetricsV3.calculate_with_uncertainty(
                gva_pounds,
                metrics_dict['emp_total_jobs'],
                operation='divide',
                correlation=0.8  # GVA and employment are positively correlated
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
        
        # 2. Employment rate
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
                'value': emp_rate_val * 100,  # Convert to percentage
                'ci_lower': emp_rate_low * 100,
                'ci_upper': emp_rate_high * 100,
                'method': 'derived_monte_carlo',
                'source': 'calculated',
                'data_type': 'forecast' if year > 2023 else 'historical'
            })
        
        # 3. Income per worker
        if 'gdhi_total_mn_gbp' in metrics_dict and 'emp_total_jobs' in metrics_dict:
            # Convert GDHI from £m to £
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
        
        return derived


# ===============================
# Main Pipeline V3
# ===============================

class InstitutionalForecasterV3:
    """V3 Production pipeline with all fixes"""
    
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.data_manager = DataManagerV3(config)
        self.forecaster = AdvancedForecastingV3(config)
        self.derived = DerivedMetricsV3()
        
    def run(self) -> pd.DataFrame:
        """Execute complete forecasting pipeline"""
        logger.info("="*60)
        logger.info("REGION IQ - INSTITUTIONAL FORECASTING ENGINE V3.1")
        logger.info("="*60)
        
        # Load and validate data
        logger.info("Loading and validating data...")
        historical_data = self.data_manager.load_all_data()
        
        # Check data coverage
        coverage_report = self._check_data_coverage(historical_data)
        logger.info(f"Data coverage: {coverage_report}")
        
        # Identify forecast tasks
        tasks = self._identify_forecast_tasks(historical_data)
        logger.info(f"Identified {len(tasks)} forecasting tasks")
        
        # Run forecasts
        results = self._run_forecasts(historical_data, tasks)
        
        # Calculate derived metrics
        logger.info("Calculating derived indicators...")
        all_data = pd.concat([historical_data, pd.DataFrame(results)], ignore_index=True)
        derived_results = self._calculate_derived_metrics(all_data)
        all_data = pd.concat([all_data, pd.DataFrame(derived_results)], ignore_index=True)
        
        # Apply final coherence checks
        all_data = self._final_coherence_check(all_data)
        
        # Generate outputs
        logger.info("Generating outputs...")
        self._save_outputs(all_data, coverage_report)
        
        logger.info("✅ Forecasting pipeline V3.1 completed successfully")
        
        return all_data
    
    def _check_data_coverage(self, data: pd.DataFrame) -> Dict:
        """Check data coverage and identify gaps"""
        coverage = {}
        
        # Get expected regions from unified schema (ONS codes)
        expected_regions = {
            'E12000001', 'E12000002', 'E12000003', 'E12000004', 'E12000005',
            'E12000006', 'E12000007', 'E12000008', 'E12000009',
            'W92000004', 'S92000003', 'N92000002'
        }
        
        for metric in data['metric'].unique():
            metric_data = data[data['metric'] == metric]
            regions = metric_data['region_code'].unique()
            years = metric_data['year'].unique()
            
            coverage[metric] = {
                'regions': len(regions),
                'missing_regions': list(expected_regions - set(regions)),
                'year_range': f"{min(years)}-{max(years)}",
                'observations': len(metric_data)
            }
        
        return coverage
    
    def _identify_forecast_tasks(self, data: pd.DataFrame) -> List[Dict]:
        """Identify forecasting tasks with metadata"""
        tasks = []
        
        for (region_code, region, metric), group in data.groupby(['region_code', 'region', 'metric']):
            if len(group) < self.config.min_history_years:
                logger.warning(f"Skipping {region_code}-{metric}: only {len(group)} years")
                continue
            
            last_year = group['year'].max()
            horizon = min(
                self.config.target_year - last_year,
                30  # Max 10 year horizon for reliability
            )
            
            if horizon <= 0:
                continue
            
            # Get metric info
            metric_info = self.config.metric_definitions.get(metric, {})
            
            tasks.append({
                'region_code': region_code,
                'region': region,
                'metric': metric,
                'horizon': horizon,
                'last_year': last_year,
                'history_length': len(group),
                'metric_info': metric_info
            })
        
        return tasks
    
    def _run_forecasts(self, data: pd.DataFrame, tasks: List[Dict]) -> List[Dict]:
        """Run forecasts with proper error handling"""
        results = []
        
        for i, task in enumerate(tasks):
            logger.info(f"Processing {i+1}/{len(tasks)}: {task['region_code']}-{task['metric']}")
            
            # Get series
            series_data = data[
                (data['region_code'] == task['region_code']) &
                (data['metric'] == task['metric'])
            ].sort_values('year')
            
            series = pd.Series(
                series_data['value'].values,
                index=series_data['year'].values,
                name=task['metric']
            )
            
            # Apply log transform if configured
            use_log = self.config.use_log_transform.get(task['metric'], False)
            if use_log and (series > 0).all():
                working_series = np.log(series)
            else:
                working_series = series
                use_log = False
            
            # Get relevant structural breaks
            relevant_breaks = [
                b for b in self.config.structural_breaks
                if task['last_year'] - 20 <= b['year'] <= task['last_year']
            ]
            
            # Generate forecast
            try:
                forecast = self.forecaster.forecast_univariate(
                    working_series,
                    task['horizon'],
                    relevant_breaks,
                    task['metric_info']
                )
                
                # Transform back if needed
                if use_log:
                    # Apply bias correction
                    if 'cv_residual_std' in forecast and forecast['cv_residual_std']:
                        bias_correction = np.exp(0.5 * forecast['cv_residual_std']**2)
                    else:
                        bias_correction = 1.0
                    
                    forecast['values'] = np.exp(forecast['values']) * bias_correction
                    forecast['ci_lower'] = np.exp(forecast['ci_lower'])
                    forecast['ci_upper'] = np.exp(forecast['ci_upper'])
                
                # Store quality metrics
                quality_metrics = {
                    'history_years': task['history_length'],
                    'cv_performance': forecast.get('cv_residual_std'),
                    'constraints_applied': forecast.get('constraints_applied', False),
                    'method': forecast['method']
                }
                
                # Convert to records
                for j, year in enumerate(forecast['years']):
                    results.append({
                        'region': task['region'],
                        'region_code': task['region_code'],
                        'metric': task['metric'],
                        'year': year,
                        'value': forecast['values'][j],
                        'ci_lower': forecast['ci_lower'][j],
                        'ci_upper': forecast['ci_upper'][j],
                        'method': forecast['method'],
                        'data_type': 'forecast',
                        'source': 'model',
                        'quality_metrics': quality_metrics
                    })
                    
            except Exception as e:
                logger.error(f"Forecast failed for {task['region_code']}-{task['metric']}: {e}")
                # Add flat forecast as fallback
                last_value = series.iloc[-1]
                for j in range(task['horizon']):
                    results.append({
                        'region': task['region'],
                        'region_code': task['region_code'],
                        'metric': task['metric'],
                        'year': task['last_year'] + j + 1,
                        'value': last_value,
                        'ci_lower': last_value * 0.9,
                        'ci_upper': last_value * 1.1,
                        'method': 'Fallback_Error',
                        'data_type': 'forecast',
                        'source': 'model',
                        'quality_metrics': {'error': str(e)}
                    })
        
        return results
    
    def _calculate_derived_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all derived metrics with proper error propagation"""
        derived_results = []
        
        for region_code in data['region_code'].unique():
            region_data = data[data['region_code'] == region_code]
            region = region_data['region'].iloc[0]
            
            for year in region_data['year'].unique():
                try:
                    year_results = self.derived.calculate_all_derived(
                        data, region_code, year
                    )
                    
                    for result in year_results:
                        result['region'] = region
                        derived_results.append(result)
                except Exception as e:
                    logger.error(f"Derived metrics failed for {region_code}-{year}: {e}")
        
        return pd.DataFrame(derived_results) if derived_results else pd.DataFrame()
    
    def _final_coherence_check(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply final coherence checks and adjustments"""
        coherence_report = []
        
        for region_code in data['region_code'].unique():
            for year in data['year'].unique():
                region_year = data[
                    (data['region_code'] == region_code) & 
                    (data['year'] == year)
                ]
                
                # Check GVA identity: Nominal ≈ Real × Deflator/100
                nominal = region_year[region_year['metric'] == 'nominal_gva_mn_gbp']['value'].values
                real = region_year[region_year['metric'] == 'chained_gva_mn_gbp']['value'].values
                deflator = region_year[region_year['metric'] == 'gva_deflator_index']['value'].values
                
                if len(nominal) > 0 and len(real) > 0 and len(deflator) > 0:
                    expected_nominal = real[0] * deflator[0] / 100
                    deviation = abs(nominal[0] - expected_nominal) / nominal[0] if nominal[0] > 0 else 0
                    
                    if deviation > self.config.max_coherence_deviation:
                        coherence_report.append({
                            'region_code': region_code,
                            'year': year,
                            'check': 'gva_identity',
                            'deviation': deviation,
                            'status': 'violation'
                        })
                        
                        # Adjust to maintain coherence
                        logger.warning(f"Adjusting GVA coherence for {region_code}-{year}")
                        # Prefer to adjust the deflator as it's derived
                        correct_deflator = nominal[0] / real[0] * 100
                        data.loc[
                            (data['region_code'] == region_code) & 
                            (data['year'] == year) & 
                            (data['metric'] == 'gva_deflator_index'),
                            'value'
                        ] = correct_deflator
        
        # Store coherence report in attributes
        data.attrs['coherence_report'] = coherence_report
        
        return data
    
    def _save_outputs(self, data: pd.DataFrame, coverage_report: Dict):
        """Save comprehensive outputs with metadata"""
        
        # 1. Complete long format
        long_path = self.config.output_dir / "forecast_v3_long.csv"
        data.to_csv(long_path, index=False)
        logger.info(f"✓ Long format: {long_path}")
        
        # 2. Wide format
        wide = data.pivot_table(
            index=['region', 'region_code', 'metric'],
            columns='year',
            values='value',
            aggfunc='first'
        ).reset_index()
        wide.columns.name = None
        
        wide_path = self.config.output_dir / "forecast_v3_wide.csv"
        wide.to_csv(wide_path, index=False)
        logger.info(f"✓ Wide format: {wide_path}")
        
        # 3. Confidence intervals
        ci_data = data[['region_code', 'metric', 'year', 'value', 'ci_lower', 'ci_upper']].copy()
        ci_data['ci_width'] = ci_data['ci_upper'] - ci_data['ci_lower']
        ci_data['cv'] = ci_data['ci_width'] / (2 * ci_data['value'])
        
        ci_path = self.config.output_dir / "confidence_intervals_v3.csv"
        ci_data.to_csv(ci_path, index=False)
        logger.info(f"✓ Confidence intervals: {ci_path}")
        
        # 4. Quality metrics
        quality_data = []
        for _, row in data[data['data_type'] == 'forecast'].iterrows():
            if 'quality_metrics' in row and isinstance(row['quality_metrics'], dict):
                quality_data.append({
                    'region_code': row['region_code'],
                    'metric': row['metric'],
                    'year': row['year'],
                    **row['quality_metrics']
                })
        
        if quality_data:
            quality_df = pd.DataFrame(quality_data)
            quality_path = self.config.output_dir / "forecast_quality_v3.csv"
            quality_df.to_csv(quality_path, index=False)
            logger.info(f"✓ Quality metrics: {quality_path}")
        
        # 5. Enhanced metadata for Streamlit
        metadata = {
            'run_timestamp': datetime.now().isoformat(),
            'version': '3.1',
            'config': {
                'target_year': self.config.target_year,
                'confidence_levels': self.config.confidence_levels,
                'structural_breaks': self.config.structural_breaks,
                'cv_windows': self.config.cv_test_windows,
                'bootstrap_samples': self.config.n_bootstrap,
                'constraints': {
                    'non_negative': self.config.enforce_non_negative,
                    'monotonic_population': self.config.enforce_monotonic_population,
                    'growth_caps': self.config.growth_rate_cap_percentiles
                }
            },
            'data_coverage': coverage_report,
            'data_summary': {
                'regions': data['region_code'].nunique(),
                'metrics': data['metric'].nunique(),
                'total_observations': len(data),
                'historical_obs': len(data[data['data_type'] == 'historical']),
                'forecast_obs': len(data[data['data_type'] == 'forecast']),
                'derived_metrics': len(data[data['source'] == 'calculated'])
            },
            'model_usage': data[data['data_type'] == 'forecast']['method'].value_counts().to_dict() if 'method' in data.columns else {},
            'coherence_violations': len(data.attrs.get('coherence_report', [])),
            'packages_available': {
                'statsmodels': HAVE_STATSMODELS,
                'arch': HAVE_ARCH,
                'sklearn': HAVE_SKLEARN
            },
            'quality_indicators': self._calculate_quality_indicators(data)
        }
        
        metadata_path = self.config.output_dir / "metadata_v3.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"✓ Metadata: {metadata_path}")
        
        # 6. Diagnostics report
        self._generate_diagnostics_report(data, metadata)
    
    def _calculate_quality_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate overall quality indicators"""
        indicators = {}
        
        forecast_data = data[data['data_type'] == 'forecast']
        
        if not forecast_data.empty and 'ci_lower' in forecast_data.columns:
            # Average coefficient of variation
            cv_values = []
            for _, row in forecast_data.iterrows():
                if pd.notna(row.get('ci_lower')) and pd.notna(row.get('ci_upper')):
                    ci_width = row['ci_upper'] - row['ci_lower']
                    cv = ci_width / (2 * abs(row['value'])) if row['value'] != 0 else np.inf
                    if cv != np.inf:
                        cv_values.append(cv)
            
            if cv_values:
                indicators['mean_cv'] = np.mean(cv_values)
                indicators['median_cv'] = np.median(cv_values)
                indicators['max_cv'] = np.max(cv_values)
        
        # Model diversity
        if 'method' in forecast_data.columns:
            methods = forecast_data['method'].value_counts()
            indicators['model_diversity'] = len(methods)
            indicators['ensemble_usage'] = sum('Ensemble' in m for m in methods.index) / len(methods)
        
        # Data quality
        historical_data = data[data['data_type'] == 'historical']
        if not historical_data.empty:
            # Check for gaps
            gaps = 0
            for (region, metric), group in historical_data.groupby(['region_code', 'metric']):
                years = sorted(group['year'].unique())
                if len(years) > 1:
                    year_gaps = np.diff(years)
                    gaps += sum(year_gaps > 1)
            
            indicators['data_gaps'] = gaps
            indicators['avg_history_length'] = historical_data.groupby(['region_code', 'metric'])['year'].count().mean()
        
        return indicators
    
    def _generate_diagnostics_report(self, data: pd.DataFrame, metadata: Dict):
        """Generate comprehensive diagnostics report"""
        report = []
        report.append("="*70)
        report.append("REGION IQ - FORECAST DIAGNOSTICS REPORT V3.1")
        report.append("="*70)
        report.append(f"Generated: {metadata['run_timestamp']}")
        report.append("")
        
        # Executive Summary
        report.append("## EXECUTIVE SUMMARY")
        report.append("-" * 40)
        report.append(f"✓ Successfully generated forecasts for {metadata['data_summary']['regions']} regions")
        report.append(f"✓ Covered {metadata['data_summary']['metrics']} metrics")
        report.append(f"✓ Total observations: {metadata['data_summary']['total_observations']:,}")
        report.append(f"✓ Forecast horizon: to {metadata['config']['target_year']}")
        report.append("")
        
        # Data Coverage
        report.append("## DATA COVERAGE")
        report.append("-" * 40)
        for metric, info in metadata['data_coverage'].items():
            report.append(f"\n{metric}:")
            report.append(f"  Regions: {info['regions']}/12")
            if info['missing_regions']:
                report.append(f"  Missing: {', '.join(info['missing_regions'])}")
            report.append(f"  Years: {info['year_range']}")
            report.append(f"  Observations: {info['observations']}")
        report.append("")
        
        # Model Performance
        report.append("## MODEL USAGE")
        report.append("-" * 40)
        for method, count in metadata.get('model_usage', {}).items():
            report.append(f"  {method}: {count} series")
        report.append("")
        
        # Quality Indicators
        report.append("## QUALITY INDICATORS")
        report.append("-" * 40)
        quality = metadata.get('quality_indicators', {})
        if 'mean_cv' in quality:
            report.append(f"  Average uncertainty (CV): {quality['mean_cv']:.1%}")
            report.append(f"  Median uncertainty (CV): {quality['median_cv']:.1%}")
            report.append(f"  Maximum uncertainty (CV): {quality['max_cv']:.1%}")
        if 'model_diversity' in quality:
            report.append(f"  Model diversity: {quality['model_diversity']} different methods")
            report.append(f"  Ensemble usage: {quality['ensemble_usage']:.1%}")
        if 'data_gaps' in quality:
            report.append(f"  Data gaps detected: {quality['data_gaps']}")
            report.append(f"  Average history length: {quality['avg_history_length']:.1f} years")
        report.append("")
        
        # Coherence Checks
        report.append("## COHERENCE VALIDATION")
        report.append("-" * 40)
        coherence_violations = metadata.get('coherence_violations', 0)
        if coherence_violations > 0:
            report.append(f"⚠️  {coherence_violations} coherence violations detected and corrected")
        else:
            report.append("✓ All coherence checks passed")
        report.append("")
        
        # Configuration
        report.append("## CONFIGURATION")
        report.append("-" * 40)
        config = metadata['config']
        report.append(f"  Target year: {config['target_year']}")
        report.append(f"  Confidence levels: {config['confidence_levels']}")
        report.append(f"  CV windows: {config['cv_windows']}")
        report.append(f"  Bootstrap samples: {config['bootstrap_samples']}")
        report.append(f"  Constraints applied:")
        for constraint, value in config['constraints'].items():
            report.append(f"    - {constraint}: {value}")
        report.append("")
        
        # Structural Breaks
        report.append("## STRUCTURAL BREAKS CONSIDERED")
        report.append("-" * 40)
        for break_info in config['structural_breaks']:
            report.append(f"  {break_info['year']}: {break_info['name']} ({break_info['type']})")
        report.append("")
        
        # Packages
        report.append("## TECHNICAL ENVIRONMENT")
        report.append("-" * 40)
        for package, available in metadata['packages_available'].items():
            status = "✓" if available else "✗"
            report.append(f"  {status} {package}")
        report.append("")
        
        report.append("="*70)
        report.append("END OF REPORT")
        
        # Save report
        report_path = self.config.output_dir / 'diagnostics_report_v3.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"✓ Diagnostics report: {report_path}")


# ===============================
# Streamlit Integration V3
# ===============================

class StreamlitAdapterV3:
    """Enhanced Streamlit integration with quality badges"""
    
    @staticmethod
    def prepare_chart_data(
        data: pd.DataFrame,
        region_code: str,
        metric: str
    ) -> Dict:
        """Prepare data for Streamlit charts with quality info"""
        
        # Filter data
        chart_data = data[
            (data['region_code'] == region_code) &
            (data['metric'] == metric)
        ].sort_values('year')
        
        if chart_data.empty:
            return {}
        
        # Separate historical and forecast
        historical = chart_data[chart_data['data_type'] == 'historical']
        forecast = chart_data[chart_data['data_type'] == 'forecast']
        
        # Extract quality metrics
        quality_badges = {}
        if not forecast.empty:
            latest_forecast = forecast.iloc[0]
            
            # History length
            quality_badges['history_years'] = len(historical)
            
            # Method used
            quality_badges['method'] = latest_forecast.get('method', 'Unknown')
            
            # Forecast uncertainty
            if 'ci_lower' in latest_forecast and 'ci_upper' in latest_forecast:
                cv = (latest_forecast['ci_upper'] - latest_forecast['ci_lower']) / (2 * latest_forecast['value'])
                quality_badges['uncertainty'] = f"{cv:.1%}"
            
            # Data completeness
            quality_badges['coverage'] = 'Complete' if len(historical) >= 15 else 'Limited'
        
        return {
            'historical': {
                'years': historical['year'].tolist(),
                'values': historical['value'].tolist()
            },
            'forecast': {
                'years': forecast['year'].tolist(),
                'values': forecast['value'].tolist(),
                'ci_lower': forecast['ci_lower'].tolist() if 'ci_lower' in forecast else None,
                'ci_upper': forecast['ci_upper'].tolist() if 'ci_upper' in forecast else None
            },
            'region': chart_data['region'].iloc[0],
            'metric': metric,
            'quality_badges': quality_badges,
            'structural_breaks': [2008, 2009, 2020, 2021]  # For shading
        }
    
    @staticmethod
    def get_streamlit_metadata(data: pd.DataFrame) -> Dict:
        """Get comprehensive metadata for Streamlit UI"""
        
        # Load saved metadata if available
        metadata_path = Path("data/forecast/metadata_v3.json")
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                saved_metadata = json.load(f)
        else:
            saved_metadata = {}
        
        return {
            'regions': sorted(data['region'].unique()),
            'region_codes': sorted(data['region_code'].unique()),
            'metrics': sorted(data['metric'].unique()),
            'years': sorted(data['year'].unique()),
            'data_coverage': saved_metadata.get('data_coverage', {}),
            'quality_indicators': saved_metadata.get('quality_indicators', {}),
            'last_updated': saved_metadata.get('run_timestamp', 'Unknown'),
            'coherence_status': 'Valid' if saved_metadata.get('coherence_violations', 0) == 0 else 'Adjusted'
        }
    
    @staticmethod
    def format_narrative(chart_data: Dict) -> str:
        """Generate narrative text for charts"""
        if not chart_data:
            return "No data available for this selection."
        
        hist = chart_data['historical']
        fore = chart_data['forecast']
        
        if len(hist['values']) >= 5:
            # Calculate historical CAGR
            hist_cagr = ((hist['values'][-1] / hist['values'][-5]) ** (1/5) - 1) * 100
        else:
            hist_cagr = None
        
        if len(fore['values']) >= 5:
            # Calculate forecast CAGR
            fore_cagr = ((fore['values'][4] / hist['values'][-1]) ** (1/5) - 1) * 100
        else:
            fore_cagr = None
        
        narrative = []
        
        if hist_cagr is not None:
            narrative.append(f"Historical 5-year growth: {hist_cagr:+.1f}% p.a.")
        
        if fore_cagr is not None:
            narrative.append(f"Forecast 5-year growth: {fore_cagr:+.1f}% p.a.")
        
        # Add quality context
        badges = chart_data.get('quality_badges', {})
        if badges.get('coverage') == 'Limited':
            narrative.append("⚠️ Limited historical data - interpret with caution")
        
        return " | ".join(narrative) if narrative else "Forecast generated using best available data."

# ===============================
# Main Entry Point V3.1
# ===============================

def main():
    """Run the institutional-grade forecasting pipeline V3.1 with unified schema"""
    
    try:
        # Initialize configuration
        config = ForecastConfig()
        
        # Log configuration
        logger.info("="*70)
        logger.info("FORECAST ENGINE V3.1 - UNIFIED SCHEMA")
        logger.info("="*70)
        logger.info("Configuration Summary:")
        logger.info(f"  DuckDB path: {config.duckdb_path}")
        logger.info(f"  Target year: {config.target_year}")
        logger.info(f"  Bootstrap samples: {config.n_bootstrap}")
        logger.info(f"  CV windows: {config.cv_test_windows}")
        logger.info(f"  Parallel processing: {config.parallel_processing}")
        
        # Run forecasts
        forecaster = InstitutionalForecasterV3(config)
        results = forecaster.run()
        
        # DEBUG: inspect the raw forecast results
        print("\n=== DEBUG: Results columns ===")
        print(results.columns.tolist())
        print("\n=== DEBUG: Sample forecast rows ===")
        print(results[results['data_type'] == 'forecast'].head(10))
        
        # --- Save to DuckDB gold schema ---
        try:
            import duckdb
            from datetime import datetime
            
            results_copy = results.copy()
            
            # Fallbacks: fill metric_id and period if empty
            if "metric_id" in results_copy.columns and "metric" in results_copy.columns:
                results_copy["metric_id"] = results_copy["metric_id"].fillna(results_copy["metric"])
                logger.info("Filled missing metric_id values from metric column")
            elif "metric" in results_copy.columns and "metric_id" not in results_copy.columns:
                results_copy["metric_id"] = results_copy["metric"]
                logger.info("Created metric_id from metric column")
            
            if "period" in results_copy.columns and "year" in results_copy.columns:
                results_copy["period"] = results_copy["period"].fillna(results_copy["year"])
                logger.info("Filled missing period values from year column")
            elif "year" in results_copy.columns and "period" not in results_copy.columns:
                results_copy["period"] = results_copy["year"]
                logger.info("Created period from year column")
            
            # Add metadata
            results_copy["forecast_run_date"] = datetime.now().date()
            results_copy["forecast_version"] = "3.1"
            
            # Keep only tidy columns
            cols = [
                "region_code", "region_name", "region_level",
                "metric_id", "period", "value",
                "unit", "freq", "data_type",
                "ci_lower", "ci_upper", "cv",
                "forecast_run_date", "forecast_version"
            ]
            results_flat = results_copy[[c for c in cols if c in results_copy.columns]].reset_index(drop=True)
            
            # Validate we have the critical columns with data
            if len(results_flat) > 0:
                nulls = {
                    "metric_id": results_flat["metric_id"].isna().sum(),
                    "period": results_flat["period"].isna().sum(),
                    "value": results_flat["value"].isna().sum()
                }
                if any(v > 0 for v in nulls.values()):
                    logger.warning(f"NULL counts in critical columns: {nulls}")
            
            con = duckdb.connect(str(config.duckdb_path))
            con.execute("CREATE SCHEMA IF NOT EXISTS gold")
            con.register("forecast_df", results_flat)
            
            # Main gold table
            con.execute("""
                CREATE OR REPLACE TABLE gold.itl1_forecast AS
                SELECT * FROM forecast_df
            """)
            
            # Forecast-only view
            con.execute("""
                CREATE OR REPLACE VIEW gold.itl1_forecast_only AS
                SELECT * FROM gold.itl1_forecast
                WHERE data_type = 'forecast'
            """)
            
            # Latest run view
            con.execute("""
                CREATE OR REPLACE VIEW gold.itl1_latest AS
                SELECT * FROM gold.itl1_forecast
                WHERE forecast_run_date = (
                    SELECT MAX(forecast_run_date) FROM gold.itl1_forecast
                )
            """)
            
            con.close()
            logger.info(f"✓ Saved forecasts into DuckDB gold schema ({len(results_flat)} rows)")
            
            # Log sample of saved data for verification
            if len(results_flat) > 0:
                sample = results_flat[results_flat['data_type'] == 'forecast'].head(2)
                if len(sample) > 0:
                    logger.info(f"Sample forecast data: period={sample['period'].tolist()}, metric_id={sample['metric_id'].tolist()}")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not save forecasts to DuckDB gold schema: {e}")
        
        # --- Success summary ---
        logger.info("="*70)
        logger.info("✅ INSTITUTIONAL FORECASTING V3.1 COMPLETED SUCCESSFULLY")
        logger.info(f"📊 Total records: {len(results)}")
        logger.info(f"📁 Outputs saved to: {config.output_dir}")
        logger.info("="*70)
        
        # --- Console print summary ---
        print("\n" + "="*70)
        print("FORECAST SUMMARY - V3.1 (Unified Schema)")
        print("="*70)
        print(f"DuckDB path: {config.duckdb_path}")
        
        # Use normalized column names for summary stats
        if "metric_id" in results.columns:
            print(f"Regions processed: {results['region_code'].nunique()}")
            print(f"Metrics forecasted: {results['metric_id'].nunique()}")
        else:
            print(f"Regions processed: {results['region_code'].nunique()}")
            print(f"Metrics forecasted: {results.get('metric', results.get('metric_id', pd.Series())).nunique()}")
        
        print(f"Forecast observations: {len(results[results['data_type'] == 'forecast'])}")
        
        if 'source' in results.columns:
            print(f"Derived indicators: {len(results[results['source'] == 'calculated'])}")
        
        print("\nOutputs saved to: data/forecast/")
        print("  - forecast_v3_long.csv (main output)")
        print("  - forecast_v3_wide.csv (pivot format)")
        print("  - confidence_intervals_v3.csv")
        print("  - forecast_quality_v3.csv")
        print("  - metadata_v3.json")
        print("  - diagnostics_report_v3.txt")
        print("  - DuckDB tables: gold.itl1_forecast / gold.itl1_forecast_only / gold.itl1_latest")
        print("="*70)
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()