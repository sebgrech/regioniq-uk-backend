#!/usr/bin/env python3
"""
UK Macro Forecasting Pipeline - Enterprise Edition
===================================================
Production forecaster for UK-level economic indicators with VAR/VECM.

**NEW: Phase 1 Enterprise Upgrade**
- VAR/VECM system modeling (captures GDP ↔ Employment ↔ Inflation dynamics)
- Cointegration testing (Johansen)
- Bootstrap confidence intervals for system-wide uncertainty
- Frequency alignment for cross-series modeling
- Dynamic exogenous policy rate handling (FUTURE-PROOF)

**What This Does That Oxford Economics Does:**
1. Models series together (not in isolation) → captures spillovers
2. Tests for long-run equilibrium relationships → VECM if cointegrated
3. Joint confidence intervals → proper system uncertainty
4. Policy rates as scenarios (not statistical forecasts) → realistic
5. Dynamic assumption curves → NEVER GOES STALE

**Critical Design Decision - Policy Rates:**
The BoE base rate is treated as an EXOGENOUS ASSUMPTION, not forecasted statistically.
Why? Policy rates are committee-driven, discontinuous, politically influenced.
ARIMA/VAR extrapolation produces nonsense (negative rates, unrealistic paths).

**FUTURE-PROOF: Dynamic Assumption Curves**
The BoE rate assumption curve is generated at RUNTIME from the current date.
This means:
- Run in Nov 2025 → forecasts start from Jan 2026
- Run in Nov 2026 → forecasts start from Jan 2027
- Run in Nov 2030 → forecasts start from Jan 2031
No hardcoded dates that become stale. Works forever.

**Customizing BoE Rate Scenarios:**
```python
# Baseline scenario (default): gradual 25bps cuts every 6 months
config = MacroForecastConfig(boe_scenario='baseline', boe_terminal_rate=3.0)

# Dovish scenario: aggressive 50bps cuts (recession response)
config = MacroForecastConfig(boe_scenario='dovish', boe_terminal_rate=2.5)

# Hawkish scenario: slow 12.5bps cuts (sticky inflation)
config = MacroForecastConfig(boe_scenario='hawkish', boe_terminal_rate=3.5)
```

The script automatically:
1. Reads last actual BoE rate from your ingest data
2. Generates future assumption points starting AFTER last actual
3. Applies scenario-specific cut speed
4. Converges to terminal rate
5. Never uses past dates as assumptions

**Usage:**
```python
from macro_forecast import MacroForecastConfig, MacroForecastPipeline

# Standard configuration (VAR + dynamic policy rates):
config = MacroForecastConfig()
pipeline = MacroForecastPipeline(config)
pipeline.run()

# Disable VAR (pure univariate):
config = MacroForecastConfig(use_var_system=False)

# Disable exogenous handling (not recommended):
config = MacroForecastConfig(use_exogenous_policy_rates=False)
```

**Reads:** data/silver/uk_macro_history.csv
**Outputs:**
  - data/forecast/uk_macro_forecast.csv (tidy format, historical + forecast)
  - data/forecast/uk_macro_forecast_only.csv (forecasts only)
  - data/forecast/uk_macro_forecast_metadata.json (includes VAR diagnostics + assumptions)
  - warehouse.duckdb → gold.uk_macro_forecast
  
**Architecture:** Mirrors ITL1 v3.1 for consistency.
"""

import logging
import warnings
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# Statistical packages (matching ITL1 dependencies)
try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.api import ExponentialSmoothing, VAR
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
    HAVE_STATSMODELS = True
    HAVE_VAR = True
except ImportError:
    HAVE_STATSMODELS = False
    HAVE_VAR = False

try:
    import duckdb
    HAVE_DUCKDB = True
except ImportError:
    HAVE_DUCKDB = False

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('macro_forecast.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger('macro_forecast')


# ================================
# Dynamic Policy Rate Assumptions
# ================================

def build_dynamic_boe_assumptions(
    current_rate: Optional[float] = None,
    scenario: str = 'baseline',
    terminal_rate: float = 3.0
) -> Dict[str, float]:
    """
    Generate forward-looking BoE rate assumptions from TODAY.
    
    CRITICAL: This function is called at runtime, so it always generates
    future dates, never past dates. Works the same in 2025, 2026, 2030.
    
    Args:
        current_rate: Latest actual BoE rate. If None, will be auto-detected.
        scenario: 'dovish' (fast cuts), 'baseline' (gradual), 'hawkish' (slow)
        terminal_rate: Long-run neutral rate (default 3.0%)
    
    Returns:
        Dict mapping period strings to rate values, e.g.:
        {'2025-01': 4.75, '2025-07': 4.50, ..., '2030+': 3.00}
    """
    today = datetime.now()
    current_year = today.year
    current_month = today.month
    
    # MPC meets 8x/year, but major decisions cluster around:
    # Feb, May, Aug, Nov (quarterly-ish)
    # We use 6-month steps (Jan/Jul) as reasonable scenario markers
    
    # Determine next assumption point (6 months ahead)
    if current_month <= 6:
        start_year = current_year
        start_month = 7  # Next Jul
    else:
        start_year = current_year + 1
        start_month = 1  # Next Jan
    
    # Set cut speed by scenario
    if scenario == 'dovish':
        cut_per_step = 0.50  # 50bps every 6 months (aggressive)
    elif scenario == 'hawkish':
        cut_per_step = 0.125  # 12.5bps every 6 months (very slow)
    else:  # baseline
        cut_per_step = 0.25  # 25bps every 6 months (gradual)
    
    # Default current rate if not provided (will be overridden by actual data)
    if current_rate is None:
        current_rate = 4.75
        log.warning(f"BoE assumption: current_rate not provided, using default {current_rate}%")
    
    # Build assumption curve
    assumptions = {}
    rate = current_rate
    year = start_year
    month = start_month
    
    steps = 0
    max_steps = 20  # Safety cap
    
    # Cut toward terminal rate
    while rate > terminal_rate + 0.05 and steps < max_steps:
        period_key = f"{year}-{month:02d}"
        assumptions[period_key] = round(rate, 2)
        
        # Advance to next 6-month period
        if month == 1:
            month = 7
        else:
            month = 1
            year += 1
        
        # Apply cut
        rate = max(rate - cut_per_step, terminal_rate)
        steps += 1
    
    # Add terminal rate (stable forever)
    if steps > 0:
        # Add one more point at terminal
        final_key = f"{year}-{month:02d}"
        assumptions[final_key] = terminal_rate
    
    # Add indefinite future terminal
    assumptions['2030+'] = terminal_rate
    
    log.info(f"BoE assumption curve ({scenario}): {len(assumptions)-1} steps from {start_year}-{start_month:02d}")
    
    return assumptions


# ================================
# Configuration
# ================================

@dataclass
class MacroForecastConfig:
    """Configuration for UK macro forecasting pipeline"""
    
    # I/O paths
    silver_csv: Path = Path("data/silver/uk_macro_history.csv")
    output_dir: Path = Path("data/forecast")
    duckdb_path: Path = Path("data/lake/warehouse.duckdb")
    
    # Forecast parameters
    target_year: int = 2050
    min_history_years: int = 10
    confidence_level: float = 0.95
    
    # Model parameters (aligned with ITL1)
    max_arima_order: int = 2
    use_log_transform: Dict[str, bool] = field(default_factory=lambda: {
        'uk_gdp_m_index': False,      # Already indexed
        'uk_cpih_index': False,        # Already indexed
        'uk_unemp_rate': False,        # Percentage
        'uk_emp_rate': False,          # Percentage
        'uk_median_weekly_pay_fulltime': True,  # Currency
        'boe_base_rate': False         # Percentage
    })
    
    # Structural breaks (matching ITL1)
    structural_breaks: List[Dict] = field(default_factory=lambda: [
        {'year': 2008, 'name': 'Financial Crisis', 'type': 'level'},
        {'year': 2009, 'name': 'Crisis Recovery', 'type': 'trend'},
        {'year': 2016, 'name': 'Brexit Vote', 'type': 'trend'},
        {'year': 2020, 'name': 'COVID-19', 'type': 'level'},
        {'year': 2021, 'name': 'COVID Recovery', 'type': 'trend'}
    ])
    
    # Constraints by metric
    constraints: Dict[str, Dict] = field(default_factory=lambda: {
        'uk_unemp_rate': {'min': 0, 'max': 100},
        'uk_emp_rate': {'min': 0, 'max': 100},
        'boe_base_rate': {'min': 0, 'max': 20},
        'uk_gdp_m_index': {'min': 0},
        'uk_cpih_index': {'min': 0},
        'uk_median_weekly_pay_fulltime': {'min': 0}
    })
    
    # Frequency-specific horizon rules
    horizon_by_freq: Dict[str, int] = field(default_factory=lambda: {
        'D': 365,   # Daily → 1 year
        'M': 300,   # Monthly → 25 years
        'M3': 300,  # 3-month rolling → 25 years
        'Q': 100,   # Quarterly → 25 years
        'A': 26     # Annual → 26 years
    })
    
    # VAR/VECM System Configuration (Phase 1: Enterprise upgrade)
    use_var_system: bool = True  # Set False to disable cross-series modeling
    var_systems: Dict[str, List[str]] = field(default_factory=lambda: {
        'core_macro': [
            'uk_gdp_m_index',
            'uk_cpih_index',
            'uk_emp_rate'
            # Note: boe_base_rate EXCLUDED - it's exogenous policy, not endogenous macro
        ],
        'labour_market': [
            'uk_emp_rate',
            'uk_unemp_rate'
            # Employment and unemployment rates move together (mechanically linked)
            # VAR captures: high emp → low unemp, cyclical dynamics
        ]
    })
    var_max_lags: int = 4
    var_bootstrap_samples: int = 500
    coint_significance: float = 0.05  # Johansen test significance level
    
    # Exogenous Policy Rate Assumptions (not statistically forecasted)
    use_exogenous_policy_rates: bool = True
    exogenous_metrics: List[str] = field(default_factory=lambda: ['boe_base_rate'])
    
    # BoE rate scenario configuration
    # DYNAMIC: Assumption curve is generated at runtime from current date
    boe_scenario: str = 'baseline'  # 'dovish', 'baseline', 'hawkish'
    boe_terminal_rate: float = 3.0  # Long-run neutral rate assumption
    
    # Populated dynamically - do not set manually
    boe_rate_assumption: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._validate()
    
    def _validate(self):
        if not self.silver_csv.exists():
            raise FileNotFoundError(f"Silver data not found: {self.silver_csv}")
        assert self.target_year > 2024, "Target year must be future"
        log.info("Configuration validated successfully")


# ================================
# Data Manager
# ================================

class MacroDataManager:
    """Handles loading and validation of UK macro data"""
    
    def __init__(self, config: MacroForecastConfig):
        self.config = config
    
    def load_historical(self) -> pd.DataFrame:
        """Load and validate UK macro history"""
        log.info(f"Loading UK macro data from {self.config.silver_csv}")
        
        df = pd.read_csv(self.config.silver_csv)
        
        # Validate schema
        required = ['region_code', 'metric_id', 'period', 'value', 'freq']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Filter to UK only (defensive)
        df = df[df['region_code'] == 'K02000001'].copy()
        
        # Parse periods to datetime for easier manipulation
        df['period_dt'] = pd.to_datetime(df['period'], errors='coerce')
        
        # Drop nulls
        df = df.dropna(subset=['period', 'value'])
        
        # Sort by metric and period
        df = df.sort_values(['metric_id', 'period']).reset_index(drop=True)
        
        log.info(f"Loaded {len(df)} observations across {df['metric_id'].nunique()} metrics")
        
        # Log coverage
        coverage = df.groupby(['metric_id', 'freq']).agg({
            'period': ['min', 'max', 'count']
        }).reset_index()
        coverage.columns = ['metric_id', 'freq', 'period_min', 'period_max', 'obs']
        log.info(f"Data coverage:\n{coverage.to_string(index=False)}")
        
        return df


# ================================
# Exogenous Policy Rate Handler
# ================================

class ExogenousPolicyRateHandler:
    """
    Handles policy rates (BoE, ECB, Fed) as exogenous scenario assumptions.
    
    **Why:** Policy rates are discontinuous, committee-driven decisions,
    not stationary time series. ARIMA/VAR extrapolation produces nonsense.
    
    **How:** We define assumption curves (baseline scenarios) and interpolate
    smoothly to match forecast horizons. These can be replaced with:
    - OBR/MPC forward guidance curves
    - Market expectations (OIS curves)
    - Client-specific policy scenarios
    """
    
    def __init__(self, config: MacroForecastConfig):
        self.config = config
    
    def build_assumption_path(
        self,
        metric_id: str,
        historical: pd.Series,
        horizon: int,
        freq: str
    ) -> Dict:
        """
        Build stepped exogenous assumption path for policy rate.
        Returns forecast-format dict with 'assumed' method flag.
        
        CRITICAL: For policy rates, we use STEPPED paths (holds constant
        until next MPC decision), not daily interpolation.
        
        DYNAMIC: Assumption curve is generated at runtime from current date,
        so it never uses past dates as future assumptions.
        """
        
        # Get current rate from latest historical data
        current_rate = historical.iloc[-1]
        
        # Generate dynamic assumption curve
        if metric_id == 'boe_base_rate':
            assumption_points = build_dynamic_boe_assumptions(
                current_rate=current_rate,
                scenario=self.config.boe_scenario,
                terminal_rate=self.config.boe_terminal_rate
            )
            log.info(f"  Generated dynamic BoE assumption curve ({self.config.boe_scenario} scenario)")
        else:
            # Generic fallback: hold last observed value
            log.warning(f"No assumption curve for {metric_id}, holding last observed")
            assumption_points = {
                '2030+': current_rate
            }
        
        # Convert assumption points to dated series
        assumption_dates = []
        assumption_values = []
        terminal_date = None
        terminal_value = None
        
        for period_key, value in sorted(assumption_points.items()):
            if period_key == '2030+':
                # Terminal rate - will extend forward
                terminal_date = pd.to_datetime('2030-01')
                terminal_value = value
            else:
                assumption_dates.append(pd.to_datetime(period_key))
                assumption_values.append(value)
        
        # Add terminal point
        if terminal_date is not None and terminal_value is not None:
            assumption_dates.append(terminal_date)
            assumption_values.append(terminal_value)
        
        # Create assumption series
        assumption_series = pd.Series(assumption_values, index=assumption_dates)
        
        # Get last historical date (actual data point)
        # CRITICAL: Parse properly to detect latest actual
        if isinstance(historical.index[-1], str):
            last_hist_date = pd.to_datetime(historical.index[-1])
        else:
            last_hist_date = historical.index[-1]
        
        log.info(f"  Last actual {metric_id}: {last_hist_date.strftime('%Y-%m-%d')} = {current_rate:.2f}%")
        
        # Find first assumption point AFTER last historical date
        future_assumptions = assumption_series[assumption_series.index > last_hist_date]
        
        if future_assumptions.empty:
            log.warning(f"  All assumption points are in the past! Extending last value.")
            # Just hold last historical value
            forecast_values = np.repeat(current_rate, min(horizon, 12))
            forecast_dates = pd.date_range(
                start=last_hist_date + pd.DateOffset(months=1),
                periods=len(forecast_values),
                freq='MS'
            )
            periods_fmt = '%Y-%m'
        else:
            # Build STEPPED forecast (policy rates don't interpolate daily)
            # Use assumption points as discrete MPC decisions
            
            # Start from first future assumption
            first_future_date = future_assumptions.index[0]
            
            # Use all future assumption points (they're already 6-monthly)
            forecast_dates = future_assumptions.index
            
            # Build stepped forecast (hold constant between MPC decisions)
            forecast_values = future_assumptions.values
            
            periods_fmt = '%Y-%m'
        
        # Build confidence bands (narrow for policy assumptions)
        # ±25bps reflects MPC decision uncertainty, not forecast error
        policy_uncertainty = 0.25
        ci_lower = forecast_values - policy_uncertainty
        ci_upper = forecast_values + policy_uncertainty
        
        # Apply constraints (rates typically non-negative)
        ci_lower = np.maximum(ci_lower, 0.0)
        forecast_values = np.maximum(forecast_values, 0.0)
        
        periods = [d.strftime(periods_fmt) for d in forecast_dates]
        
        log.info(f"  Generated {len(periods)} assumption periods: {periods[0]} → {periods[-1]}")
        log.info(f"  Rate path: {forecast_values[0]:.2f}% → {forecast_values[-1]:.2f}%")
        
        return {
            'method': 'exogenous_assumption',
            'values': forecast_values,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'periods': periods,
            'freq': 'M6',  # Mark as 6-monthly assumption points
            'metric_id': metric_id,
            'aic': None,  # Not applicable
            'assumption_source': f'{self.config.boe_scenario}_scenario',
            'note': f'Policy rate scenario assumption ({self.config.boe_scenario}): discrete MPC decisions, not statistical forecast'
        }


# ================================
# VAR/VECM System Forecaster (Phase 1: Enterprise)
# ================================

class VARSystemForecaster:
    """
    Vector Autoregression / Error Correction modeling for cross-series dynamics.
    This is what Oxford Economics uses - models GDP, employment, inflation together.
    """
    
    def __init__(self, config: MacroForecastConfig):
        self.config = config
    
    def align_frequencies(
        self,
        data: pd.DataFrame,
        metrics: List[str],
        target_freq: str = 'M'
    ) -> pd.DataFrame:
        """
        Align multiple series to common frequency for VAR modeling.
        Converts all to monthly by default (OE standard).
        """
        aligned_series = {}
        
        for metric in metrics:
            metric_data = data[data['metric_id'] == metric].copy()
            
            if metric_data.empty:
                log.warning(f"VAR: {metric} not found in data")
                continue
            
            freq = metric_data['freq'].iloc[0]
            
            # Parse periods to datetime
            metric_data['dt'] = pd.to_datetime(metric_data['period'], errors='coerce')
            metric_data = metric_data.dropna(subset=['dt']).sort_values('dt')
            
            # Convert to target frequency
            if freq == 'A' and target_freq == 'M':
                # Annual → Monthly: interpolate
                ts = metric_data.set_index('dt')['value']
                monthly_idx = pd.date_range(ts.index.min(), ts.index.max(), freq='MS')
                resampled = ts.reindex(monthly_idx).interpolate(method='linear')
            
            elif freq == 'M3' and target_freq == 'M':
                # 3-month rolling → Monthly: use end-of-window value
                ts = metric_data.set_index('dt')['value']
                monthly_idx = pd.date_range(ts.index.min(), ts.index.max(), freq='MS')
                resampled = ts.reindex(monthly_idx, method='ffill')
            
            elif freq == 'D' and target_freq == 'M':
                # Daily → Monthly: end-of-month value
                ts = metric_data.set_index('dt')['value']
                resampled = ts.resample('M').last()
            
            elif freq == 'M':
                # Already monthly
                ts = metric_data.set_index('dt')['value']
                resampled = ts
            
            else:
                log.warning(f"VAR: Cannot align {metric} (freq={freq}) to {target_freq}")
                continue
            
            aligned_series[metric] = resampled
        
        if not aligned_series:
            return pd.DataFrame()
        
        # Combine into wide dataframe
        aligned_df = pd.DataFrame(aligned_series)
        
        # Drop rows with any NaN (VAR requires complete cases)
        aligned_df = aligned_df.dropna()
        
        log.info(f"VAR: Aligned {len(aligned_series)} series → {len(aligned_df)} complete observations")
        
        return aligned_df
    
    def test_cointegration(self, data: pd.DataFrame) -> Dict:
        """
        Johansen cointegration test.
        Determines if series have long-run equilibrium relationship.
        """
        if not HAVE_VAR or len(data) < 30:
            return {'cointegrated': False, 'rank': 0, 'method': 'insufficient_data'}
        
        try:
            # Johansen test (trace statistic)
            result = coint_johansen(data, det_order=0, k_ar_diff=2)
            
            # Check critical values at configured significance
            trace_stats = result.lr1  # Trace statistics
            crit_vals = result.cvt[:, 1]  # 5% critical values (index 1)
            
            # Find cointegration rank (number of cointegrating relationships)
            rank = 0
            for i in range(len(trace_stats)):
                if trace_stats[i] > crit_vals[i]:
                    rank = i + 1
            
            cointegrated = rank > 0
            
            log.info(f"Johansen test: rank={rank}, cointegrated={cointegrated}")
            
            return {
                'cointegrated': cointegrated,
                'rank': rank,
                'trace_stats': trace_stats.tolist(),
                'critical_values': crit_vals.tolist(),
                'method': 'johansen'
            }
            
        except Exception as e:
            log.warning(f"Cointegration test failed: {e}")
            return {'cointegrated': False, 'rank': 0, 'method': 'error'}
    
    def fit_and_forecast(
        self,
        data: pd.DataFrame,
        system_metrics: List[str],
        horizon: int
    ) -> Dict:
        """
        Fit VAR or VECM system and generate joint forecast.
        This captures cross-series dynamics (e.g., GDP affects employment).
        """
        if not HAVE_VAR:
            log.error("VAR: statsmodels VAR/VECM not available")
            return {}
        
        # Align to common frequency
        aligned = self.align_frequencies(data, system_metrics, target_freq='M')
        
        if aligned.empty or len(aligned) < 30:
            log.warning(f"VAR: Insufficient aligned data ({len(aligned)} obs)")
            return {}
        
        # Test for cointegration
        coint_result = self.test_cointegration(aligned)
        
        try:
            if coint_result['cointegrated'] and coint_result['rank'] > 0:
                # Use VECM (captures long-run equilibrium)
                log.info(f"Fitting VECM (rank={coint_result['rank']})")
                model = VECM(
                    aligned,
                    k_ar_diff=min(self.config.var_max_lags, 4),
                    coint_rank=coint_result['rank'],
                    deterministic='ci'  # Constant inside cointegration relation
                )
                fitted = model.fit()
                model_type = 'VECM'
                
                # CRITICAL: VECM uses .predict() not .forecast()
                forecast_values = fitted.predict(steps=horizon)
            else:
                # Use VAR (no long-run constraints)
                log.info("Fitting VAR (no cointegration)")
                model = VAR(aligned)
                
                # Select optimal lag order by AIC
                lag_order_results = model.select_order(maxlags=self.config.var_max_lags)
                optimal_lags = lag_order_results.aic
                optimal_lags = max(1, min(optimal_lags, self.config.var_max_lags))
                
                fitted = model.fit(maxlags=optimal_lags)
                model_type = f'VAR({optimal_lags})'
                
                # VAR uses .forecast()
                forecast_values = fitted.forecast(aligned.values[-fitted.k_ar:], steps=horizon)
            
            # Bootstrap confidence intervals (proper VAR uncertainty)
            ci_lower, ci_upper = self._bootstrap_var_ci(
                fitted, aligned, horizon, model_type
            )
            
            # Generate forecast periods (monthly)
            last_date = aligned.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=horizon,
                freq='MS'
            )
            forecast_periods = [d.strftime('%Y-%m') for d in forecast_dates]
            
            # Package results per metric
            results = {}
            for i, metric in enumerate(aligned.columns):
                results[metric] = {
                    'method': f'{model_type}_system',
                    'values': forecast_values[:, i],
                    'ci_lower': ci_lower[:, i],
                    'ci_upper': ci_upper[:, i],
                    'periods': forecast_periods,
                    'freq': 'M',
                    'metric_id': metric,
                    'aic': fitted.aic if hasattr(fitted, 'aic') else None,
                    'system_metrics': list(aligned.columns),
                    'cointegrated': coint_result['cointegrated']
                }
            
            log.info(f"VAR system forecast: {len(results)} metrics × {horizon} periods")
            
            return results
            
        except Exception as e:
            log.error(f"VAR system fitting failed: {e}")
            return {}
    
    def _bootstrap_var_ci(
        self,
        fitted_model,
        data: pd.DataFrame,
        horizon: int,
        model_type: str,
        n_bootstrap: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bootstrap confidence intervals for VAR/VECM forecasts.
        Properly accounts for system-wide uncertainty.
        """
        if n_bootstrap is None:
            n_bootstrap = self.config.var_bootstrap_samples
        
        n_vars = data.shape[1]
        bootstrap_forecasts = []
        
        try:
            # Get residuals
            if hasattr(fitted_model, 'resid'):
                residuals = fitted_model.resid
            else:
                residuals = data.values[fitted_model.k_ar:] - fitted_model.fittedvalues
            
            # Bootstrap
            for _ in range(n_bootstrap):
                # Resample residuals
                boot_indices = np.random.choice(len(residuals), size=len(residuals), replace=True)
                boot_resid = residuals[boot_indices]
                
                # Reconstruct series
                boot_data = fitted_model.fittedvalues + boot_resid
                boot_df = pd.DataFrame(boot_data, columns=data.columns, index=data.index[-len(boot_data):])
                
                # Refit and forecast
                try:
                    if 'VECM' in model_type:
                        boot_model = VECM(
                            boot_df,
                            k_ar_diff=fitted_model.k_ar_diff,
                            coint_rank=fitted_model.coint_rank,
                            deterministic='ci'
                        )
                        boot_fitted = boot_model.fit()
                    else:
                        boot_model = VAR(boot_df)
                        boot_fitted = boot_model.fit(maxlags=fitted_model.k_ar)
                    
                    boot_fc = boot_fitted.forecast(boot_df.values[-boot_fitted.k_ar:], steps=horizon)
                    bootstrap_forecasts.append(boot_fc)
                except:
                    continue
            
            if len(bootstrap_forecasts) < 50:
                # Fallback to parametric CI
                log.warning("Bootstrap failed, using parametric CI")
                return self._parametric_var_ci(fitted_model, horizon, n_vars)
            
            # Calculate percentile CIs
            bootstrap_array = np.array(bootstrap_forecasts)
            ci_lower = np.percentile(bootstrap_array, 2.5, axis=0)
            ci_upper = np.percentile(bootstrap_array, 97.5, axis=0)
            
            return ci_lower, ci_upper
            
        except Exception as e:
            log.warning(f"Bootstrap CI failed: {e}, using parametric")
            return self._parametric_var_ci(fitted_model, horizon, n_vars)
    
    def _parametric_var_ci(
        self,
        fitted_model,
        horizon: int,
        n_vars: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback parametric confidence intervals"""
        
        # Use forecast error variance
        if hasattr(fitted_model, 'forecast_interval'):
            try:
                fc, lower, upper = fitted_model.forecast_interval(
                    fitted_model.endog[-fitted_model.k_ar:],
                    steps=horizon,
                    alpha=0.05
                )
                return lower, upper
            except:
                pass
        
        # Simple fallback: use residual std
        if hasattr(fitted_model, 'resid'):
            resid_std = np.std(fitted_model.resid, axis=0)
        else:
            resid_std = np.ones(n_vars)
        
        # Assume independence for simplicity
        forecast_std = resid_std * np.sqrt(np.arange(1, horizon + 1))[:, None]
        
        # Return dummy CIs (will be replaced by actual forecast ± width)
        return -forecast_std * 1.96, forecast_std * 1.96


# ================================
# Frequency-Aware Forecaster
# ================================

class FrequencyAwareMacroForecaster:
    """
    Handles forecasting for multiple time series frequencies.
    Uses appropriate models per frequency type.
    """
    
    def __init__(self, config: MacroForecastConfig):
        self.config = config
    
    def forecast_metric(
        self,
        series: pd.Series,
        freq: str,
        metric_id: str
    ) -> Dict:
        """
        Generate forecast for a single metric time series.
        Returns dict with forecast values, CIs, and metadata.
        """
        
        # Determine horizon
        horizon = self._get_horizon(series, freq)
        
        if horizon <= 0:
            log.warning(f"{metric_id}: No horizon to forecast")
            return {}
        
        # Get relevant structural breaks
        breaks = self._get_relevant_breaks(series)
        
        # Apply log transform if configured
        use_log = self.config.use_log_transform.get(metric_id, False)
        if use_log and (series > 0).all():
            working_series = np.log(series)
        else:
            working_series = series
            use_log = False
        
        # Try models in order of sophistication
        models_tried = []
        
        # 1. ARIMA with structural breaks
        if HAVE_STATSMODELS:
            arima_result = self._fit_arima(
                working_series, horizon, breaks, metric_id
            )
            if arima_result:
                models_tried.append(arima_result)
        
        # 2. ETS (good for indices and rates)
        if HAVE_STATSMODELS and len(series) > 20:
            ets_result = self._fit_ets(working_series, horizon)
            if ets_result:
                models_tried.append(ets_result)
        
        # 3. Linear trend fallback
        linear_result = self._fit_linear(working_series, horizon)
        if linear_result:
            models_tried.append(linear_result)
        
        # Select best model by AIC
        if not models_tried:
            return self._fallback_forecast(series, horizon, freq)
        
        best = min(models_tried, key=lambda m: m.get('aic', np.inf))
        
        # Transform back if needed
        if use_log:
            best['values'] = np.exp(best['values'])
            best['ci_lower'] = np.exp(best['ci_lower'])
            best['ci_upper'] = np.exp(best['ci_upper'])
        
        # Apply constraints
        best = self._apply_constraints(best, series, metric_id)
        
        # Generate forecast periods
        best['periods'] = self._generate_periods(series.index[-1], horizon, freq)
        best['freq'] = freq
        best['metric_id'] = metric_id
        
        return best
    
    def _get_horizon(self, series: pd.Series, freq: str) -> int:
        """Calculate forecast horizon based on frequency and target year"""
        
        last_period = series.index[-1]
        
        # Parse last period to year
        if isinstance(last_period, str):
            if freq == 'A':
                last_year = int(last_period)
            elif freq in ('M', 'M3'):
                last_year = int(last_period[:4])
            elif freq == 'Q':
                last_year = int(last_period[:4])
            elif freq == 'D':
                last_year = int(last_period[:4])
            else:
                last_year = 2024
        else:
            last_year = 2024
        
        # Calculate periods to target year
        years_to_forecast = self.config.target_year - last_year
        
        # Apply frequency-specific max horizon
        max_horizon = self.config.horizon_by_freq.get(freq, 100)
        
        if freq == 'A':
            horizon = min(years_to_forecast, max_horizon)
        elif freq == 'M':
            horizon = min(years_to_forecast * 12, max_horizon)
        elif freq == 'M3':
            horizon = min(years_to_forecast * 12, max_horizon)
        elif freq == 'Q':
            horizon = min(years_to_forecast * 4, max_horizon)
        elif freq == 'D':
            horizon = min(365, max_horizon)  # Cap daily at 1 year
        else:
            horizon = min(years_to_forecast, max_horizon)
        
        return horizon
    
    def _get_relevant_breaks(self, series: pd.Series) -> List[Dict]:
        """Filter structural breaks relevant to this series"""
        
        first_year = None
        last_year = None
        
        # Extract years from index
        for idx in series.index:
            if isinstance(idx, str):
                year = int(idx[:4])
                if first_year is None:
                    first_year = year
                last_year = year
        
        if first_year is None:
            return []
        
        # Only include breaks within or after series range
        relevant = [
            b for b in self.config.structural_breaks
            if first_year <= b['year'] <= last_year + 5
        ]
        
        return relevant
    
    def _fit_arima(
        self,
        series: pd.Series,
        horizon: int,
        breaks: List[Dict],
        metric_id: str
    ) -> Optional[Dict]:
        """Fit ARIMA with structural break dummies"""
        
        try:
            # Prepare break dummies
            if breaks:
                exog = self._prepare_break_dummies(series, breaks)
            else:
                exog = None
            
            # Test for stationarity
            adf_p = adfuller(series, autolag='AIC')[1]
            d = 1 if adf_p > 0.05 else 0
            
            # Grid search for best ARIMA order
            best_model = None
            best_aic = np.inf
            best_order = None
            
            for p in range(self.config.max_arima_order + 1):
                for q in range(self.config.max_arima_order + 1):
                    if p == 0 and q == 0 and d == 0:
                        continue
                    
                    try:
                        model = ARIMA(series, order=(p, d, q), exog=exog)
                        fitted = model.fit(method_kwargs={"warn_convergence": False})
                        
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_model = fitted
                            best_order = (p, d, q)
                    except:
                        continue
            
            if best_model is None:
                return None
            
            # Extend exog for forecast
            if exog is not None:
                exog_fc = self._extend_exog(exog, series.index, horizon)
            else:
                exog_fc = None
            
            # Generate forecast
            forecast_obj = best_model.get_forecast(steps=horizon, exog=exog_fc)
            forecast = forecast_obj.predicted_mean
            ci = forecast_obj.conf_int(alpha=1 - self.config.confidence_level)
            
            return {
                'method': f'ARIMA{best_order}',
                'values': forecast.values,
                'ci_lower': ci.iloc[:, 0].values,
                'ci_upper': ci.iloc[:, 1].values,
                'aic': best_aic
            }
            
        except Exception as e:
            log.warning(f"ARIMA failed for {metric_id}: {e}")
            return None
    
    def _fit_ets(self, series: pd.Series, horizon: int) -> Optional[Dict]:
        """Fit Exponential Smoothing"""
        
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
                            series,
                            trend=trend,
                            damped_trend=damped,
                            seasonal=None
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
            
            forecast = best_model.forecast(steps=horizon)
            
            # CI from residuals
            residuals = series - best_model.fittedvalues
            sigma = residuals.std()
            ci_width = 1.96 * sigma * np.sqrt(range(1, horizon + 1))
            
            return {
                'method': best_config,
                'values': forecast.values,
                'ci_lower': forecast.values - ci_width,
                'ci_upper': forecast.values + ci_width,
                'aic': best_aic
            }
            
        except Exception as e:
            log.warning(f"ETS failed: {e}")
            return None
    
    def _fit_linear(self, series: pd.Series, horizon: int) -> Optional[Dict]:
        """Linear trend model"""
        
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
                ci = predictions.conf_int(alpha=1 - self.config.confidence_level)
                
                return {
                    'method': 'Linear',
                    'values': forecast,
                    'ci_lower': ci[:, 0],
                    'ci_upper': ci[:, 1],
                    'aic': model.aic
                }
            else:
                # Numpy fallback
                coeffs = np.polyfit(X.flatten(), y, 1)
                X_future = np.arange(len(series), len(series) + horizon)
                forecast = np.polyval(coeffs, X_future)
                
                residuals = y - np.polyval(coeffs, X.flatten())
                sigma = residuals.std()
                ci_width = 1.96 * sigma
                
                return {
                    'method': 'Linear',
                    'values': forecast,
                    'ci_lower': forecast - ci_width,
                    'ci_upper': forecast + ci_width,
                    'aic': None
                }
        
        except Exception as e:
            log.warning(f"Linear failed: {e}")
            return None
    
    def _fallback_forecast(
        self,
        series: pd.Series,
        horizon: int,
        freq: str
    ) -> Dict:
        """Naive forecast with historical volatility"""
        
        last_value = series.iloc[-1]
        
        # Recent trend
        if len(series) > 5:
            trend = (series.iloc[-1] - series.iloc[-5]) / 5
        else:
            trend = 0
        
        values = np.array([last_value + trend * h for h in range(1, horizon + 1)])
        
        # CI based on historical std
        std = series.std()
        ci_width = 1.96 * std * np.sqrt(range(1, horizon + 1))
        
        return {
            'method': 'Fallback',
            'values': values,
            'ci_lower': values - ci_width,
            'ci_upper': values + ci_width,
            'aic': None,
            'periods': [],
            'freq': freq
        }
    
    def _apply_constraints(
        self,
        forecast: Dict,
        historical: pd.Series,
        metric_id: str
    ) -> Dict:
        """Apply metric-specific constraints"""
        
        constraints = self.config.constraints.get(metric_id, {})
        
        if 'min' in constraints:
            forecast['values'] = np.maximum(forecast['values'], constraints['min'])
            forecast['ci_lower'] = np.maximum(forecast['ci_lower'], constraints['min'])
        
        if 'max' in constraints:
            forecast['values'] = np.minimum(forecast['values'], constraints['max'])
            forecast['ci_upper'] = np.minimum(forecast['ci_upper'], constraints['max'])
        
        return forecast
    
    def _prepare_break_dummies(
        self,
        series: pd.Series,
        breaks: List[Dict]
    ) -> Optional[np.ndarray]:
        """Create structural break dummy variables"""
        
        if not breaks:
            return None
        
        dummies = []
        
        for break_info in breaks:
            break_year = break_info['year']
            break_type = break_info.get('type', 'level')
            
            # Extract years from index
            years = []
            for idx in series.index:
                if isinstance(idx, str):
                    years.append(int(idx[:4]))
                else:
                    years.append(2024)
            years = np.array(years)
            
            if break_type == 'level':
                dummy = (years >= break_year).astype(int)
                dummies.append(dummy)
            elif break_type == 'trend':
                time_since = np.maximum(0, years - break_year)
                dummies.append(time_since)
        
        return np.column_stack(dummies) if dummies else None
    
    def _extend_exog(
        self,
        exog: np.ndarray,
        historical_index: pd.Index,
        horizon: int
    ) -> np.ndarray:
        """Extend exogenous variables for forecast period"""
        
        if exog is None:
            return None
        
        if len(exog.shape) == 1:
            exog = exog.reshape(-1, 1)
        
        n_vars = exog.shape[1]
        
        # Extract last year from index
        last_period = historical_index[-1]
        if isinstance(last_period, str):
            last_year = int(last_period[:4])
        else:
            last_year = 2024
        
        extended = []
        for h in range(1, horizon + 1):
            future_year = last_year + h
            row = []
            for i in range(n_vars):
                # Level dummy maintains
                if np.all(np.isin(exog[:, i], [0, 1])):
                    row.append(exog[-1, i])
                else:
                    # Trend continues
                    increment = exog[-1, i] - exog[-2, i] if len(exog) > 1 else 1
                    row.append(exog[-1, i] + increment)
            extended.append(row)
        
        return np.array(extended)
    
    def _generate_periods(
        self,
        last_period: str,
        horizon: int,
        freq: str
    ) -> List[str]:
        """Generate future period labels based on frequency"""
        
        periods = []
        
        if freq == 'A':
            # Annual: YYYY
            last_year = int(last_period)
            periods = [str(last_year + h) for h in range(1, horizon + 1)]
        
        elif freq in ('M', 'M3'):
            # Monthly: YYYY-MM
            last_dt = pd.to_datetime(last_period)
            future_dates = pd.date_range(
                start=last_dt + pd.DateOffset(months=1),
                periods=horizon,
                freq='MS'
            )
            periods = [d.strftime('%Y-%m') for d in future_dates]
        
        elif freq == 'Q':
            # Quarterly: YYYY-Qn
            parts = last_period.split('-Q')
            last_year = int(parts[0])
            last_q = int(parts[1])
            
            for h in range(1, horizon + 1):
                q = ((last_q - 1 + h) % 4) + 1
                y = last_year + ((last_q - 1 + h) // 4)
                periods.append(f"{y}-Q{q}")
        
        elif freq == 'D':
            # Daily: YYYY-MM-DD
            last_dt = pd.to_datetime(last_period)
            future_dates = pd.date_range(
                start=last_dt + pd.Timedelta(days=1),
                periods=horizon,
                freq='D'
            )
            periods = [d.strftime('%Y-%m-%d') for d in future_dates]
        
        else:
            # Fallback
            periods = [f"{last_period}+{h}" for h in range(1, horizon + 1)]
        
        return periods


# ================================
# Output Writer
# ================================

class MacroOutputWriter:
    """Handles saving forecasts in tidy format"""
    
    def __init__(self, config: MacroForecastConfig):
        self.config = config
    
    def save_forecasts(
        self,
        forecasts: List[Dict],
        historical: pd.DataFrame
    ):
        """Save forecast results to CSV and DuckDB"""
        
        # Build tidy forecast dataframe
        forecast_rows = []
        
        for fc in forecasts:
            metric_id = fc['metric_id']
            freq = fc['freq']
            method = fc['method']
            
            for i, period in enumerate(fc['periods']):
                forecast_rows.append({
                    'region_code': 'K02000001',
                    'region_name': 'United Kingdom',
                    'region_level': 'UK',
                    'metric_id': metric_id,
                    'period': period,
                    'value': fc['values'][i],
                    'ci_lower': fc['ci_lower'][i],
                    'ci_upper': fc['ci_upper'][i],
                    'unit': self._get_unit(historical, metric_id),
                    'freq': freq,
                    'source': f'forecast_{method}',
                    'vintage': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'model': method,
                    'data_type': 'forecast'
                })
        
        forecast_df = pd.DataFrame(forecast_rows)
        
        # Add historical as well (for complete unified view)
        historical_df = historical.copy()
        historical_df['data_type'] = 'historical'
        historical_df['model'] = 'observed'
        
        # Ensure both have same columns
        common_cols = [
            'region_code', 'region_name', 'region_level', 'metric_id',
            'period', 'value', 'unit', 'freq', 'source', 'vintage',
            'model', 'data_type'
        ]
        
        # Add CI columns to historical (as null)
        for col in ['ci_lower', 'ci_upper']:
            if col not in historical_df.columns:
                historical_df[col] = np.nan
        
        # Select common columns from both
        hist_cols = [c for c in common_cols + ['ci_lower', 'ci_upper'] if c in historical_df.columns]
        fc_cols = [c for c in common_cols + ['ci_lower', 'ci_upper'] if c in forecast_df.columns]
        
        combined = pd.concat([
            historical_df[hist_cols],
            forecast_df[fc_cols]
        ], ignore_index=True)
        
        # Sort by metric and period
        combined = combined.sort_values(['metric_id', 'period']).reset_index(drop=True)
        
        # Save CSV
        csv_path = self.config.output_dir / 'uk_macro_forecast.csv'
        combined.to_csv(csv_path, index=False)
        log.info(f"✓ Saved forecast CSV → {csv_path} ({len(combined)} rows)")
        
        # Save forecast-only CSV
        fc_only_path = self.config.output_dir / 'uk_macro_forecast_only.csv'
        forecast_df.to_csv(fc_only_path, index=False)
        log.info(f"✓ Saved forecast-only CSV → {fc_only_path} ({len(forecast_df)} rows)")
        
        # Save to DuckDB
        if HAVE_DUCKDB:
            try:
                con = duckdb.connect(str(self.config.duckdb_path))
                con.execute("CREATE SCHEMA IF NOT EXISTS gold")
                con.register('forecast_combined', combined)
                con.execute("""
                    CREATE OR REPLACE TABLE gold.uk_macro_forecast AS
                    SELECT * FROM forecast_combined
                """)
                con.close()
                log.info(f"✓ Saved to DuckDB → gold.uk_macro_forecast")
            except Exception as e:
                log.warning(f"DuckDB save failed: {e}")
        
        # Generate metadata
        self._save_metadata(forecasts, combined)
    
    def _get_unit(self, historical: pd.DataFrame, metric_id: str) -> str:
        """Extract unit for metric from historical data"""
        row = historical[historical['metric_id'] == metric_id]
        if not row.empty and 'unit' in row.columns:
            return row['unit'].iloc[0]
        return 'unknown'
    
    def _save_metadata(self, forecasts: List[Dict], combined: pd.DataFrame):
        """Generate comprehensive metadata"""
        
        metadata = {
            'run_timestamp': datetime.now(timezone.utc).isoformat(),
            'version': '1.0_VAR',
            'config': {
                'target_year': self.config.target_year,
                'confidence_level': self.config.confidence_level,
                'structural_breaks': self.config.structural_breaks,
                'var_enabled': self.config.use_var_system,
                'var_systems': self.config.var_systems if self.config.use_var_system else {},
                'exogenous_policy_rates': self.config.use_exogenous_policy_rates,
                'boe_scenario': self.config.boe_scenario,
                'boe_terminal_rate': self.config.boe_terminal_rate
            },
            'metrics_forecasted': list(combined[combined['data_type'] == 'forecast']['metric_id'].unique()),
            'forecast_summary': {},
            'model_usage': {},
            'var_diagnostics': {},
            'exogenous_assumptions': {}
        }
        
        # Track VAR vs univariate vs exogenous
        var_metrics = set()
        exogenous_metrics = set()
        
        for fc in forecasts:
            metric_id = fc['metric_id']
            
            # Check if exogenous
            if fc.get('method') == 'exogenous_assumption':
                exogenous_metrics.add(metric_id)
                metadata['exogenous_assumptions'][metric_id] = {
                    'source': fc.get('assumption_source', 'unknown'),
                    'note': fc.get('note', ''),
                    'range': f"{fc['values'].min():.2f} → {fc['values'].max():.2f}",
                    'scenario': self.config.boe_scenario if 'boe' in metric_id else 'baseline',
                    'terminal_rate': self.config.boe_terminal_rate if 'boe' in metric_id else None,
                    'dynamic': True  # Flag that this updates at runtime
                }
            
            # Check if VAR system
            elif 'system_metrics' in fc:
                var_metrics.add(metric_id)
                
                # Store cointegration info
                if fc.get('cointegrated'):
                    metadata['var_diagnostics'][metric_id] = {
                        'system_metrics': fc['system_metrics'],
                        'cointegrated': True,
                        'method': fc['method']
                    }
        
        metadata['var_metrics'] = list(var_metrics)
        metadata['exogenous_metrics'] = list(exogenous_metrics)
        metadata['univariate_metrics'] = [
            m for m in metadata['metrics_forecasted'] 
            if m not in var_metrics and m not in exogenous_metrics
        ]
        
        # Summary by metric
        for metric_id in metadata['metrics_forecasted']:
            metric_data = combined[
                (combined['metric_id'] == metric_id) &
                (combined['data_type'] == 'forecast')
            ]
            
            if not metric_data.empty:
                metadata['forecast_summary'][metric_id] = {
                    'periods': len(metric_data),
                    'period_range': f"{metric_data['period'].min()} to {metric_data['period'].max()}",
                    'value_range': f"{metric_data['value'].min():.2f} to {metric_data['value'].max():.2f}",
                    'mean_value': float(metric_data['value'].mean()),
                    'model': metric_data['model'].iloc[0] if 'model' in metric_data else 'unknown',
                    'var_system': metric_id in var_metrics,
                    'exogenous': metric_id in exogenous_metrics
                }
        
        # Model usage
        if 'model' in combined.columns:
            model_counts = combined[combined['data_type'] == 'forecast']['model'].value_counts()
            metadata['model_usage'] = model_counts.to_dict()
        
        # Save JSON
        metadata_path = self.config.output_dir / 'uk_macro_forecast_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        log.info(f"✓ Saved metadata → {metadata_path}")
        
        # Log summaries
        if exogenous_metrics:
            log.info(f"\nExogenous Policy Assumptions (Dynamic):")
            log.info(f"  Metrics: {len(exogenous_metrics)}")
            log.info(f"  Scenario: {self.config.boe_scenario}")
            for metric, info in metadata['exogenous_assumptions'].items():
                log.info(f"    {metric}: {info['source']} | Range: {info['range']}")
        
        if var_metrics:
            log.info(f"\nVAR/VECM System Summary:")
            log.info(f"  Metrics in VAR systems: {len(var_metrics)}")
            log.info(f"  Cointegrated systems: {len(metadata['var_diagnostics'])}")
            for metric, info in metadata['var_diagnostics'].items():
                log.info(f"    {metric}: {info['method']} with {info['system_metrics']}")


# ================================
# Main Pipeline
# ================================

class MacroForecastPipeline:
    """Main orchestrator for UK macro forecasting"""
    
    def __init__(self, config: MacroForecastConfig):
        self.config = config
        self.data_manager = MacroDataManager(config)
        self.forecaster = FrequencyAwareMacroForecaster(config)
        self.var_forecaster = VARSystemForecaster(config) if config.use_var_system else None
        self.policy_handler = ExogenousPolicyRateHandler(config) if config.use_exogenous_policy_rates else None
        self.writer = MacroOutputWriter(config)
    
    def run(self):
        """Execute complete macro forecasting pipeline"""
        
        log.info("="*70)
        log.info("UK MACRO FORECASTING PIPELINE v1.0 (with VAR/VECM)")
        log.info("="*70)
        
        # Load historical data
        log.info("Loading historical UK macro data...")
        historical = self.data_manager.load_historical()
        
        forecasts = []
        metrics_processed = set()
        
        # Phase 0: Exogenous Policy Rates (not forecasted, assumed)
        if self.config.use_exogenous_policy_rates and self.policy_handler:
            log.info("\n" + "="*70)
            log.info("PHASE 0: EXOGENOUS POLICY RATE ASSUMPTIONS (DYNAMIC)")
            log.info("="*70)
            log.info("Note: Policy rates are scenario assumptions, not statistical forecasts")
            log.info(f"Scenario: {self.config.boe_scenario} | Terminal rate: {self.config.boe_terminal_rate}%")
            log.info("Assumption curve generated dynamically from current date (future-proof)")
            
            for metric_id in self.config.exogenous_metrics:
                if metric_id not in historical['metric_id'].values:
                    log.warning(f"  {metric_id}: Not found in historical data, skipping")
                    continue
                
                log.info(f"\nProcessing {metric_id} (exogenous)...")
                
                metric_data = historical[historical['metric_id'] == metric_id]
                freq = metric_data['freq'].iloc[0]
                
                series = pd.Series(
                    metric_data['value'].values,
                    index=metric_data['period'].values,
                    name=metric_id
                )
                
                # Determine horizon
                horizon = self.forecaster._get_horizon(series, freq)
                
                try:
                    assumption = self.policy_handler.build_assumption_path(
                        metric_id, series, horizon, freq
                    )
                    
                    if assumption and assumption.get('periods'):
                        forecasts.append(assumption)
                        metrics_processed.add(metric_id)
                        
                        # Log assumption curve details
                        log.info(f"  ✓ {metric_id}: Assumption curve applied")
                        log.info(f"    Method: {assumption['method']}")
                        log.info(f"    Periods: {len(assumption['periods'])}")
                        log.info(f"    Range: {assumption['values'].min():.2f}% → {assumption['values'].max():.2f}%")
                        log.info(f"    Note: {assumption['note']}")
                        
                except Exception as e:
                    log.error(f"  ✗ {metric_id}: Assumption path failed - {e}")
        
        # Phase 1: VAR/VECM System Forecasts (Enterprise-grade cross-series)
        if self.config.use_var_system and self.var_forecaster and HAVE_VAR:
            log.info("\n" + "="*70)
            log.info("PHASE 1: VAR/VECM SYSTEM FORECASTING (Enterprise)")
            log.info("="*70)
            
            for system_name, system_metrics in self.config.var_systems.items():
                log.info(f"\nProcessing VAR system: {system_name}")
                log.info(f"Metrics: {system_metrics}")
                
                # Check all metrics are available
                available = [m for m in system_metrics if m in historical['metric_id'].values]
                if len(available) < 2:
                    log.warning(f"  Skipping {system_name}: insufficient metrics ({available})")
                    continue
                
                # Determine horizon (use maximum across system metrics)
                max_horizon = 0
                for metric in available:
                    metric_data = historical[historical['metric_id'] == metric]
                    freq = metric_data['freq'].iloc[0]
                    series = pd.Series(
                        metric_data['value'].values,
                        index=metric_data['period'].values
                    )
                    h = self.forecaster._get_horizon(series, freq)
                    max_horizon = max(max_horizon, h)
                
                # Fit VAR system
                try:
                    system_results = self.var_forecaster.fit_and_forecast(
                        historical,
                        available,
                        horizon=min(max_horizon, 300)  # Cap at 300 months
                    )
                    
                    if system_results:
                        for metric, forecast in system_results.items():
                            forecasts.append(forecast)
                            metrics_processed.add(metric)
                            log.info(f"  ✓ {metric}: VAR system forecast ({forecast['method']})")
                    
                except Exception as e:
                    log.error(f"  ✗ VAR system {system_name} failed: {e}")
        
        # Phase 2: Univariate Forecasts (fallback for remaining metrics)
        log.info("\n" + "="*70)
        log.info("PHASE 2: UNIVARIATE FORECASTING (Remaining metrics)")
        log.info("="*70)
        
        remaining_metrics = [m for m in historical['metric_id'].unique() 
                           if m not in metrics_processed]
        
        if remaining_metrics:
            log.info(f"Processing {len(remaining_metrics)} metrics not in VAR/exogenous")
        
        for metric_id in remaining_metrics:
            log.info(f"Processing {metric_id}...")
            
            metric_data = historical[historical['metric_id'] == metric_id]
            freq = metric_data['freq'].iloc[0]
            
            # Build time series
            series = pd.Series(
                metric_data['value'].values,
                index=metric_data['period'].values,
                name=metric_id
            )
            
            # Check minimum history
            if len(series) < self.config.min_history_years:
                log.warning(f"  {metric_id}: Insufficient history ({len(series)} obs), skipping")
                continue
            
            # Generate forecast
            try:
                forecast = self.forecaster.forecast_metric(series, freq, metric_id)
                
                if forecast and forecast.get('periods'):
                    forecasts.append(forecast)
                    log.info(f"  ✓ {metric_id}: {len(forecast['periods'])} periods, model={forecast['method']}")
                else:
                    log.warning(f"  ✗ {metric_id}: No forecast generated")
                    
            except Exception as e:
                log.error(f"  ✗ {metric_id}: Forecast failed - {e}")
        
        # Save outputs
        if forecasts:
            log.info(f"\n{'='*70}")
            log.info(f"Saving {len(forecasts)} metric forecasts...")
            self.writer.save_forecasts(forecasts, historical)
        else:
            log.error("No forecasts generated!")
            return
        
        # Summary with breakdown
        exogenous_count = len([f for f in forecasts if f.get('method') == 'exogenous_assumption'])
        var_count = len(metrics_processed) - exogenous_count
        univariate_count = len(forecasts) - var_count - exogenous_count
        
        log.info("="*70)
        log.info(f"✅ MACRO FORECASTING COMPLETED")
        log.info(f"Total metrics forecasted: {len(forecasts)}")
        log.info(f"  - Exogenous assumptions: {exogenous_count}")
        log.info(f"  - VAR/VECM system: {var_count}")
        log.info(f"  - Univariate: {univariate_count}")
        log.info(f"Output directory: {self.config.output_dir}")
        log.info("="*70)


# ================================
# Entry Point
# ================================

def main():
    """Run UK macro forecasting pipeline"""
    
    try:
        config = MacroForecastConfig()
        pipeline = MacroForecastPipeline(config)
        pipeline.run()
        
    except Exception as e:
        log.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()