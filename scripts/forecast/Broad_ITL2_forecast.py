#!/usr/bin/env python3
"""
Region IQ - ITL2 Regional Forecasting Engine (Aligned with ITL1 V3.5)
=============================================================================

Architecture aligned with proven ITL1 V3.5 patterns:
- 7 base forecast metrics + 1 calculated (gdhi_per_head_gbp)
- 2 derived metrics (productivity, income_per_worker) → separate table

Metric Structure:
  Base (forecast directly):
    - nominal_gva_mn_gbp, gdhi_total_mn_gbp, emp_total_jobs
    - population_total, population_16_64
    - employment_rate_pct, unemployment_rate_pct
  Calculated (from reconciled totals, stored with base):
    - gdhi_per_head_gbp
  Derived (Monte Carlo, stored in gold.itl2_derived):
    - productivity_gbp_per_job, income_per_worker_gbp

Pipeline Flow:
  1. Load ITL2 history from silver layer (DuckDB/CSV)
  2. Forecast 7 base metrics (VAR/VECM + univariate ensemble)
  3. Apply growth caps and mean reversion to ITL1 parent
  4. Reconcile 5 additive metrics to ITL1 parent totals
  5. Calculate ALL derived metrics (productivity, income, gdhi_per_head)
  6. Split gdhi_per_head → merge into base_data
  7. Save: base → gold.itl2_forecast, derived → gold.itl2_derived

Outputs:
  - data/forecast/itl2_forecast_long.csv
  - data/forecast/itl2_forecast_wide.csv
  - data/forecast/itl2_derived.csv
  - data/forecast/itl2_confidence_intervals.csv
  - data/forecast/itl2_metadata.json
  - gold.itl2_forecast (base + gdhi_per_head)
  - gold.itl2_derived (productivity, income_per_worker)

Author: Region IQ
"""

import warnings
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List, Sequence

import numpy as np
np.random.seed(42)
import pandas as pd
import json

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("itl2_forecast.log", mode="w"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ITL2_V4_1_MERGED")

# -----------------------------------------------------------------------------
# Optional dependencies
# -----------------------------------------------------------------------------
try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
    from scipy import stats
    HAVE_STATSMODELS = True
except ImportError as e:
    logger.warning(f"Statistical packages not fully available: {e}")
    HAVE_STATSMODELS = False

try:
    import duckdb
    HAVE_DUCKDB = True
except ImportError:
    HAVE_DUCKDB = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ForecastConfigITL2:
    """V4.1 Configuration - NOMIS rates + Growth Dampening"""

    # Data paths
    silver_dir: Path = Path("data/silver")
    duckdb_path: Path = Path("data/lake/warehouse.duckdb")
    use_duckdb: bool = True

    # Output
    output_dir: Path = Path("data/forecast")
    cache_dir: Path = Path("data/cache")

    # Forecast parameters
    target_year: int = 2050
    min_history_years: int = 12
    confidence_levels: List[float] = field(default_factory=lambda: [0.80, 0.95])

    # Model parameters
    max_arima_order: int = 2
    max_var_lags: int = 3

    # VAR/VECM systems
    use_var_systems: bool = True
    var_systems: Dict[str, List[str]] = field(default_factory=lambda: {
        "gva_employment": ["nominal_gva_mn_gbp", "emp_total_jobs"]
    })
    var_bootstrap_samples: int = 300

    # Top-down reconciliation to ITL1 (ADDITIVE METRICS ONLY)
    use_macro_anchoring: bool = True
    macro_anchor_map: Dict[str, str] = field(default_factory=lambda: {
        # ITL2 metric → ITL1 metric_id (ADDITIVE ONLY)
        "nominal_gva_mn_gbp": "nominal_gva_mn_gbp",
        "gdhi_total_mn_gbp": "gdhi_total_mn_gbp",
        "emp_total_jobs": "emp_total_jobs",
        # NI-only jobs metric stays in NI bubble (no UK macro anchoring)
        "emp_total_jobs_ni": "emp_total_jobs_ni",
        "population_total": "population_total",
        "population_16_64": "population_16_64"
        # NOTE: Rate metrics are NOT additive and NOT reconciled
    })

    # Metric definitions (NOMIS rates as primary, not derived)
    metric_definitions: Dict[str, Dict] = None
    use_log_transform: Dict[str, bool] = None
    structural_breaks: List[Dict] = None

    # Cross-validation (reserved for future use)
    cv_min_train_size: int = 15
    cv_test_windows: int = 3
    cv_horizon: int = 2

    # Performance / caching (reserved for future use)
    n_bootstrap: int = 200
    cache_enabled: bool = True

    # Constraints (mechanical)
    enforce_non_negative: bool = True
    enforce_monotonic_population: bool = True
    growth_rate_cap_percentiles: Tuple[float, float] = (2, 98)

    # -------------------------------------------------------------------------
    # Growth dampening parameters (V4.0 Growth Dampening logic)
    # -------------------------------------------------------------------------

    # Hard caps on annual growth (aligned with ITL1 V3.5)
    max_annual_growth: Dict[str, float] = field(default_factory=lambda: {
        "nominal_gva_mn_gbp": 0.06,      # 6% max nominal
        "gdhi_total_mn_gbp": 0.06,       # 6% max nominal
        "emp_total_jobs": 0.025,         # 2.5% max
        "emp_total_jobs_ni": 0.025,      # same cap, NI-only
        "population_total": 0.015,       # 1.5% max
        "population_16_64": 0.012,       # 1.2% max (working-age grows slower)
        # Rates: bounded 0-100, no growth caps needed
    })

    # Historical CAGR buffer (cap at historical + this value)
    historical_cagr_buffer: float = 0.02  # 2% buffer above historical

    # Mean reversion parameters
    mean_reversion_start_year: int = 5    # Start blending after year 5
    mean_reversion_full_year: int = 15    # Fully converged by year 15
    mean_reversion_weight: float = 0.7    # At full convergence, 70% parent growth

    # Minimum growth (prevent collapse)
    min_annual_growth: Dict[str, float] = field(default_factory=lambda: {
        "nominal_gva_mn_gbp": -0.05,     # -5% min
        "gdhi_total_mn_gbp": -0.05,      # -5% min
        "emp_total_jobs": -0.03,         # -3% min
        "emp_total_jobs_ni": -0.03,      # same min, NI-only
        "population_total": -0.005,      # -0.5% min
        "population_16_64": -0.01        # -1% min (working-age can decline faster)
    })

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Metric definitions: additive + NOMIS rates as primary
        if self.metric_definitions is None:
            self.metric_definitions = {
                # Additive metrics (reconciled to ITL1)
                "population_total": {
                    "unit": "persons",
                    "transform": "log",
                    "monotonic": True,
                    "reconcile": True,
                    "description": "Total population"
                },
                "population_16_64": {
                    "unit": "persons",
                    "transform": "none",  # Not log - working-age can decline
                    "monotonic": False,
                    "reconcile": True,
                    "description": "Working-age population (16-64)"
                },
                "gdhi_total_mn_gbp": {
                    "unit": "GBP_m",
                    "transform": "log",
                    "monotonic": False,
                    "reconcile": True,
                    "description": "Gross Disposable Household Income (total)"
                },
                "nominal_gva_mn_gbp": {
                    "unit": "GBP_m",
                    "transform": "log",
                    "monotonic": False,
                    "reconcile": True,
                    "description": "Gross Value Added, current prices"
                },
                "emp_total_jobs": {
                    "unit": "jobs",
                    "transform": "log",
                    "monotonic": False,
                    "reconcile": True,
                    "description": "Total employment (jobs)"
                },
                "emp_total_jobs_ni": {
                    "unit": "jobs",
                    "transform": "log",
                    "monotonic": False,
                    "reconcile": True,
                    "description": "NI employee jobs (NI-only metric)"
                },
                # Rate metrics from NOMIS (NOT reconciled, NOT derived)
                "employment_rate_pct": {
                    "unit": "percent",
                    "transform": "none",
                    "monotonic": False,
                    "reconcile": False,
                    "bounds": (40.0, 95.0),
                    "description": "Employment rate"
                },
                "unemployment_rate_pct": {
                    "unit": "percent",
                    "transform": "none",
                    "monotonic": False,
                    "reconcile": False,
                    "bounds": (1.0, 20.0),
                    "description": "Unemployment rate"
                }
            }

        # Log transform flags
        if self.use_log_transform is None:
            self.use_log_transform = {
                metric: info.get("transform") == "log"
                for metric, info in self.metric_definitions.items()
            }

        # Structural breaks
        if self.structural_breaks is None:
            self.structural_breaks = [
                {"year": 2008, "name": "Financial Crisis", "type": "level"},
                {"year": 2009, "name": "Crisis Recovery", "type": "trend"},
                {"year": 2016, "name": "Brexit Vote", "type": "trend"},
                {"year": 2020, "name": "COVID-19", "type": "level"},
                {"year": 2021, "name": "COVID Recovery", "type": "trend"}
            ]

        self._validate()

    def _validate(self):
        if not self.silver_dir.exists():
            raise FileNotFoundError(f"Silver dir not found: {self.silver_dir}")
        assert self.target_year > 2024
        logger.info("=" * 70)
        logger.info("ITL2 V4.1 MERGED CONFIGURATION (NOMIS RATES + GROWTH DAMPENING)")
        logger.info("=" * 70)


# =============================================================================
# ITL1 Anchor Manager
# =============================================================================

class ITL1AnchorManager:
    """Loads ITL1 forecasts for top-down reconciliation and parent CAGRs."""

    def __init__(self, duckdb_path: Path):
        self.duckdb_path = duckdb_path
        self.anchors = self._load_anchors()

    def _load_anchors(self) -> pd.DataFrame:
        """Load ITL1 forecasts from DuckDB gold.itl1_forecast."""
        if not HAVE_DUCKDB or not self.duckdb_path.exists():
            logger.warning("Cannot load ITL1 anchors - DuckDB unavailable")
            return pd.DataFrame()

        try:
            con = duckdb.connect(str(self.duckdb_path), read_only=True)
            anchors = con.execute("""
                SELECT
                    region_code,
                    metric_id,
                    period,
                    value,
                    data_type
                FROM gold.itl1_forecast
            """).fetchdf()
            con.close()

            if anchors.empty:
                logger.warning("No ITL1 data in gold.itl1_forecast")
                return pd.DataFrame()

            anchors["year"] = pd.to_numeric(anchors["period"], errors="coerce").astype(int)
            logger.info(
                f"✓ ITL1 anchors loaded: {anchors['metric_id'].nunique()} metrics, "
                f"{anchors['region_code'].nunique()} regions"
            )
            return anchors

        except Exception as e:
            logger.error(f"Failed to load ITL1 anchors: {e}")
            return pd.DataFrame()

    def has_anchors(self) -> bool:
        return not self.anchors.empty

    def get_parent_value(self, metric_id: str, year: int, parent_region_code: str) -> Optional[float]:
        """Get ITL1 parent value for given metric/year/region (FORECAST only)."""
        if self.anchors.empty:
            return None

        mask = (
            (self.anchors["metric_id"] == metric_id) &
            (self.anchors["year"] == year) &
            (self.anchors["region_code"] == parent_region_code) &
            (self.anchors["data_type"] == "forecast")
        )
        match = self.anchors[mask]
        if match.empty:
            return None
        return float(match["value"].iloc[0])


# =============================================================================
# Top-Down Reconciler ITL2 → ITL1
# =============================================================================

class TopDownReconcilerITL2:
    """Reconciles ITL2 forecasts to ITL1 parent totals for additive metrics."""

    def __init__(self, config: ForecastConfigITL2, anchor_manager: ITL1AnchorManager):
        self.config = config
        self.anchors = anchor_manager
        self.parent_mapping = self._build_parent_mapping()

    def _build_parent_mapping(self) -> pd.DataFrame:
        lookup_path = Path("data/reference/master_2025_geography_lookup.csv")

        if not lookup_path.exists():
            logger.warning(f"Lookup file not found: {lookup_path}")
            return pd.DataFrame()

        try:
            lookup = pd.read_csv(lookup_path)
            lookup.columns = [col.replace("\ufeff", "") for col in lookup.columns]

            mapping = lookup[["ITL225CD", "ITL125CD"]].drop_duplicates()
            mapping.columns = ["itl2_code", "itl1_code_tlc"]

            # TLC → ONS ITL1 codes
            TLC_TO_ONS = {
                "TLC": "E12000001", "TLD": "E12000002", "TLE": "E12000003",
                "TLF": "E12000004", "TLG": "E12000005", "TLH": "E12000006",
                "TLI": "E12000007", "TLJ": "E12000008", "TLK": "E12000009",
                "TLL": "W92000004", "TLM": "S92000003", "TLN": "N92000002"
            }

            mapping["itl1_code"] = mapping["itl1_code_tlc"].map(TLC_TO_ONS)
            mapping = mapping.drop(columns=["itl1_code_tlc"])

            logger.info(
                f"✓ ITL2→ITL1 mapping: {len(mapping)} ITL2 regions → "
                f"{mapping['itl1_code'].nunique()} ITL1 parents"
            )
            return mapping

        except Exception as e:
            logger.error(f"Failed to build parent mapping: {e}")
            return pd.DataFrame()

    def reconcile(self, data: pd.DataFrame) -> pd.DataFrame:
        """Reconcile ADDITIVE metrics to ITL1, then recalculate GDHI per head."""
        if not self.anchors.has_anchors():
            logger.warning("No ITL1 anchors - skipping reconciliation")
            return data

        if self.parent_mapping.empty:
            logger.warning("No parent mapping - skipping reconciliation")
            return data

        # Merge parent codes
        data = data.merge(
            self.parent_mapping,
            left_on="region_code",
            right_on="itl2_code",
            how="left"
        )
        if "itl2_code" in data.columns:
            data = data.drop(columns=["itl2_code"])

        logger.info("\n" + "=" * 70)
        logger.info("TOP-DOWN RECONCILIATION ITL2 → ITL1")
        logger.info("=" * 70)

        reconciliation_log = []
        forecast_data = data[data["data_type"] == "forecast"]

        if forecast_data.empty:
            logger.info("No forecast data to reconcile.")
            data.attrs["reconciliation_log"] = reconciliation_log
            return data

        forecast_years = sorted(forecast_data["year"].unique())

        # STEP 1: Reconcile additive metrics
        for metric in forecast_data["metric"].unique():
            itl1_metric = self.config.macro_anchor_map.get(metric)
            if not itl1_metric:
                continue

            logger.info(f"\n  Reconciling {metric} → ITL1")

            for parent_region in forecast_data["itl1_code"].dropna().unique():
                for year in forecast_years:
                    year_int = int(year)

                    itl1_value = self.anchors.get_parent_value(itl1_metric, year_int, parent_region)
                    if itl1_value is None:
                        continue

                    mask = (
                        (data["year"] == year_int) &
                        (data["metric"] == metric) &
                        (data["data_type"] == "forecast") &
                        (data["itl1_code"] == parent_region)
                    )

                    if not mask.any():
                        continue

                    regional_sum_before = data.loc[mask, "value"].sum()
                    if regional_sum_before <= 0:
                        continue

                    scale_factor = itl1_value / regional_sum_before

                    data.loc[mask, "value"] *= scale_factor
                    if "ci_lower" in data.columns:
                        data.loc[mask, "ci_lower"] *= scale_factor
                    if "ci_upper" in data.columns:
                        data.loc[mask, "ci_upper"] *= scale_factor

                    reconciliation_log.append({
                        "year": year_int,
                        "metric": metric,
                        "parent": parent_region,
                        "scale_factor": scale_factor
                    })

                    if year_int in [2025, 2030, 2040, 2050]:
                        logger.info(f"    {parent_region} {year_int}: SF={scale_factor:.4f}")

        data.attrs["reconciliation_log"] = reconciliation_log
        logger.info(f"\n  ✓ Additive metrics: {len(reconciliation_log)} adjustments")

        # NOTE: gdhi_per_head_gbp is now calculated by DerivedMetricsCalculator
        # after reconciliation, along with other derived metrics

        return data


# =============================================================================
# Data Manager ITL2
# =============================================================================

class DataManagerITL2:
    """Data management for ITL2 - DuckDB preferred, CSV fallback."""

    def __init__(self, config: ForecastConfigITL2):
        self.config = config

    def load_all_data(self) -> pd.DataFrame:
        """Load ITL2 history from DuckDB or CSV."""
        if self.config.use_duckdb and HAVE_DUCKDB and self.config.duckdb_path.exists():
            try:
                con = duckdb.connect(str(self.config.duckdb_path), read_only=True)
                df = con.execute("SELECT * FROM silver.itl2_history").fetchdf()
                con.close()

                if not df.empty:
                    logger.info(f"✓ Loaded ITL2 history from DuckDB: {len(df)} rows")
                    df = self._standardize_columns(df)
                    return df
            except Exception as e:
                logger.warning(f"DuckDB load failed, falling back to CSV: {e}")

        df = self._load_from_csv_files()
        df = self._standardize_columns(df)
        return df

    def _load_from_csv_files(self) -> pd.DataFrame:
        """Load from split ITL2 history CSVs (legacy fallback)."""
        files_to_load = [
            "itl2_population_history.csv",
            "itl2_employment_history.csv",
            "itl2_gva_history.csv",
            "itl2_gdhi_history.csv"
        ]

        dfs = []
        for filename in files_to_load:
            filepath = self.config.silver_dir / filename
            if filepath.exists():
                logger.info(f"Loading: {filename}")
                df = pd.read_csv(filepath)
                dfs.append(df)

        if not dfs:
            raise FileNotFoundError(f"No ITL2 history files found in {self.config.silver_dir}")

        return pd.concat(dfs, ignore_index=True)

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize ITL2 history column names and types."""
        rename_map = {
            "metric_id": "metric",
            "period": "year",
            "region_name": "region"
        }
        for old, new in rename_map.items():
            if old in df.columns and new not in df.columns:
                df[new] = df[old]

        # Ensure required columns exist
        if "region_code" not in df.columns:
            raise KeyError("ITL2 history must contain 'region_code' column")
        if "metric" not in df.columns:
            raise KeyError("ITL2 history must contain 'metric' or 'metric_id' column")
        if "year" not in df.columns:
            raise KeyError("ITL2 history must contain 'year' or 'period' column")
        if "value" not in df.columns:
            raise KeyError("ITL2 history must contain 'value' column")

        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        if "data_type" not in df.columns:
            df["data_type"] = "historical"

        df = df.dropna(subset=["year", "value"])

        # Allow zero/negative only for rate metrics
        rate_metrics = ["employment_rate_pct", "unemployment_rate_pct"]
        df = df[(df["value"] > 0) | df["metric"].isin(rate_metrics)]

        logger.info(
            f"  Standardized ITL2: {len(df)} rows, "
            f"{df['region_code'].nunique()} regions, "
            f"{df['metric'].nunique()} metrics"
        )
        logger.info(f"  Metrics: {sorted(df['metric'].unique())}")
        return df


# =============================================================================
# VAR System Forecaster
# =============================================================================

class VARSystemForecaster:
    """VAR/VECM for multi-metric systems (e.g. GVA + Employment)."""

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
        if not HAVE_STATSMODELS:
            return None

        try:
            series_dict = {}
            for metric in metrics:
                metric_data = data[
                    (data["region_code"] == region_code) &
                    (data["metric"] == metric)
                ].sort_values("year")

                if metric_data.empty:
                    return None

                values = metric_data["value"].values
                if self.config.use_log_transform.get(metric, False) and (values > 0).all():
                    values = np.log(values)

                series_dict[metric] = pd.Series(
                    values,
                    index=metric_data["year"].values.astype(int)
                )

            system_df = pd.DataFrame(series_dict).dropna()
            if len(system_df) < self.config.min_history_years:
                return None

            # Cointegration test
            coint_result = self._test_cointegration(system_df)

            if coint_result["cointegrated"] and coint_result["rank"] > 0:
                model = VECM(
                    system_df,
                    k_ar_diff=min(self.config.max_var_lags, 3),
                    coint_rank=coint_result["rank"],
                    deterministic="ci"
                )
                fitted = model.fit()
                method_name = f"VECM(r={coint_result['rank']})"
                forecast_values = fitted.predict(steps=horizon)
            else:
                model = VAR(system_df)
                lag_results = model.select_order(maxlags=self.config.max_var_lags)
                try:
                    optimal_lags = lag_results.selected_orders["aic"]
                except (AttributeError, KeyError):
                    optimal_lags = lag_results.aic
                optimal_lags = max(1, min(optimal_lags, self.config.max_var_lags))

                fitted = model.fit(maxlags=optimal_lags)
                method_name = f"VAR({optimal_lags})"
                forecast_values = fitted.forecast(
                    system_df.values[-fitted.k_ar:], steps=horizon
                )

            # Bootstrap CIs
            ci_lower, ci_upper = self._bootstrap_ci(fitted, system_df, horizon)

            last_year = int(system_df.index[-1])
            forecast_years = list(range(last_year + 1, last_year + horizon + 1))

            results = {}
            for i, metric in enumerate(metrics):
                vals = forecast_values[:, i]
                ci_l, ci_u = ci_lower[:, i], ci_upper[:, i]

                if self.config.use_log_transform.get(metric, False):
                    vals, ci_l, ci_u = np.exp(vals), np.exp(ci_l), np.exp(ci_u)

                results[metric] = {
                    "method": method_name,
                    "values": vals,
                    "years": forecast_years,
                    "ci_lower": ci_l,
                    "ci_upper": ci_u,
                    "aic": getattr(fitted, "aic", None),
                    "system_metrics": metrics,
                    "cointegrated": coint_result["cointegrated"]
                }

            if not self._validate_results(results, region_code):
                return None

            return results

        except Exception as e:
            logger.debug(f"VAR system failed for {region_code}: {e}")
            return None

    def _validate_results(self, results: Dict, region_code: str) -> bool:
        for metric, fc in results.items():
            if np.isnan(fc["values"]).any():
                return False

            if metric in ["emp_total_jobs", "nominal_gva_mn_gbp", "population_total"]:
                if (fc["values"] < 0).any():
                    return False
                long_term = fc["values"][-min(10, len(fc["values"])):]
                if (long_term < 1e-6).any():
                    return False

            pct_changes = np.diff(fc["values"]) / (fc["values"][:-1] + 1e-10)
            if np.any(pct_changes > 0.25):
                return False

        return True

    def _test_cointegration(self, data: pd.DataFrame) -> Dict:
        if len(data) < 15:
            return {"cointegrated": False, "rank": 0}
        try:
            result = coint_johansen(data, det_order=0, k_ar_diff=2)
            rank = int(np.sum(result.lr1 > result.cvt[:, 1]))
            return {"cointegrated": rank > 0, "rank": rank}
        except Exception:
            return {"cointegrated": False, "rank": 0}

    def _bootstrap_ci(self, fitted, data, horizon, n_boot: int = 200):
        n_vars = data.shape[1]
        boot_fcs = []

        residuals = getattr(fitted, "resid", np.zeros((len(data), n_vars)))

        for _ in range(n_boot):
            try:
                boot_resid = residuals[np.random.choice(len(residuals), len(residuals), replace=True)]
                boot_df = pd.DataFrame(
                    fitted.fittedvalues + boot_resid,
                    columns=data.columns,
                    index=data.index[-len(fitted.fittedvalues):]
                )

                if hasattr(fitted, "coint_rank"):
                    boot_model = VECM(
                        boot_df, k_ar_diff=fitted.k_ar_diff,
                        coint_rank=fitted.coint_rank, deterministic="ci"
                    ).fit()
                    boot_fc = boot_model.predict(steps=horizon)
                else:
                    boot_model = VAR(boot_df).fit(maxlags=fitted.k_ar)
                    boot_fc = boot_model.forecast(boot_df.values[-fitted.k_ar:], steps=horizon)

                boot_fcs.append(boot_fc)
            except Exception:
                continue

        if len(boot_fcs) < 50:
            resid_std = np.std(residuals, axis=0)
            fc_std = resid_std * np.sqrt(np.arange(1, horizon + 1))[:, None]
            return -fc_std * 1.96, fc_std * 1.96

        boot_array = np.array(boot_fcs)
        return (
            np.percentile(boot_array, 2.5, axis=0),
            np.percentile(boot_array, 97.5, axis=0)
        )


# =============================================================================
# Growth Dampening Engine
# =============================================================================

class GrowthDampeningEngine:
    """
    Applies growth constraints to prevent runaway extrapolation.

    Three-layer dampening:
      1. Hard caps (absolute maximum annual growth)
      2. Historical ceiling (cap at historical CAGR + buffer)
      3. Mean reversion (blend toward parent/national growth over time)
    """

    def __init__(self, config: ForecastConfigITL2):
        self.config = config
        self.parent_growth_rates: Dict[Tuple[str, str], float] = {}

    def load_parent_growth_rates(self, itl1_forecasts: pd.DataFrame):
        """Pre-compute ITL1 parent growth rates for mean reversion from ITL1 forecast."""
        if itl1_forecasts.empty:
            logger.info("No ITL1 anchors for growth dampening.")
            return

        logger.info("  Computing ITL1 parent growth rates for mean reversion...")

        for metric in self.config.macro_anchor_map.keys():
            metric_data = itl1_forecasts[itl1_forecasts["metric_id"] == metric]
            if metric_data.empty:
                continue

            for region_code in metric_data["region_code"].unique():
                region_data = metric_data[
                    (metric_data["region_code"] == region_code) &
                    (metric_data["data_type"] == "forecast")
                ].sort_values("year")

                if len(region_data) < 2:
                    continue

                values = region_data["value"].values
                years = region_data["year"].values

                if len(values) > 1 and values[0] > 0:
                    total_growth = values[-1] / values[0]
                    n_years = years[-1] - years[0]
                    if n_years > 0 and total_growth > 0:
                        cagr = total_growth ** (1.0 / n_years) - 1.0
                        self.parent_growth_rates[(region_code, metric)] = cagr

        logger.info(f"    Computed {len(self.parent_growth_rates)} parent growth rates")

    def calculate_historical_cagr(
        self,
        historical: pd.Series,
        min_years: int = 5
    ) -> Optional[float]:
        """Calculate historical CAGR from a time series."""
        if len(historical) < min_years:
            return None

        lookback = min(10, len(historical))
        start_val = historical.iloc[-lookback]
        end_val = historical.iloc[-1]

        if start_val <= 0 or end_val <= 0:
            return None

        cagr = (end_val / start_val) ** (1.0 / lookback) - 1.0
        return cagr

    def get_parent_growth_rate(
        self,
        itl2_code: str,
        metric: str,
        itl2_to_itl1_map: Dict[str, str]
    ) -> Optional[float]:
        """Get ITL1 parent's growth rate for mean reversion."""
        itl1_code = itl2_to_itl1_map.get(itl2_code)
        if itl1_code is None:
            return None
        return self.parent_growth_rates.get((itl1_code, metric))

    def apply_growth_dampening(
        self,
        forecast_values: np.ndarray,
        forecast_years: np.ndarray,
        historical: pd.Series,
        metric: str,
        itl2_code: str,
        itl2_to_itl1_map: Dict[str, str],
        ci_lower: Optional[np.ndarray] = None,
        ci_upper: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Apply three-layer growth dampening:
          1. Hard caps
          2. Historical ceiling
          3. Mean reversion to parent

        Returns: (dampened_values, dampened_ci_lower, dampened_ci_upper)
        """
        if len(forecast_values) == 0:
            return forecast_values, ci_lower, ci_upper

        # Get caps for metric
        max_growth = self.config.max_annual_growth.get(metric, 0.10)
        min_growth = self.config.min_annual_growth.get(metric, -0.05)

        # Historical ceiling
        hist_cagr = self.calculate_historical_cagr(historical)
        if hist_cagr is not None:
            hist_ceiling = hist_cagr + self.config.historical_cagr_buffer
            max_growth = min(max_growth, hist_ceiling)

        # ITL1 parent growth rate
        parent_cagr = self.get_parent_growth_rate(itl2_code, metric, itl2_to_itl1_map)

        dampened = np.zeros_like(forecast_values, dtype=float)
        last_value = historical.iloc[-1]
        base_year = int(historical.index[-1])

        for i, (value, year) in enumerate(zip(forecast_values, forecast_years)):
            prev_value = last_value if i == 0 else dampened[i - 1]

            if prev_value <= 0:
                dampened[i] = value
                continue

            raw_growth = (value - prev_value) / prev_value

            # Layer 1: hard caps
            capped_growth = np.clip(raw_growth, min_growth, max_growth)

            # Layer 2: mean reversion toward ITL1 parent
            years_out = int(year) - base_year
            if parent_cagr is not None and years_out > self.config.mean_reversion_start_year:
                if self.config.mean_reversion_full_year > self.config.mean_reversion_start_year:
                    reversion_progress = min(
                        1.0,
                        (years_out - self.config.mean_reversion_start_year) /
                        (self.config.mean_reversion_full_year - self.config.mean_reversion_start_year)
                    )
                else:
                    reversion_progress = 1.0

                blend_weight = reversion_progress * self.config.mean_reversion_weight
                capped_growth = (1 - blend_weight) * capped_growth + blend_weight * parent_cagr

            dampened[i] = prev_value * (1 + capped_growth)

        # Scale CIs proportionally if present
        if ci_lower is not None and ci_upper is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                scale_factors = np.where(
                    np.abs(forecast_values) > 1e-10,
                    dampened / (forecast_values + 1e-10),
                    1.0
                )
            dampened_ci_lower = ci_lower * scale_factors
            dampened_ci_upper = ci_upper * scale_factors
        else:
            dampened_ci_lower = ci_lower
            dampened_ci_upper = ci_upper

        return dampened, dampened_ci_lower, dampened_ci_upper

    def summarize_dampening(
        self,
        original_values: np.ndarray,
        dampened_values: np.ndarray,
        years: np.ndarray,
        metric: str,
        region_code: str
    ) -> Dict:
        """Generate summary of dampening applied."""
        if len(original_values) < 2:
            return {}

        n_years = int(years[-1] - years[0])
        if n_years <= 0:
            return {}

        if original_values[0] <= 0 or dampened_values[0] <= 0:
            return {}

        original_cagr = (original_values[-1] / original_values[0]) ** (1.0 / n_years) - 1.0
        dampened_cagr = (dampened_values[-1] / dampened_values[0]) ** (1.0 / n_years) - 1.0

        if original_values[-1] <= 0:
            reduction_pct = 0.0
        else:
            reduction_pct = (1.0 - dampened_values[-1] / original_values[-1]) * 100.0

        return {
            "region_code": region_code,
            "metric": metric,
            "original_cagr": original_cagr,
            "dampened_cagr": dampened_cagr,
            "cagr_reduction": original_cagr - dampened_cagr,
            "original_final": original_values[-1],
            "dampened_final": dampened_values[-1],
            "reduction_pct": reduction_pct
        }


# =============================================================================
# Advanced Forecasting (Univariate + Ensemble)
# =============================================================================

class AdvancedForecastingV4:
    """Forecasting engine with ensemble + mechanical constraints."""

    def __init__(self, config: ForecastConfigITL2):
        self.config = config
        self.var_forecaster = VARSystemForecaster(config) if config.use_var_systems else None

    def forecast_univariate(
        self,
        series: pd.Series,
        horizon: int,
        structural_breaks: Optional[Sequence[Dict]] = None,
        metric_info: Optional[Dict] = None
    ) -> Dict:
        """
        Univariate forecasting with ARIMA/ETS/Linear ensemble.

        NOTE: This returns the raw ensemble forecast.
        Growth dampening and mechanical constraints are applied externally.
        """
        models: List[Dict] = []

        # ARIMA
        arima_result = self._fit_arima(series, horizon, structural_breaks)
        if arima_result:
            models.append(arima_result)

        # ETS
        if HAVE_STATSMODELS and len(series) > 20:
            ets_result = self._fit_ets(series, horizon)
            if ets_result:
                models.append(ets_result)

        # Linear trend
        linear_result = self._fit_linear(series, horizon)
        if linear_result:
            models.append(linear_result)

        if not models:
            return self._fallback_forecast(series, horizon)

        if len(models) > 1:
            combined = self._combine_forecasts(models)
        else:
            combined = models[0]

        return combined

    def _fit_arima(
        self,
        series: pd.Series,
        horizon: int,
        structural_breaks: Optional[Sequence[Dict]] = None
    ) -> Optional[Dict]:
        if not HAVE_STATSMODELS:
            return None

        try:
            exog = self._prepare_break_dummies(series, structural_breaks) if structural_breaks else None

            best_model = None
            best_aic = np.inf
            best_order = None

            adf_p = adfuller(series, autolag="AIC")[1]
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
                        aicc = fitted.aic + (2 * k * (k + 1)) / max(1, n - k - 1)
                        if aicc < best_aic:
                            best_aic = aicc
                            best_model = fitted
                            best_order = (p, d, q)
                    except Exception:
                        continue

            if best_model is None:
                return None

            exog_forecast = self._extend_exog(exog, series.index, horizon) if exog is not None else None
            forecast_obj = best_model.get_forecast(steps=horizon, exog=exog_forecast)
            forecast = forecast_obj.predicted_mean
            ci = forecast_obj.conf_int(alpha=0.05)

            last_year = int(series.index[-1])
            forecast_years = list(range(last_year + 1, last_year + horizon + 1))

            if hasattr(ci, "iloc"):
                ci_lower = ci.iloc[:, 0].values
                ci_upper = ci.iloc[:, 1].values
            else:
                ci_lower = ci[:, 0]
                ci_upper = ci[:, 1]

            return {
                "method": f"ARIMA{best_order}",
                "values": forecast.values,
                "years": forecast_years,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "aic": best_aic,
                "order": best_order
            }

        except Exception as e:
            logger.debug(f"ARIMA failed: {e}")
            return None

    def _fit_ets(self, series: pd.Series, horizon: int) -> Optional[Dict]:
        if not HAVE_STATSMODELS:
            return None

        try:
            best_model = None
            best_aic = np.inf
            best_config = None

            for trend in ["add", "mul", None]:
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
                "method": best_config,
                "values": forecast.values,
                "years": forecast_years,
                "ci_lower": ci_lower.values if hasattr(ci_lower, "values") else ci_lower,
                "ci_upper": ci_upper.values if hasattr(ci_upper, "values") else ci_upper,
                "aic": best_aic
            }

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
                X_future = np.arange(len(series), len(series) + horizon).reshape(-1, 1)
                X_future_sm = sm.add_constant(X_future)
                predictions = model.get_prediction(X_future_sm)
                forecast = predictions.predicted_mean
                ci = predictions.conf_int(alpha=0.05)
                ci_lower = ci[:, 0]
                ci_upper = ci[:, 1]
                aic = model.aic
            else:
                coeffs = np.polyfit(X.flatten(), y, 1)
                X_future = np.arange(len(series), len(series) + horizon)
                forecast = np.polyval(coeffs, X_future)
                sigma = (y - np.polyval(coeffs, X.flatten())).std()
                ci_lower = forecast - 1.96 * sigma
                ci_upper = forecast + 1.96 * sigma
                aic = len(series) * np.log(np.var(y - np.polyval(coeffs, X.flatten()))) + 4

            last_year = int(series.index[-1])
            forecast_years = list(range(last_year + 1, last_year + horizon + 1))

            return {
                "method": "Linear",
                "values": np.asarray(forecast),
                "years": forecast_years,
                "ci_lower": np.asarray(ci_lower),
                "ci_upper": np.asarray(ci_upper),
                "aic": aic
            }

        except Exception as e:
            logger.debug(f"Linear forecast failed: {e}")
            return None

    def _combine_forecasts(self, models: List[Dict]) -> Dict:
        """Combine forecasts using AIC weights."""
        aics = np.array([m.get("aic", np.inf) for m in models])
        delta_aic = aics - np.nanmin(aics)
        weights = np.exp(-0.5 * delta_aic)
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        weights = weights / weights.sum()

        n_steps = len(models[0]["values"])
        combined_values = np.zeros(n_steps)
        combined_ci_lower = np.zeros(n_steps)
        combined_ci_upper = np.zeros(n_steps)

        for model, w in zip(models, weights):
            combined_values += w * model["values"]
            combined_ci_lower += w * model["ci_lower"]
            combined_ci_upper += w * model["ci_upper"]

        return {
            "method": f"Ensemble({len(models)})",
            "values": combined_values,
            "years": models[0]["years"],
            "ci_lower": combined_ci_lower,
            "ci_upper": combined_ci_upper,
            "weights": weights.tolist(),
            "component_methods": [m["method"] for m in models]
        }

    def _fallback_forecast(self, series: pd.Series, horizon: int) -> Dict:
        last_value = series.iloc[-1]
        growth_series = series.pct_change().dropna()
        avg_growth = growth_series.mean() if len(growth_series) > 0 else 0.0
        avg_growth = float(np.clip(avg_growth, -0.02, 0.05))

        values = np.array([last_value * (1 + avg_growth) ** i for i in range(1, horizon + 1)])
        last_year = int(series.index[-1])
        forecast_years = list(range(last_year + 1, last_year + horizon + 1))

        std = series.std()
        ci_width = 1.96 * std * np.sqrt(np.arange(1, horizon + 1))

        return {
            "method": "Fallback",
            "values": values,
            "years": forecast_years,
            "ci_lower": values - ci_width,
            "ci_upper": values + ci_width,
            "aic": None
        }

    def _prepare_break_dummies(
        self,
        series: pd.Series,
        breaks: Optional[Sequence[Dict]]
    ) -> Optional[np.ndarray]:
        if not breaks:
            return None

        dummies = []
        years = series.index

        for break_info in breaks:
            if "year" in break_info:
                break_year = break_info["year"]
                break_type = break_info.get("type", "level")

                if any(y >= break_year for y in years):
                    if break_type == "level":
                        dummy = (years >= break_year).astype(int)
                        dummies.append(dummy)
                    elif break_type == "trend":
                        time_since_break = np.maximum(0, years - break_year)
                        dummies.append(time_since_break)

        return np.column_stack(dummies) if dummies else None

    def _extend_exog(
        self,
        exog: np.ndarray,
        historical_index,
        horizon: int
    ) -> Optional[np.ndarray]:
        if exog is None:
            return None

        if len(exog.shape) == 1:
            exog = exog.reshape(-1, 1)

        n_vars = exog.shape[1]
        extended = []

        for h in range(1, horizon + 1):
            row = []
            for i in range(n_vars):
                if np.all(np.isin(exog[:, i], [0, 1])):
                    row.append(exog[-1, i])
                else:
                    increment = exog[-1, i] - exog[-2, i] if len(exog) > 1 else 1
                    row.append(exog[-1, i] + increment * h)
            extended.append(row)

        return np.array(extended)

    def apply_constraints(
        self,
        forecast: Dict,
        historical: pd.Series,
        metric_info: Dict,
        config: ForecastConfigITL2
    ) -> Dict:
        """Apply mechanical constraints: non-negative, monotonic, bounds, growth caps."""
        values = np.asarray(forecast["values"], dtype=float)
        ci_lower = np.asarray(forecast.get("ci_lower", values.copy()), dtype=float)
        ci_upper = np.asarray(forecast.get("ci_upper", values.copy()), dtype=float)

        # Non-negative
        if config.enforce_non_negative and not metric_info.get("bounds"):
            values = np.maximum(values, 0)
            ci_lower = np.maximum(ci_lower, 0)

        # Monotonic population
        if metric_info.get("monotonic", False) and config.enforce_monotonic_population:
            for i in range(1, len(values)):
                values[i] = max(values[i], values[i - 1])

        # Bounds for rates
        bounds = metric_info.get("bounds")
        if bounds:
            lower_bound, upper_bound = bounds
            values = np.clip(values, lower_bound, upper_bound)
            ci_lower = np.clip(ci_lower, lower_bound, upper_bound)
            ci_upper = np.clip(ci_upper, lower_bound, upper_bound)

        # Growth rate caps for non-rate metrics (percentile-based)
        if config.growth_rate_cap_percentiles and not bounds:
            historical_growth = historical.pct_change().dropna()
            if len(historical_growth) > 5:
                p_low, p_high = config.growth_rate_cap_percentiles
                growth_bounds = np.percentile(historical_growth, [p_low, p_high])

                last_value = historical.iloc[-1]
                for i in range(len(values)):
                    prev_value = last_value if i == 0 else values[i - 1]
                    if prev_value > 0:
                        growth = (values[i] - prev_value) / prev_value
                        if growth < growth_bounds[0]:
                            values[i] = prev_value * (1 + growth_bounds[0])
                        elif growth > growth_bounds[1]:
                            values[i] = prev_value * (1 + growth_bounds[1])

        forecast["values"] = values
        forecast["ci_lower"] = ci_lower
        forecast["ci_upper"] = ci_upper
        return forecast


# =============================================================================
# Derived Metrics (Monte Carlo)
# =============================================================================

class DerivedMetricsV4:
    """Calculate derived indicators with Monte Carlo uncertainty propagation."""

    @staticmethod
    def calculate_with_uncertainty(
        numerator: Tuple[float, float, float],
        denominator: Tuple[float, float, float],
        operation: str = "divide",
        n_simulations: int = 5000
    ) -> Tuple[float, float, float]:
        num_val, num_low, num_high = numerator
        den_val, den_low, den_high = denominator

        # Handle NaN CIs (historical components in mixed-vintage scenarios)
        import math
        if math.isnan(num_low) or math.isnan(num_high):
            num_se = max(abs(num_val) * 0.02, 1e-10)  # 2% uncertainty for historical
        else:
            num_se = max((num_high - num_low) / (2 * 1.96), 1e-10)

        if math.isnan(den_low) or math.isnan(den_high):
            den_se = max(abs(den_val) * 0.02, 1e-10)  # 2% uncertainty for historical
        else:
            den_se = max((den_high - den_low) / (2 * 1.96), 1e-10)

        if operation == "divide" and abs(den_val) < 1e-10:
            return 0.0, 0.0, 0.0

        num_samples = np.random.normal(num_val, num_se, n_simulations)
        den_samples = np.random.normal(den_val, den_se, n_simulations)

        if operation == "divide":
            den_samples = np.where(np.abs(den_samples) < 1e-10, 1e-10, den_samples)
            result_samples = num_samples / den_samples
        else:
            result_samples = num_samples * den_samples

        result_val = float(np.mean(result_samples))  # Use mean, not median
        result_low = float(np.percentile(result_samples, 2.5))
        result_high = float(np.percentile(result_samples, 97.5))

        # Ensure proper ordering (FIX for inverted CIs from skewed distributions)
        if result_low > result_high:
            result_low, result_high = result_high, result_low

        # Clamp point estimate within CI bounds
        result_val = np.clip(result_val, result_low, result_high)

        # Enforce non-negative for productivity/income metrics
        result_val = max(0.0, result_val)
        result_low = max(0.0, result_low)
        result_high = max(0.0, result_high)

        return result_val, result_low, result_high

    @staticmethod
    def calculate_all_derived(
        data: pd.DataFrame,
        region_code: str,
        year: int
    ) -> List[Dict]:
        """Calculate derived metrics for a given region-year."""
        derived: List[Dict] = []
        region_year_data = data[
            (data["region_code"] == region_code) &
            (data["year"] == year)
        ]

        if region_year_data.empty:
            return derived

        metrics_dict: Dict[str, Tuple[float, float, float]] = {}
        for _, row in region_year_data.iterrows():
            metric = row["metric"]
            value = row["value"]
            ci_lower = row.get("ci_lower", value * 0.95)
            ci_upper = row.get("ci_upper", value * 1.05)
            metrics_dict[metric] = (value, ci_lower, ci_upper)

        region = region_year_data["region"].iloc[0] if "region" in region_year_data.columns else region_code
        data_type = "forecast" if year > 2024 else "historical"

        # Productivity (GVA per worker)
        if "nominal_gva_mn_gbp" in metrics_dict and "emp_total_jobs" in metrics_dict:
            gva_pounds = tuple(v * 1e6 for v in metrics_dict["nominal_gva_mn_gbp"])
            prod_val, prod_low, prod_high = DerivedMetricsV4.calculate_with_uncertainty(
                gva_pounds, metrics_dict["emp_total_jobs"], "divide"
            )
            derived.append({
                "region": region,
                "region_code": region_code,
                "year": year,
                "metric": "productivity_gbp_per_job",
                "value": prod_val,
                "ci_lower": prod_low,
                "ci_upper": prod_high,
                "method": "derived_monte_carlo",
                "data_type": data_type
            })

        # Income per worker
        if "gdhi_total_mn_gbp" in metrics_dict and "emp_total_jobs" in metrics_dict:
            gdhi_pounds = tuple(v * 1e6 for v in metrics_dict["gdhi_total_mn_gbp"])
            income_val, income_low, income_high = DerivedMetricsV4.calculate_with_uncertainty(
                gdhi_pounds, metrics_dict["emp_total_jobs"], "divide"
            )
            derived.append({
                "region": region,
                "region_code": region_code,
                "year": year,
                "metric": "income_per_worker_gbp",
                "value": income_val,
                "ci_lower": income_low,
                "ci_upper": income_high,
                "method": "derived_monte_carlo",
                "data_type": data_type
            })

        # GDHI per head
        if "gdhi_total_mn_gbp" in metrics_dict and "population_total" in metrics_dict:
            gdhi_pounds = tuple(v * 1e6 for v in metrics_dict["gdhi_total_mn_gbp"])
            per_head_val, per_head_low, per_head_high = DerivedMetricsV4.calculate_with_uncertainty(
                gdhi_pounds, metrics_dict["population_total"], "divide"
            )
            derived.append({
                "region": region,
                "region_code": region_code,
                "year": year,
                "metric": "gdhi_per_head_gbp",
                "value": per_head_val,
                "ci_lower": per_head_low,
                "ci_upper": per_head_high,
                "method": "derived_monte_carlo",
                "data_type": data_type
            })

        return derived


# =============================================================================
# Main Pipeline - InstitutionalForecasterITL2V4_1
# =============================================================================

class InstitutionalForecasterITL2V4:
    """V4.1 Production pipeline for ITL2 (merged architecture)."""

    def __init__(self, config: ForecastConfigITL2):
        self.config = config
        self.data_manager = DataManagerITL2(config)
        self.forecaster = AdvancedForecastingV4(config)
        self.derived = DerivedMetricsV4()
        self._dampening_summaries: List[Dict] = []

        # ITL1 anchors + reconciler
        if config.use_macro_anchoring and HAVE_DUCKDB:
            self.anchor_manager = ITL1AnchorManager(config.duckdb_path)
            if self.anchor_manager.has_anchors():
                self.reconciler = TopDownReconcilerITL2(config, self.anchor_manager)
            else:
                self.reconciler = None
        else:
            self.anchor_manager = None
            self.reconciler = None

        # ITL2 → ITL1 mapping (for mean reversion)
        if self.reconciler and not self.reconciler.parent_mapping.empty:
            mapping_df = self.reconciler.parent_mapping.dropna(subset=["itl1_code"])
            self.itl2_to_itl1_map = dict(zip(mapping_df["itl2_code"], mapping_df["itl1_code"]))
        else:
            self.itl2_to_itl1_map = {}

        # Growth dampening engine
        self.growth_dampener = GrowthDampeningEngine(config)
        if self.anchor_manager and self.anchor_manager.has_anchors():
            self.growth_dampener.load_parent_growth_rates(self.anchor_manager.anchors)
        else:
            logger.info("Growth dampening: no ITL1 parent CAGRs (only local caps will apply).")

    def run(self) -> pd.DataFrame:
        """Execute the full ITL2 V4.1 pipeline."""
        # Load historical data
        historical = self.data_manager.load_all_data()

        # Identify tasks
        tasks = self._identify_tasks(historical)
        logger.info(f"✓ Forecasting tasks identified: {len(tasks)}")

        # Run forecasts
        results = self._run_forecasts(historical, tasks)
        logger.info(f"✓ Forecast observations generated: {len(results)}")

        # Combine historical + forecasts
        all_data = pd.concat([historical, pd.DataFrame(results)], ignore_index=True)

        # Reconcile additive metrics to ITL1
        if self.reconciler:
            all_data = self.reconciler.reconcile(all_data)

        # Fill internal year gaps for rate metrics (prevents ITL3 QA LAD-gate failures)
        all_data = self._fill_internal_rate_year_gaps(all_data)

        # Calculate derived metrics
        logger.info("\nCalculating derived metrics...")
        derived = self._calculate_derived(all_data)
        if derived:
            all_data = pd.concat([all_data, pd.DataFrame(derived)], ignore_index=True)
            logger.info(f"✓ Derived metric observations: {len(derived)}")

        # Fix any inverted CIs (belt-and-suspenders check)
        all_data = self._fix_inverted_cis(all_data)

        # Save outputs
        self._save_outputs(all_data)

        return all_data

    def _fill_internal_rate_year_gaps(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure continuous year coverage for rate metrics within the LAD-gated window.

        ITL3 QA checks that, for each ITL3 and each base metric, there are no missing
        years between min(period) and max(period) for period>=2020. ITL3 rate series
        often inherit from ITL2; if ITL2 has holes (e.g. 2023-2024 missing between
        2022 historical and 2025+ forecast), ITL3 will inherit those holes and QA fails.

        We fill *internal* gaps (between existing years) for rate metrics by linear
        interpolation of value and CI bounds. Filled rows are marked data_type='forecast'
        with method='gap_fill_interpolate'.
        """
        required_cols = {"region_code", "region", "metric", "year", "value"}
        if not required_cols.issubset(set(data.columns)):
            return data

        rate_metrics = {"employment_rate_pct", "unemployment_rate_pct"}
        df = data.copy()

        filled_rows = []
        for (region_code, metric), g in df[df["metric"].isin(rate_metrics)].groupby(["region_code", "metric"]):
            gg = g.copy()
            gg["year"] = gg["year"].astype(int)
            gg = gg.sort_values("year")

            # Apply the same window as ITL3 QA uses for LAD readiness (period>=2020).
            gg_w = gg[gg["year"] >= 2020].copy()
            if gg_w.empty:
                continue

            years = sorted(set(gg_w["year"].tolist()))
            y0, y1 = years[0], years[-1]
            full = list(range(y0, y1 + 1))
            missing = [y for y in full if y not in years]
            if not missing:
                continue

            # Interpolate value + CIs across the full span using the existing points.
            series = gg.set_index("year").sort_index()
            # Use 'value' as base; CI columns are optional.
            v = series["value"].astype(float)
            v_full = v.reindex(full).interpolate(method="linear").ffill().bfill()

            if "ci_lower" in series.columns and series["ci_lower"].notna().any():
                lo = series["ci_lower"].astype(float).reindex(full).interpolate(method="linear").ffill().bfill()
            else:
                lo = v_full * 0.95

            if "ci_upper" in series.columns and series["ci_upper"].notna().any():
                hi = series["ci_upper"].astype(float).reindex(full).interpolate(method="linear").ffill().bfill()
            else:
                hi = v_full * 1.05

            # Preserve region label (prefer any existing value)
            region_name = gg["region"].iloc[0] if "region" in gg.columns and len(gg) else region_code

            for y in missing:
                filled_rows.append({
                    "region": region_name,
                    "region_code": region_code,
                    "metric": metric,
                    "year": int(y),
                    "value": float(v_full.loc[y]),
                    "ci_lower": float(lo.loc[y]),
                    "ci_upper": float(hi.loc[y]),
                    "method": "gap_fill_interpolate",
                    "data_type": "forecast",
                })

        if not filled_rows:
            return df

        filled = pd.DataFrame(filled_rows)
        out = pd.concat([df, filled], ignore_index=True)

        # Deduplicate (historical wins), in case a year exists elsewhere.
        if "data_type" in out.columns:
            out["_prio"] = out["data_type"].map({"historical": 0, "forecast": 1}).fillna(1)
            out = (
                out.sort_values(["region_code", "metric", "year", "_prio"])
                   .drop_duplicates(subset=["region_code", "metric", "year"], keep="first")
                   .drop(columns=["_prio"])
            )

        logger.info(f"✓ Filled {len(filled)} internal rate year gaps at ITL2 (period>=2020)")
        return out

    def _fix_inverted_cis(self, data: pd.DataFrame) -> pd.DataFrame:
        """Swap any remaining inverted CIs."""
        if 'ci_lower' not in data.columns or 'ci_upper' not in data.columns:
            return data

        inverted = data['ci_lower'] > data['ci_upper']
        n_inverted = inverted.sum()

        if n_inverted > 0:
            data.loc[inverted, ['ci_lower', 'ci_upper']] = data.loc[inverted, ['ci_upper', 'ci_lower']].values
            logger.info(f"  ✓ Fixed {n_inverted} inverted CIs")

        return data

    def _identify_tasks(self, data: pd.DataFrame) -> List[Dict]:
        """Identify forecasting tasks (region × metric)."""
        tasks: List[Dict] = []
        derived_only = [
            "productivity_gbp_per_job",
            "gdhi_per_head_gbp",
            "income_per_worker_gbp"
        ]
        rate_metrics = {"employment_rate_pct", "unemployment_rate_pct"}

        for (region_code, region, metric), group in data.groupby(["region_code", "region", "metric"]):
            if metric in derived_only:
                continue
            # NI-only jobs series is short (BRESHEADLGD), but we still want a forecast
            # so the NI share cascade can run. Allow a shorter history requirement.
            min_years = 8 if metric == "emp_total_jobs_ni" else self.config.min_history_years
            # NI rate series at ITL2 (TLN0) is shorter once we remove bogus early-year placeholders.
            # Allow forecasting NI rates with shorter history so ITL3/LAD can inherit coherently.
            if metric in rate_metrics and str(region_code).startswith("TLN"):
                min_years = min(min_years, 6)
            if len(group) < min_years:
                continue

            last_year = int(group["year"].max())
            horizon = self.config.target_year - last_year
            if horizon <= 0:
                continue

            tasks.append({
                "region_code": region_code,
                "region": region,
                "metric": metric,
                "horizon": horizon,
                "last_year": last_year,
                "history_length": len(group),
                "metric_info": self.config.metric_definitions.get(metric, {})
            })

        task_df = pd.DataFrame(tasks)
        if not task_df.empty:
            logger.info("  Tasks by metric:")
            for metric, count in task_df.groupby("metric").size().items():
                logger.info(f"    {metric}: {count} regions")

        return tasks

    def _forecast_rate_mean_revert(
        self,
        history: pd.DataFrame,
        forecast_years: List[int],
        metric: str,
        bounds: Tuple[float, float],
        fallback_mean: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
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
        
        Returns dict with keys: 'values', 'ci_lower', 'ci_upper', 'method', 'years'
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
            'method': 'mean_revert_regional',
            'years': np.array(forecast_years, dtype=int)
        }
    
    def _apply_growth_dampening(
        self,
        forecast: Dict,
        historical_series: pd.Series,
        metric: str,
        region_code: str
    ) -> Dict:
        """Apply growth dampening to a single forecast dict."""
        if forecast is None or len(forecast.get("values", [])) == 0:
            return forecast

        if self.growth_dampener is None:
            return forecast

        try:
            original_values = np.asarray(forecast["values"], dtype=float)
            years = np.asarray(forecast["years"], dtype=int)
            ci_lower = np.asarray(forecast.get("ci_lower")) if forecast.get("ci_lower") is not None else None
            ci_upper = np.asarray(forecast.get("ci_upper")) if forecast.get("ci_upper") is not None else None

            dampened_values, damp_ci_lower, damp_ci_upper = self.growth_dampener.apply_growth_dampening(
                forecast_values=original_values,
                forecast_years=years,
                historical=historical_series,
                metric=metric,
                itl2_code=region_code,
                itl2_to_itl1_map=self.itl2_to_itl1_map,
                ci_lower=ci_lower,
                ci_upper=ci_upper
            )

            forecast["values"] = dampened_values
            if damp_ci_lower is not None:
                forecast["ci_lower"] = damp_ci_lower
            if damp_ci_upper is not None:
                forecast["ci_upper"] = damp_ci_upper

            summary = self.growth_dampener.summarize_dampening(
                original_values=original_values,
                dampened_values=dampened_values,
                years=years,
                metric=metric,
                region_code=region_code
            )
            if summary:
                self._dampening_summaries.append(summary)

        except Exception as e:
            logger.debug(f"Growth dampening failed for {region_code}-{metric}: {e}")

        return forecast

    def _run_forecasts(self, data: pd.DataFrame, tasks: List[Dict]) -> List[Dict]:
        """Run forecasts with VAR systems + univariate ensemble + dampening."""
        results: List[Dict] = []
        processed: set = set()

        # Group tasks by region
        tasks_by_region: Dict[str, List[Dict]] = {}
        for task in tasks:
            rc = task["region_code"]
            tasks_by_region.setdefault(rc, []).append(task)

        total_regions = len(tasks_by_region)

        for idx, (region_code, region_tasks) in enumerate(tasks_by_region.items()):
            if idx % 10 == 0:
                logger.info(f"\n📍 Forecasting progress: {idx + 1}/{total_regions} regions")

            region_metrics = {t["metric"] for t in region_tasks}

            # VAR systems
            for system_name, system_metrics in self.config.var_systems.items():
                available = [m for m in system_metrics if m in region_metrics]

                if (
                    len(available) >= 2 and
                    self.config.use_var_systems and
                    self.forecaster.var_forecaster
                ):
                    horizons = [t["horizon"] for t in region_tasks if t["metric"] in available]
                    common_horizon = min(horizons) if horizons else 0

                    if common_horizon > 0:
                        breaks = [
                            b for b in self.config.structural_breaks
                            if any(
                                t["last_year"] - 20 <= b["year"] <= t["last_year"]
                                for t in region_tasks
                                if t["metric"] in available
                            )
                        ]

                        var_results = self.forecaster.var_forecaster.forecast_system(
                            data, region_code, available, common_horizon, breaks
                        )

                        if var_results:
                            for metric, fc in var_results.items():
                                processed.add((region_code, metric))
                                task = next(t for t in region_tasks if t["metric"] == metric)

                                # Growth dampening (values already on natural scale)
                                series_data = data[
                                    (data["region_code"] == region_code) &
                                    (data["metric"] == metric)
                                ].sort_values("year")
                                series = pd.Series(
                                    series_data["value"].values,
                                    index=series_data["year"].values.astype(int),
                                    name=metric
                                )

                                fc = self._apply_growth_dampening(fc, series, metric, region_code)
                                fc = self.forecaster.apply_constraints(
                                    fc, series, task["metric_info"], self.config
                                )

                                for j, year in enumerate(fc["years"]):
                                    results.append({
                                        "region": task["region"],
                                        "region_code": region_code,
                                        "metric": metric,
                                        "year": int(year),
                                        "value": float(fc["values"][j]),
                                        "ci_lower": float(fc["ci_lower"][j]),
                                        "ci_upper": float(fc["ci_upper"][j]),
                                        "method": fc["method"],
                                        "data_type": "forecast"
                                    })

            # Univariate for remaining metrics
            for task in region_tasks:
                if (task["region_code"], task["metric"]) in processed:
                    continue

                series_data = data[
                    (data["region_code"] == task["region_code"]) &
                    (data["metric"] == task["metric"])
                ].sort_values("year")

                if series_data.empty:
                    logger.warning(f"No data for {task['region_code']}-{task['metric']}")
                    continue

                series = pd.Series(
                    series_data["value"].values,
                    index=series_data["year"].values.astype(int),
                    name=task["metric"]
                )

                # Route rate metrics to mean reversion
                rate_metrics = ["employment_rate_pct", "unemployment_rate_pct"]
                if task["metric"] in rate_metrics:
                    try:
                        # Get bounds for this metric
                        bounds = task["metric_info"].get("bounds", (0, 100))
                        
                        # Get ITL1 parent mean for fallback if available
                        fallback_mean = None
                        if self.anchor_manager and self.anchor_manager.has_anchors():
                            itl1_code = self.itl2_to_itl1_map.get(task["region_code"])
                            if itl1_code:
                                # Get historical mean from ITL1 parent
                                parent_data = self.anchor_manager.anchors[
                                    (self.anchor_manager.anchors["metric_id"] == task["metric"]) &
                                    (self.anchor_manager.anchors["region_code"] == itl1_code) &
                                    (self.anchor_manager.anchors["data_type"] == "historical")
                                ]
                                if not parent_data.empty:
                                    fallback_mean = float(parent_data["value"].mean())
                        
                        # Convert series_data to DataFrame format for mean reversion
                        history_df = series_data[["year", "value"]].copy()
                        forecast_years = list(range(int(series.index[-1]) + 1, self.config.target_year + 1))
                        
                        fc = self._forecast_rate_mean_revert(
                            history_df, forecast_years, task["metric"], bounds, fallback_mean
                        )
                        
                        if fc is None or "values" not in fc or "years" not in fc:
                            logger.warning(f"Invalid mean reversion forecast for {task['region_code']}-{task['metric']}")
                            continue
                        
                        if len(fc["values"]) == 0 or len(fc["years"]) == 0:
                            logger.warning(f"Empty mean reversion forecast for {task['region_code']}-{task['metric']}")
                            continue
                        
                        # Mean reversion doesn't need growth dampening or constraints
                        # (it's already bounded and mean-reverting)
                        
                        for j, year in enumerate(fc["years"]):
                            results.append({
                                "region": task["region"],
                                "region_code": task["region_code"],
                                "metric": task["metric"],
                                "year": int(year),
                                "value": float(fc["values"][j]),
                                "ci_lower": float(fc["ci_lower"][j]),
                                "ci_upper": float(fc["ci_upper"][j]),
                                "method": fc["method"],
                                "data_type": "forecast"
                            })
                        
                        continue  # Skip to next task
                        
                    except Exception as e:
                        logger.warning(
                            f"Mean reversion failed for {task['region_code']}-{task['metric']}: "
                            f"{type(e).__name__}: {e}"
                        )
                        # Fall through to standard univariate forecasting

                use_log = self.config.use_log_transform.get(task["metric"], False)
                working = np.log(series) if use_log and (series > 0).all() else series

                breaks = [
                    b for b in self.config.structural_breaks
                    if task["last_year"] - 20 <= b["year"] <= task["last_year"]
                ]

                try:
                    fc = self.forecaster.forecast_univariate(
                        working, task["horizon"], breaks, task["metric_info"]
                    )

                    if fc is None or "values" not in fc or "years" not in fc:
                        logger.warning(f"Invalid forecast for {task['region_code']}-{task['metric']}")
                        continue

                    if len(fc["values"]) == 0 or len(fc["years"]) == 0:
                        logger.warning(f"Empty forecast for {task['region_code']}-{task['metric']}")
                        continue

                    # Back-transform if log
                    if use_log:
                        fc["values"] = np.exp(fc["values"])
                        fc["ci_lower"] = np.exp(fc["ci_lower"])
                        fc["ci_upper"] = np.exp(fc["ci_upper"])

                    # Growth dampening
                    fc = self._apply_growth_dampening(fc, series, task["metric"], task["region_code"])

                    # Mechanical constraints
                    fc = self.forecaster.apply_constraints(
                        fc, series, task["metric_info"], self.config
                    )

                    for j, year in enumerate(fc["years"]):
                        results.append({
                            "region": task["region"],
                            "region_code": task["region_code"],
                            "metric": task["metric"],
                            "year": int(year),
                            "value": float(fc["values"][j]),
                            "ci_lower": float(fc["ci_lower"][j]),
                            "ci_upper": float(fc["ci_upper"][j]),
                            "method": fc["method"],
                            "data_type": "forecast"
                        })

                except Exception as e:
                    logger.warning(
                        f"Forecast failed for {task['region_code']}-{task['metric']}: "
                        f"{type(e).__name__}: {e}"
                    )

        return results

    def _calculate_derived(self, data: pd.DataFrame) -> List[Dict]:
        """Calculate derived metrics for all forecast years."""
        derived: List[Dict] = []
        forecast_years = data[data["data_type"] == "forecast"]["year"].unique()

        for region_code in data["region_code"].unique():
            for year in forecast_years:
                try:
                    year_results = self.derived.calculate_all_derived(data, region_code, int(year))
                    derived.extend(year_results)
                except Exception as e:
                    logger.debug(f"Derived metrics failed {region_code}-{year}: {e}")

        return derived

    def _save_outputs(self, data: pd.DataFrame):
        """Save ITL2 outputs: CSVs, DuckDB (base + derived), metadata."""
        
        # Define which metrics go where
        # Base metrics (7 forecast + 1 calculated + optional NI-only jobs): stored in gold.itl2_forecast
        base_metrics = [
            "nominal_gva_mn_gbp", "gdhi_total_mn_gbp", "emp_total_jobs",
            "emp_total_jobs_ni",
            "population_total", "population_16_64",
            "employment_rate_pct", "unemployment_rate_pct",
            "gdhi_per_head_gbp"  # Calculated, but stored with base
        ]
        # Derived metrics (2): stored in gold.itl2_derived
        derived_metrics = ["productivity_gbp_per_job", "income_per_worker_gbp"]
        
        # Split data
        base_data = data[data["metric"].isin(base_metrics)].copy()
        derived_data = data[data["metric"].isin(derived_metrics)].copy()
        
        # Long format (all data)
        long_path = self.config.output_dir / "itl2_forecast_long.csv"
        data.to_csv(long_path, index=False)
        logger.info(f"\n✓ Long format saved: {long_path}")

        # Wide format (values only)
        wide = data.pivot_table(
            index=["region", "region_code", "metric"],
            columns="year",
            values="value",
            aggfunc="first"
        ).reset_index()
        wide.columns.name = None
        wide_path = self.config.output_dir / "itl2_forecast_wide.csv"
        wide.to_csv(wide_path, index=False)
        logger.info(f"✓ Wide format saved: {wide_path}")
        
        # Derived metrics CSV
        if not derived_data.empty:
            derived_path = self.config.output_dir / "itl2_derived.csv"
            derived_data.to_csv(derived_path, index=False)
            logger.info(f"✓ Derived metrics saved: {derived_path}")

        # Confidence intervals
        if "ci_lower" in data.columns and "ci_upper" in data.columns:
            ci_data = data[["region_code", "metric", "year", "value", "ci_lower", "ci_upper"]].copy()
            ci_data = ci_data.dropna(subset=["ci_lower", "ci_upper"])
            if not ci_data.empty:
                ci_data["ci_width"] = ci_data["ci_upper"] - ci_data["ci_lower"]
                ci_path = self.config.output_dir / "itl2_confidence_intervals.csv"
                ci_data.to_csv(ci_path, index=False)
                logger.info(f"✓ Confidence intervals saved: {ci_path}")

        # Metadata
        forecast_data = data[data["data_type"] == "forecast"]
        metadata = {
            "run_timestamp": datetime.now().isoformat(),
            "version": "itl2_aligned_with_itl1_v3.5",
            "level": "ITL2",
            "config": {
                "target_year": self.config.target_year,
                "var_enabled": self.config.use_var_systems,
                "min_history_years": self.config.min_history_years,
                "reconciled_to_itl1": self.config.use_macro_anchoring
            },
            "data_summary": {
                "regions": int(data["region_code"].nunique()),
                "metrics": int(data["metric"].nunique()),
                "total_obs": int(len(data)),
                "forecast_obs": int(len(forecast_data)),
                "base_metrics": base_metrics,
                "derived_metrics": derived_metrics
            },
            "model_usage": forecast_data["method"].value_counts().to_dict()
            if "method" in forecast_data.columns else {}
        }

        if hasattr(data, "attrs") and "reconciliation_log" in data.attrs:
            recon = data.attrs["reconciliation_log"]
            if recon:
                metadata["reconciliation_summary"] = {
                    "adjustments": len(recon),
                    "avg_scale_factor": float(np.mean([r["scale_factor"] for r in recon]))
                }

        metadata_path = self.config.output_dir / "itl2_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"✓ Metadata saved: {metadata_path}")

        # DuckDB - split into base and derived tables
        if HAVE_DUCKDB:
            try:
                def prepare_for_duckdb(df: pd.DataFrame) -> pd.DataFrame:
                    """Prepare dataframe for DuckDB storage."""
                    df_copy = df.copy()
                    if "metric" in df_copy.columns:
                        df_copy["metric_id"] = df_copy["metric"]
                    if "year" in df_copy.columns:
                        df_copy["period"] = df_copy["year"]
                    if "region" in df_copy.columns:
                        df_copy["region_name"] = df_copy["region"]
                    
                    df_copy["region_level"] = "ITL2"
                    df_copy["forecast_run_date"] = datetime.now().date()
                    df_copy["forecast_version"] = "aligned_itl1_v3.5"
                    
                    metric_to_unit = {
                        "nominal_gva_mn_gbp": "GBP_m",
                        "gdhi_total_mn_gbp": "GBP_m",
                        "population_total": "persons",
                        "population_16_64": "persons",
                        "emp_total_jobs": "jobs",
                        "emp_total_jobs_ni": "jobs",
                        "employment_rate_pct": "percent",
                        "unemployment_rate_pct": "percent",
                        "productivity_gbp_per_job": "GBP",
                        "income_per_worker_gbp": "GBP",
                        "gdhi_per_head_gbp": "GBP"
                    }
                    if "unit" not in df_copy.columns:
                        df_copy["unit"] = df_copy["metric_id"].map(metric_to_unit).fillna("unknown")
                    if "freq" not in df_copy.columns:
                        df_copy["freq"] = "A"
                    
                    cols = [
                        "region_code", "region_name", "region_level", "metric_id",
                        "period", "value", "unit", "freq", "data_type",
                        "ci_lower", "ci_upper", "forecast_run_date", "forecast_version"
                    ]
                    return df_copy[[c for c in cols if c in df_copy.columns]].reset_index(drop=True)
                
                con = duckdb.connect(str(self.config.duckdb_path))
                con.execute("CREATE SCHEMA IF NOT EXISTS gold")
                
                # Save base table (7 forecast + gdhi_per_head)
                base_flat = prepare_for_duckdb(base_data)
                con.register("base_df", base_flat)
                con.execute("""
                    CREATE OR REPLACE TABLE gold.itl2_forecast AS
                    SELECT * FROM base_df
                """)
                con.execute("""
                    CREATE OR REPLACE VIEW gold.itl2_forecast_only AS
                    SELECT * FROM gold.itl2_forecast
                    WHERE data_type = 'forecast'
                """)
                logger.info(f"✓ DuckDB: gold.itl2_forecast ({len(base_flat)} rows)")
                
                # Save derived table (productivity, income_per_worker only)
                if not derived_data.empty:
                    derived_flat = prepare_for_duckdb(derived_data)
                    con.register("derived_df", derived_flat)
                    con.execute("""
                        CREATE OR REPLACE TABLE gold.itl2_derived AS
                        SELECT * FROM derived_df
                    """)
                    logger.info(f"✓ DuckDB: gold.itl2_derived ({len(derived_flat)} rows)")
                
                con.close()

            except Exception as e:
                logger.warning(f"DuckDB save failed: {e}")

        # Growth dampening summary
        if self._dampening_summaries:
            dampening_df = pd.DataFrame(self._dampening_summaries)

            logger.info("\n" + "=" * 70)
            logger.info(" GROWTH DAMPENING SUMMARY")
            logger.info("=" * 70)

            top_dampened = dampening_df.sort_values("cagr_reduction", ascending=False).head(10)
            logger.info("\nTop 10 most dampened (CAGR reduction):")
            for _, row in top_dampened.iterrows():
                logger.info(
                    f"  {row['region_code']} {row['metric']}: "
                    f"{row['original_cagr'] * 100:.1f}% → {row['dampened_cagr'] * 100:.1f}% "
                    f"(-{row['cagr_reduction'] * 100:.1f} pp)"
                )

            dampening_path = self.config.output_dir / "itl2_dampening_summary.csv"
            dampening_df.to_csv(dampening_path, index=False)
            logger.info(f"\n  Full dampening summary saved: {dampening_path}")

        # Final summary
        forecast_data = data[data["data_type"] == "forecast"]
        logger.info("\n" + "=" * 70)
        logger.info(" ✅ ITL2 FORECAST COMPLETE (ALIGNED WITH ITL1 V3.5)")
        logger.info("=" * 70)
        logger.info(f"📊 Regions:       {data['region_code'].nunique()}")
        logger.info(f"📊 Base metrics:  {[m for m in base_metrics if m in data['metric'].unique()]}")
        logger.info(f"📊 Derived:       {[m for m in derived_metrics if m in data['metric'].unique()]}")
        logger.info(f"📊 Total rows:    {len(data)}")
        logger.info(f"📊 Forecast rows: {len(forecast_data)}")
        if "method" in forecast_data.columns:
            logger.info("\n📈 Model usage:")
            for method, count in forecast_data["method"].value_counts().head(10).items():
                pct = 100 * count / len(forecast_data)
                logger.info(f"    {method}: {count} ({pct:.1f}%)")


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Run ITL2 forecast pipeline (aligned with ITL1 V3.5)."""
    try:
        config = ForecastConfigITL2()
        forecaster = InstitutionalForecasterITL2V4(config)
        results = forecaster.run()
        return results
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()