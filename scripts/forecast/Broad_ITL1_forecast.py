#!/usr/bin/env python3
"""
Region IQ - ITL1 Regional Forecasting Engine V3.5
==================================================

V3.5: Aligned with Macro V3.5 architecture
- Base metrics forecast to gold.itl1_forecast
- Derived metrics calculated separately to gold.itl1_derived
- population_16_64 added as base metric
- Blended VAR/univariate approach

Base Metrics (7 forecast + 1 calculated = 8 in gold.itl1_forecast):
  - nominal_gva_mn_gbp
  - gdhi_total_mn_gbp
  - emp_total_jobs
  - population_total
  - population_16_64
  - employment_rate_pct
  - unemployment_rate_pct
  - gdhi_per_head_gbp (calculated from reconciled totals, stored in base)

Derived Metrics (2, separate table gold.itl1_derived):
  - productivity_gbp_per_job = GVA / jobs × 1e6
  - income_per_worker_gbp = GDHI / jobs × 1e6

Outputs:
  - data/forecast/itl1_forecast_long.csv (base + gdhi_per_head)
  - data/forecast/itl1_derived.csv (productivity, income_per_worker only)
  - gold.itl1_forecast (base + gdhi_per_head)
  - gold.itl1_derived (productivity, income_per_worker only)

Author: Region IQ
Version: 3.5
License: Proprietary - Core IP
"""

import warnings
warnings.filterwarnings("ignore")

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Sequence

import json
import hashlib
import pickle

import numpy as np
np.random.seed(42)
import pandas as pd

# ===============================
# Logging
# ===============================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("itl1_forecast_v3.5.log"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger("ITL1_V3.5")

# ===============================
# Optional dependencies
# ===============================

try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.vector_ar.vecm import VECM, coint_johansen
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    from scipy import stats
    from scipy.stats import jarque_bera

    HAVE_STATSMODELS = True
except Exception as e:
    logger.warning(f"Statistical packages not fully available: {e}")
    HAVE_STATSMODELS = False

try:
    import duckdb
    HAVE_DUCKDB = True
except Exception:
    HAVE_DUCKDB = False


# ===============================
# V3.5 GROWTH CAPS - BASE METRICS ONLY
# ===============================

GROWTH_CAPS = {
    'nominal_gva_mn_gbp': {'min': -0.05, 'max': 0.06},
    'gdhi_total_mn_gbp': {'min': -0.05, 'max': 0.06},
    'emp_total_jobs': {'min': -0.03, 'max': 0.025},
    # NI-only employment jobs metric (not anchored to UK macro)
    'emp_total_jobs_ni': {'min': -0.03, 'max': 0.025},
    'population_total': {'min': -0.005, 'max': 0.015},
    'population_16_64': {'min': -0.01, 'max': 0.012},
    # Rates don't need growth caps - bounded 0-100
}

SANITY_CAPS = {
    'nominal_gva_mn_gbp': 1_500_000,  # £1.5tn per region max
    'gdhi_total_mn_gbp': 1_200_000,   # £1.2tn per region max
    'emp_total_jobs': 12_000_000,     # 12m jobs per region max
    'emp_total_jobs_ni': 12_000_000,  # same cap, NI-only metric
    'population_total': 15_000_000,   # 15m per region max
    'population_16_64': 10_000_000,   # 10m working age max
}


# ===============================
# Config
# ===============================

@dataclass
class ForecastConfig:
    """
    Central configuration for ITL1 V3.5 engine.
    """

    # Data paths
    silver_path: Path = Path("data/silver/itl1_unified_history.csv")
    duckdb_path: Path = Path("data/lake/warehouse.duckdb")
    use_duckdb: bool = False

    # Output
    output_dir: Path = Path("data/forecast")
    cache_dir: Path = Path("data/cache")

    # Forecast horizon
    target_year: int = 2050
    min_history_years: int = 10
    confidence_levels: List[float] = field(default_factory=lambda: [0.80, 0.95])

    # Models
    max_arima_order: int = 2
    max_var_lags: int = 3
    use_log_transform: Dict[str, bool] = None

    # VAR/VECM system spec
    use_var_systems: bool = True
    var_systems: Dict[str, List[str]] = field(default_factory=lambda: {
        "gva_employment": ["nominal_gva_mn_gbp", "emp_total_jobs"],
    })
    var_max_horizon: int = 15
    var_blend_decay: float = 0.25
    var_bootstrap_samples: int = 300

    # Top-down reconciliation
    use_macro_anchoring: bool = True
    reconciliation_method: str = "proportional"

    # ADDITIVE metrics only – reconciled to UK totals
    macro_anchor_map: Dict[str, str] = field(default_factory=lambda: {
        "nominal_gva_mn_gbp": "uk_nominal_gva_mn_gbp",
        "gdhi_total_mn_gbp": "uk_gdhi_total_mn_gbp",
        "emp_total_jobs": "uk_emp_total_jobs",
        "population_total": "uk_population_total",
        "population_16_64": "uk_population_16_64",
    })

    # Base metric definitions (8 metrics)
    metric_definitions: Dict[str, Dict] = None

    # Structural breaks
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
    enforce_growth_caps: bool = True
    growth_rate_cap_percentiles: Tuple[float, float] = (2, 98)

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.metric_definitions is None:
            # 7 BASE metrics only - gdhi_per_head_gbp is now derived
            self.metric_definitions = {
                # Additive stocks
                "population_total": {
                    "unit": "persons",
                    "transform": "log",
                    "monotonic": True,
                    "additive": True,
                    "allow_negative": False,
                    "type": "level",
                },
                "population_16_64": {
                    "unit": "persons",
                    "transform": "none",  # Working age can decline
                    "monotonic": False,
                    "additive": True,
                    "allow_negative": False,
                    "type": "level",
                },
                "nominal_gva_mn_gbp": {
                    "unit": "GBP_m",
                    "transform": "log",
                    "monotonic": False,
                    "additive": True,
                    "allow_negative": False,
                    "type": "level",
                },
                "gdhi_total_mn_gbp": {
                    "unit": "GBP_m",
                    "transform": "log",
                    "monotonic": False,
                    "additive": True,
                    "allow_negative": False,
                    "type": "level",
                },
                "emp_total_jobs": {
                    "unit": "jobs",
                    "transform": "log",
                    "monotonic": False,
                    "additive": True,
                    "allow_negative": False,
                    "type": "level",
                },
                # NI-only: employee jobs (BRESHEADLGD). Not reconciled to UK macro.
                "emp_total_jobs_ni": {
                    "unit": "jobs",
                    "transform": "log",
                    "monotonic": False,
                    "additive": True,
                    "allow_negative": False,
                    "type": "level",
                },
                # Rate metrics - PRIMARY from NOMIS
                "employment_rate_pct": {
                    "unit": "percent",
                    "transform": "none",
                    "monotonic": False,
                    "additive": False,
                    "primary_ratio": True,
                    "bounds": (0.0, 100.0),
                    "allow_negative": False,
                    "type": "rate",
                },
                "unemployment_rate_pct": {
                    "unit": "percent",
                    "transform": "none",
                    "monotonic": False,
                    "additive": False,
                    "primary_ratio": True,
                    "bounds": (0.0, 100.0),
                    "allow_negative": False,
                    "type": "rate",
                },
            }

        # Log transform flags
        if self.use_log_transform is None:
            self.use_log_transform = {
                metric: info.get("transform", "none") == "log"
                for metric, info in self.metric_definitions.items()
            }

        # Structural breaks
        if self.structural_breaks is None:
            self.structural_breaks = [
                {"year": 2008, "name": "Financial Crisis", "type": "level"},
                {"year": 2009, "name": "Crisis Recovery", "type": "trend"},
                {"year": 2016, "name": "Brexit Vote", "type": "trend"},
                {"year": 2020, "name": "COVID-19", "type": "level"},
                {"year": 2021, "name": "COVID Recovery", "type": "trend"},
            ]

        self._validate()

    def _validate(self):
        if not self.use_duckdb and not self.silver_path.exists():
            raise FileNotFoundError(f"Silver data not found: {self.silver_path}")
        if self.use_duckdb and not self.duckdb_path.exists():
            raise FileNotFoundError(f"DuckDB not found: {self.duckdb_path}")
        assert self.target_year > 2024, "Target year must be > current/last historical year"
        logger.info("ITL1 V3.5 configuration validated")


# ===============================
# Macro Anchor Manager
# ===============================

class MacroAnchorManager:
    """
    Loads UK macro forecasts for top-down reconciliation of additive metrics.
    """

    def __init__(self, duckdb_path: Path):
        self.duckdb_path = duckdb_path
        self.anchors = self._load_anchors()

    def _load_anchors(self) -> pd.DataFrame:
        if not HAVE_DUCKDB or not self.duckdb_path.exists():
            logger.warning("DuckDB unavailable - cannot load macro anchors.")
            return pd.DataFrame()

        try:
            con = duckdb.connect(str(self.duckdb_path), read_only=True)
            anchors = con.execute(
                """
                SELECT metric_id, period, value, data_type
                FROM gold.uk_macro_forecast
                WHERE data_type = 'forecast'
                """
            ).fetchdf()
            con.close()

            if anchors.empty:
                logger.warning("No UK macro forecasts found in gold.uk_macro_forecast.")
                return pd.DataFrame()

            if "year" not in anchors.columns:
                anchors["year"] = pd.to_numeric(anchors["period"], errors="coerce").astype(int)

            logger.info(
                f"✓ Loaded UK macro anchors: {anchors['metric_id'].nunique()} metrics, "
                f"{int(anchors['year'].min())}-{int(anchors['year'].max())}"
            )
            return anchors
        except Exception as e:
            logger.error(f"Failed to load macro anchors: {e}")
            return pd.DataFrame()

    def has_anchors(self) -> bool:
        return not self.anchors.empty

    def get_uk_value(self, metric_id: str, year: int) -> Optional[float]:
        if self.anchors.empty:
            return None
        sel = self.anchors[
            (self.anchors["metric_id"] == metric_id)
            & (self.anchors["year"] == int(year))
        ]
        if sel.empty:
            return None
        return float(sel["value"].iloc[0])


# ===============================
# Top-Down Reconciler
# ===============================

class TopDownReconciler:
    """
    Reconciles ITL1 additive metrics to UK macro totals.
    """

    def __init__(self, config: ForecastConfig, macro_manager: MacroAnchorManager):
        self.config = config
        self.macro = macro_manager

    def reconcile(self, data: pd.DataFrame, sanity_caps: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        if not self.macro.has_anchors():
            logger.warning("No macro anchors available – skipping reconciliation.")
            return data

        logger.info("=" * 70)
        logger.info("TOP-DOWN RECONCILIATION")
        logger.info("=" * 70)

        reconciliation_log = []

        forecast_mask = data["data_type"] == "forecast"
        forecast_data = data[forecast_mask].copy()
        forecast_years = sorted(forecast_data["year"].unique())

        # STEP 1: reconcile additive metrics
        for metric in forecast_data["metric"].unique():
            uk_metric_id = self.config.macro_anchor_map.get(metric)
            if not uk_metric_id:
                continue

            logger.info(f"  Reconciling {metric} → UK anchor {uk_metric_id}")

            for year in forecast_years:
                year = int(year)
                uk_value = self.macro.get_uk_value(uk_metric_id, year)
                if uk_value is None:
                    continue

                mask = (
                    (data["metric"] == metric)
                    & (data["year"] == year)
                    & (data["data_type"] == "forecast")
                )
                if not mask.any():
                    continue

                # If sanity caps are in play, keep capped regions fixed and scale the rest.
                cap = sanity_caps.get(metric) if sanity_caps else None

                regional_sum_before = data.loc[mask, "value"].sum()
                if regional_sum_before <= 0:
                    logger.warning(f"    {metric} {year}: regional sum ≤ 0 – skipping.")
                    continue

                if cap is not None:
                    fixed_mask = mask & (data["value"] >= cap)
                    free_mask = mask & ~fixed_mask

                    fixed_sum = data.loc[fixed_mask, "value"].sum()
                    free_sum = data.loc[free_mask, "value"].sum()

                    # If everything is capped or the cap already exceeds target, skip.
                    remaining_target = uk_value - fixed_sum
                    if free_sum <= 0 or remaining_target <= 0:
                        logger.warning(
                            f"    {metric} {year}: cap-aware reconcile skipped "
                            f"(fixed_sum={fixed_sum:,.0f}, target={uk_value:,.0f}, free_sum={free_sum:,.0f})"
                        )
                        continue

                    scale_factor = remaining_target / free_sum

                    data.loc[free_mask, "value"] *= scale_factor
                    if "ci_lower" in data.columns:
                        data.loc[free_mask, "ci_lower"] *= scale_factor
                    if "ci_upper" in data.columns:
                        data.loc[free_mask, "ci_upper"] *= scale_factor
                else:
                    scale_factor = uk_value / regional_sum_before
                    data.loc[mask, "value"] *= scale_factor
                    if "ci_lower" in data.columns:
                        data.loc[mask, "ci_lower"] *= scale_factor
                    if "ci_upper" in data.columns:
                        data.loc[mask, "ci_upper"] *= scale_factor

                regional_sum_after = data.loc[mask, "value"].sum()
                deviation = (
                    abs(regional_sum_after - uk_value) / uk_value
                    if uk_value > 0
                    else 0.0
                )

                reconciliation_log.append(
                    {
                        "year": year,
                        "metric": metric,
                        "uk_value": uk_value,
                        "regional_sum_before": regional_sum_before,
                        "regional_sum_after": regional_sum_after,
                        "scale_factor": scale_factor,
                        "deviation_pct": deviation * 100,
                    }
                )

                if year in [2025, 2030, 2040, 2050]:
                    logger.info(
                        f"    {year}: SF={scale_factor:.4f} | UK={uk_value:,.0f} | "
                        f"Regional: {regional_sum_before:,.0f}→{regional_sum_after:,.0f}"
                    )

        data.attrs["reconciliation_log"] = reconciliation_log
        logger.info(f"  ✓ Additive metrics: {len(reconciliation_log)} adjustments")

        return data


# ===============================
# Data Management
# ===============================

class DataManagerV35:
    """
    Data management for unified ITL1 silver schema.
    """

    def __init__(self, config: ForecastConfig):
        self.config = config

    def load_all_data(self) -> pd.DataFrame:
        cache_key = self._get_cache_key()
        if self.config.cache_enabled and self._cache_exists(cache_key):
            logger.info("Loading ITL1 silver data from cache.")
            return self._load_from_cache(cache_key)

        if self.config.use_duckdb:
            df = self._load_from_duckdb()
        else:
            df = self._load_from_csv()

        df = self._standardize_columns(df)
        df = self._augment_with_bottomup_ni_jobs(df)
        df = self._handle_outliers(df)

        if self.config.cache_enabled:
            self._save_to_cache(df, cache_key)

        logger.info(
            f"Loaded {len(df)} rows, "
            f"{df['region_code'].nunique()} regions, "
            f"{df['metric'].nunique()} metrics"
        )
        return df

    def _augment_with_bottomup_ni_jobs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add NI-only job metric history (`emp_total_jobs_ni`) to ITL1 history without
        touching NOMIS ITL1 ingests for other metrics.

        Source of truth is bottom-up transform output: `silver.itl1_unified_bottomup`,
        which is produced from NI LGD (LAD-equivalent) series.
        """
        target_metric = "emp_total_jobs_ni"
        ni_itl1_code = "N92000002"

        if target_metric in set(df["metric"].unique()):
            return df

        if not HAVE_DUCKDB:
            logger.warning("DuckDB unavailable; cannot augment ITL1 with emp_total_jobs_ni.")
            return df

        try:
            con = duckdb.connect(str(self.config.duckdb_path), read_only=True)
            bottomup = con.execute("""
                SELECT
                    region_code,
                    region_name as region,
                    metric_id as metric,
                    period as year,
                    value,
                    'historical' as data_type
                FROM silver.itl1_unified_bottomup
                WHERE metric_id = 'emp_total_jobs_ni'
                  AND region_code = 'N92000002'
            """).fetchdf()
            con.close()
        except Exception as e:
            logger.warning(f"Failed to load bottom-up NI ITL1 jobs: {e}")
            return df

        if bottomup is None or bottomup.empty:
            logger.warning("No bottom-up NI ITL1 jobs found (emp_total_jobs_ni).")
            return df

        bottomup["year"] = pd.to_numeric(bottomup["year"], errors="coerce").astype(int)
        bottomup["value"] = pd.to_numeric(bottomup["value"], errors="coerce")
        bottomup = bottomup.dropna(subset=["year", "value"])

        out = pd.concat([df, bottomup], ignore_index=True)
        logger.info(f"✓ Augmented ITL1 with NI jobs metric: {target_metric} ({len(bottomup)} rows) for {ni_itl1_code}")
        return out

    def _load_from_csv(self) -> pd.DataFrame:
        logger.info(f"Loading ITL1 silver from CSV: {self.config.silver_path}")
        df = pd.read_csv(self.config.silver_path)
        return df

    def _load_from_duckdb(self) -> pd.DataFrame:
        logger.info(f"Loading ITL1 silver from DuckDB: {self.config.duckdb_path}")
        try:
            con = duckdb.connect(str(self.config.duckdb_path), read_only=True)
            df = con.execute("SELECT * FROM silver.itl1_history").fetchdf()
            con.close()
            return df
        except Exception as e:
            logger.error(f"DuckDB load failed, falling back to CSV: {e}")
            return self._load_from_csv()

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {
            "metric_id": "metric",
            "period": "year",
            "region_name": "region",
        }

        for old, new in rename_map.items():
            if old in df.columns and new not in df.columns:
                df[new] = df[old]

        required = ["region_code", "region", "metric", "year", "value"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column in silver ITL1: {col}")

        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        if "data_type" not in df.columns:
            df["data_type"] = "historical"

        df = df.dropna(subset=["year", "value"])

        if self.config.enforce_non_negative:
            positive_metrics = [
                m
                for m, info in self.config.metric_definitions.items()
                if not info.get("allow_negative", False)
            ]
            pos_mask = df["metric"].isin(positive_metrics)
            df = df[~pos_mask | (df["value"] >= 0)]

        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        for (region, metric), group in df.groupby(["region_code", "metric"]):
            if len(group) < 5:
                continue

            values = group["value"].values
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            if iqr == 0:
                continue

            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = (values < lower) | (values > upper)

            if outliers.any():
                logger.debug(f"Winsorising {outliers.sum()} outliers in {region}-{metric}")
                df.loc[group[outliers].index, "value"] = np.clip(
                    values[outliers], lower, upper
                )

        return df

    def _get_cache_key(self) -> str:
        hasher = hashlib.md5()
        path = self.config.duckdb_path if self.config.use_duckdb else self.config.silver_path
        if path.exists():
            hasher.update(str(path.stat().st_mtime).encode())
        hasher.update(b"v3.5")
        return hasher.hexdigest()

    def _cache_exists(self, key: str) -> bool:
        return (self.config.cache_dir / f"itl1_data_{key}.pkl").exists()

    def _load_from_cache(self, key: str) -> pd.DataFrame:
        with open(self.config.cache_dir / f"itl1_data_{key}.pkl", "rb") as f:
            return pickle.load(f)

    def _save_to_cache(self, df: pd.DataFrame, key: str):
        with open(self.config.cache_dir / f"itl1_data_{key}.pkl", "wb") as f:
            pickle.dump(df, f)


# ===============================
# VAR / VECM system forecaster
# ===============================

class VARSystemForecaster:
    """
    Multivariate system forecasting for cross-metric coherence.
    """

    def __init__(self, config: ForecastConfig):
        self.config = config

    def forecast_system(
        self,
        data: pd.DataFrame,
        region_code: str,
        metrics: List[str],
        horizon: int,
        structural_breaks: Optional[Sequence[Dict]] = None,
    ) -> Optional[Dict]:
        if not HAVE_STATSMODELS:
            return None

        try:
            series_dict = {}
            for metric in metrics:
                subset = (
                    data[(data["region_code"] == region_code) & (data["metric"] == metric)]
                    .sort_values("year")
                )
                if subset.empty:
                    return None

                vals = subset["value"].values.astype(float)
                if self.config.use_log_transform.get(metric, False) and (vals > 0).all():
                    vals = np.log(vals)
                series_dict[metric] = pd.Series(vals, index=subset["year"].astype(int))

            df_sys = pd.DataFrame(series_dict).dropna()
            if len(df_sys) < self.config.min_history_years:
                return None

            coint_info = self._test_cointegration(df_sys)

            exog = (
                self._prep_breaks(df_sys.index, structural_breaks)
                if structural_breaks
                else None
            )

            if coint_info["cointegrated"] and coint_info["rank"] > 0:
                logger.info(f"    Using VECM (rank={coint_info['rank']})")
                model = VECM(
                    df_sys,
                    k_ar_diff=min(self.config.max_var_lags, 3),
                    coint_rank=coint_info["rank"],
                    deterministic="ci",
                    exog=exog,
                )
                fitted = model.fit()
                exog_fc = self._extend_exog(exog, df_sys.index, horizon) if exog is not None else None
                fc_vals = fitted.predict(steps=horizon, exog_fc=exog_fc)
                method_name = f"VECM(r={coint_info['rank']})"
            else:
                logger.info(f"    Using VAR")
                model = VAR(df_sys, exog=exog)
                lag_sel = model.select_order(maxlags=self.config.max_var_lags)
                try:
                    opt_lags = lag_sel.selected_orders["aic"]
                except Exception:
                    opt_lags = lag_sel.aic
                opt_lags = max(1, min(opt_lags, self.config.max_var_lags))

                fitted = model.fit(maxlags=opt_lags)
                exog_fc = self._extend_exog(exog, df_sys.index, horizon) if exog is not None else None
                fc_vals = fitted.forecast(
                    df_sys.values[-fitted.k_ar :],
                    steps=horizon,
                    exog_future=exog_fc,
                )
                method_name = f"VAR({opt_lags})"

            ci_low, ci_up = self._bootstrap_ci(fitted, df_sys, horizon, exog, exog_fc)

            last_year = int(df_sys.index[-1])
            years = list(range(last_year + 1, last_year + horizon + 1))

            results = {}
            for i, metric in enumerate(metrics):
                vals = fc_vals[:, i]
                low = ci_low[:, i]
                up = ci_up[:, i]

                if self.config.use_log_transform.get(metric, False):
                    vals, low, up = np.exp(vals), np.exp(low), np.exp(up)

                results[metric] = {
                    "method": method_name,
                    "values": vals,
                    "years": years,
                    "ci_lower": low,
                    "ci_upper": up,
                    "system_metrics": metrics,
                    "cointegrated": coint_info["cointegrated"],
                    "aic": getattr(fitted, "aic", None),
                }

            return results

        except Exception as e:
            logger.debug(f"VAR/VECM system failed for {region_code}-{metrics}: {e}")
            return None

    def _test_cointegration(self, data: pd.DataFrame) -> Dict:
        if len(data) < 30:
            return {"cointegrated": False, "rank": 0}
        try:
            res = coint_johansen(data, det_order=0, k_ar_diff=2)
            rank = int(sum(res.lr1 > res.cvt[:, 1]))
            return {"cointegrated": rank > 0, "rank": rank}
        except Exception:
            return {"cointegrated": False, "rank": 0}

    def _bootstrap_ci(
        self,
        fitted,
        data: pd.DataFrame,
        horizon: int,
        exog: Optional[np.ndarray],
        exog_fc: Optional[np.ndarray],
    ):
        n_vars = data.shape[1]
        residuals = getattr(fitted, "resid", None)
        if residuals is None:
            residuals = data.values[fitted.k_ar :] - fitted.fittedvalues

        boot_fcs = []
        for _ in range(self.config.var_bootstrap_samples):
            try:
                boot_resid = residuals[
                    np.random.choice(len(residuals), len(residuals), replace=True)
                ]
                boot_df = pd.DataFrame(
                    fitted.fittedvalues + boot_resid,
                    columns=data.columns,
                    index=data.index[-len(fitted.fittedvalues) :],
                )

                if hasattr(fitted, "coint_rank"):
                    boot_model = VECM(
                        boot_df,
                        k_ar_diff=fitted.k_ar_diff,
                        coint_rank=fitted.coint_rank,
                        deterministic="ci",
                        exog=exog,
                    ).fit()
                    boot_fc = boot_model.predict(steps=horizon, exog_fc=exog_fc)
                else:
                    boot_model = VAR(boot_df, exog=exog).fit(maxlags=fitted.k_ar)
                    boot_fc = boot_model.forecast(
                        boot_df.values[-boot_model.k_ar :],
                        steps=horizon,
                        exog_future=exog_fc,
                    )

                boot_fcs.append(boot_fc)
            except Exception:
                continue

        if len(boot_fcs) < 50:
            resid_std = np.std(residuals, axis=0)
            fc_std = resid_std * np.sqrt(np.arange(1, horizon + 1))[:, None]
            return -1.96 * fc_std, 1.96 * fc_std

        boot_arr = np.array(boot_fcs)
        ci_low = np.percentile(boot_arr, 2.5, axis=0)
        ci_up = np.percentile(boot_arr, 97.5, axis=0)
        return ci_low, ci_up

    def _prep_breaks(self, index: Sequence[int], breaks: Sequence[Dict]) -> Optional[np.ndarray]:
        if not breaks:
            return None
        idx = np.array(index, dtype=int)
        dummies = []
        for b in breaks:
            year = int(b["year"])
            btype = b.get("type", "level")
            if btype == "level":
                dummies.append((idx >= year).astype(int))
            else:
                dummies.append(np.maximum(0, idx - year))
        return np.column_stack(dummies) if dummies else None

    def _extend_exog(self, exog: np.ndarray, index: Sequence[int], horizon: int) -> Optional[np.ndarray]:
        if exog is None:
            return None
        if exog.ndim == 1:
            exog = exog.reshape(-1, 1)

        last_year = int(index[-1])
        future_years = np.arange(last_year + 1, last_year + horizon + 1)
        extended = []
        for _y in future_years:
            row = []
            for col in range(exog.shape[1]):
                col_data = exog[:, col]
                if np.all(np.isin(col_data, [0, 1])):
                    row.append(col_data[-1])
                else:
                    inc = col_data[-1] - col_data[-2] if len(col_data) > 1 else 1.0
                    row.append(col_data[-1] + inc)
            extended.append(row)
        return np.array(extended)


# ===============================
# Advanced forecasting (univariate)
# ===============================

class AdvancedForecastingV35:
    """
    Univariate forecasting with ARIMA + ETS + Linear ensemble.
    """

    def __init__(self, config: ForecastConfig):
        self.config = config
        self.var_forecaster = VARSystemForecaster(config) if config.use_var_systems else None

    def forecast_univariate(
        self,
        series: pd.Series,
        horizon: int,
        structural_breaks: Optional[Sequence[Dict]],
        metric_info: Dict,
    ) -> Dict:
        # Rate metrics behave badly under unconstrained linear trend extrapolation
        # (can drift negative and then get hard-clipped to 0, producing unrealistic 0%).
        # For rates we use a simple mean-reverting (OU-style) forecast instead.
        if metric_info.get("type") == "rate":
            fc = self._fit_mean_revert_rate(series, horizon)
            fc = self._apply_constraints(fc, series, metric_info)
            return fc

        models = []

        arima_res = self._fit_arima_with_breaks(series, horizon, structural_breaks)
        if arima_res:
            models.append(arima_res)

        if HAVE_STATSMODELS and len(series) > 20:
            ets_res = self._fit_ets_auto(series, horizon)
            if ets_res:
                models.append(ets_res)

        lin_res = self._fit_linear(series, horizon)
        if lin_res:
            models.append(lin_res)

        if not models:
            fc = self._fallback_forecast(series, horizon)
        elif len(models) == 1:
            fc = models[0]
        else:
            cv_errors = self._true_cross_validation(series, models)
            weights = self._calculate_cv_weights(cv_errors)
            fc = self._combine_forecasts(models, weights, series)

        fc = self._apply_constraints(fc, series, metric_info)
        return fc

    def _fit_mean_revert_rate(self, series: pd.Series, horizon: int) -> Dict:
        """
        Mean-reverting rate forecast (Ornstein-Uhlenbeck style).

        This avoids long-horizon drift-to-zero artifacts that can occur when the
        linear fallback extrapolates a downward slope and constraints then clip.
        """
        s = series.dropna()
        last_year = int(s.index[-1])
        years = list(range(last_year + 1, last_year + horizon + 1))

        # Use a recent-window mean to avoid pulling toward long-ago regimes
        window = min(10, len(s))
        mu = float(s.iloc[-window:].mean())
        x0 = float(s.iloc[-1])

        theta = 0.12  # annual reversion speed (moderate)
        t = np.arange(1, horizon + 1)
        values = mu + (x0 - mu) * np.exp(-theta * t)

        # CI: steady-state OU variance proxy (same idea as LAD mean-revert)
        hist_std = float(s.std(ddof=1)) if len(s) > 1 else 0.0
        if not np.isfinite(hist_std) or hist_std <= 0:
            hist_std = abs(mu) * 0.05 if np.isfinite(mu) else 0.5

        steady_std = hist_std / np.sqrt(2 * theta)
        ci_width = 1.96 * steady_std
        ci_lower = values - ci_width
        ci_upper = values + ci_width

        return {
            "method": "MeanRevertRate",
            "values": values,
            "years": years,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    def _fit_arima_with_breaks(
        self,
        series: pd.Series,
        horizon: int,
        structural_breaks: Optional[Sequence[Dict]],
    ) -> Optional[Dict]:
        if not HAVE_STATSMODELS:
            return None

        try:
            exog = (
                self._prepare_break_dummies(series, structural_breaks)
                if structural_breaks
                else None
            )

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
                        aicc = fitted.aic + (2 * k * (k + 1)) / (n - k - 1)
                        if aicc < best_aic:
                            best_aic = aicc
                            best_model = fitted
                            best_order = (p, d, q)
                    except Exception:
                        continue

            if best_model is None:
                return None

            exog_fc = (
                self._extend_exog_properly(exog, series.index, horizon)
                if exog is not None
                else None
            )
            forecast_obj = best_model.get_forecast(steps=horizon, exog=exog_fc)
            fc_vals = forecast_obj.predicted_mean

            resid = pd.Series(best_model.resid)
            resid_tests = self._test_residuals(resid)

            if not resid_tests.get("normal", True):
                ci = self._bootstrap_confidence_intervals(
                    best_model, series, horizon, exog, exog_fc, best_order
                )
            else:
                ci = forecast_obj.conf_int(alpha=0.05)

            last_year = int(series.index[-1])
            years = list(range(last_year + 1, last_year + horizon + 1))

            return {
                "method": f"ARIMA{best_order}",
                "values": fc_vals.values,
                "years": years,
                "ci_lower": ci[:, 0] if hasattr(ci, "shape") else ci[0],
                "ci_upper": ci[:, 1] if hasattr(ci, "shape") else ci[1],
                "aic": best_aic,
                "order": best_order,
                "residual_tests": resid_tests,
                "fit_function": lambda s, h: self._arima_predict(s, h, best_order, structural_breaks),
            }

        except Exception as e:
            logger.debug(f"ARIMA failed ({series.name}): {e}")
            return None

    def _bootstrap_confidence_intervals(
        self,
        model,
        series: pd.Series,
        horizon: int,
        exog: Optional[np.ndarray],
        exog_fc: Optional[np.ndarray],
        order: Tuple[int, int, int],
    ) -> np.ndarray:
        residuals = model.resid
        fitted_vals = model.fittedvalues

        boot_fcs = []
        for _ in range(self.config.n_bootstrap):
            try:
                boot_resid = np.random.choice(residuals, size=len(residuals), replace=True)
                boot_series = pd.Series(
                    fitted_vals + boot_resid, index=series.index
                )
                boot_model = ARIMA(boot_series, order=order, exog=exog).fit(
                    method_kwargs={"warn_convergence": False}
                )
                boot_fc = boot_model.forecast(steps=horizon, exog=exog_fc)
                boot_fcs.append(boot_fc.values)
            except Exception:
                continue

        if len(boot_fcs) < 10:
            return model.get_forecast(steps=horizon, exog=exog_fc).conf_int(alpha=0.05)

        arr = np.array(boot_fcs)
        low = np.percentile(arr, 2.5, axis=0)
        up = np.percentile(arr, 97.5, axis=0)
        return np.column_stack([low, up])

    def _fit_ets_auto(self, series: pd.Series, horizon: int) -> Optional[Dict]:
        if not HAVE_STATSMODELS:
            return None
        try:
            best_model = None
            best_aic = np.inf
            best_label = None

            for trend in ["add", "mul", None]:
                for damped in [True, False]:
                    if trend is None and damped:
                        continue
                    try:
                        model = ExponentialSmoothing(
                            series,
                            trend=trend,
                            damped_trend=damped,
                            seasonal=None,
                        )
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_model = fitted
                            best_label = f"ETS({trend},{damped})"
                    except Exception:
                        continue

            if best_model is None:
                return None

            fc = best_model.forecast(steps=horizon)
            resid = series - best_model.fittedvalues
            sigma = resid.std(ddof=1)

            if len(series) < 30:
                t_val = stats.t.ppf(0.975, len(series) - 1)
            else:
                t_val = 1.96

            h = np.arange(1, horizon + 1)
            ci_width = t_val * sigma * np.sqrt(h)
            ci_lower = fc.values - ci_width
            ci_upper = fc.values + ci_width

            last_year = int(series.index[-1])
            years = list(range(last_year + 1, last_year + horizon + 1))

            return {
                "method": best_label,
                "values": fc.values,
                "years": years,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "aic": best_aic,
                "fit_function": lambda s, h: best_model.forecast(steps=h).values,
            }

        except Exception as e:
            logger.debug(f"ETS failed ({series.name}): {e}")
            return None

    def _fit_linear(self, series: pd.Series, horizon: int) -> Optional[Dict]:
        try:
            X = np.arange(len(series))
            y = series.values

            if HAVE_STATSMODELS:
                X_sm = sm.add_constant(X)
                model = sm.OLS(y, X_sm).fit()
                X_future = np.arange(len(series), len(series) + horizon)
                Xf_sm = sm.add_constant(X_future)
                preds = model.get_prediction(Xf_sm)
                fc = preds.predicted_mean
                ci = preds.conf_int(alpha=0.05)
                ci_lower, ci_upper = ci[:, 0], ci[:, 1]
                aic = model.aic
            else:
                coeffs = np.polyfit(X, y, 1)
                X_future = np.arange(len(series), len(series) + horizon)
                fc = np.polyval(coeffs, X_future)
                resid = y - np.polyval(coeffs, X)
                sigma = resid.std(ddof=1)
                h = np.arange(1, horizon + 1)
                ci_width = 1.96 * sigma * np.sqrt(h)
                ci_lower = fc - ci_width
                ci_upper = fc + ci_width
                aic = None

            last_year = int(series.index[-1])
            years = list(range(last_year + 1, last_year + horizon + 1))

            return {
                "method": "Linear",
                "values": fc,
                "years": years,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "aic": aic,
                "fit_function": lambda s, h: self._linear_predict(s, h),
            }
        except Exception as e:
            logger.debug(f"Linear forecast failed ({series.name}): {e}")
            return None

    def _true_cross_validation(self, series: pd.Series, models: List[Dict]) -> Dict[int, List[float]]:
        cv_errors: Dict[int, List[float]] = {i: [] for i in range(len(models))}
        min_train = max(self.config.cv_min_train_size, len(series) // 2)

        for train_end in range(min_train, len(series) - 1):
            train = series.iloc[:train_end]
            test_val = series.iloc[train_end]

            for i, m in enumerate(models):
                try:
                    fc = m["fit_function"](train, 1)[0]
                    if (abs(fc) + abs(test_val)) > 0:
                        err = abs(fc - test_val) / (abs(fc) + abs(test_val))
                    else:
                        err = 0.0
                    cv_errors[i].append(err)
                except Exception:
                    cv_errors[i].append(np.inf)

        return cv_errors

    def _calculate_cv_weights(self, cv_errors: Dict[int, List[float]]) -> np.ndarray:
        means = []
        for i in sorted(cv_errors.keys()):
            vals = [e for e in cv_errors[i] if e != np.inf]
            means.append(np.mean(vals) if vals else np.inf)

        if all(np.isinf(means)):
            return np.ones(len(means)) / len(means)

        weights = 1.0 / (np.array(means) + 1e-10)
        weights[np.isinf(weights)] = 0
        if weights.sum() <= 0:
            return np.ones(len(weights)) / len(weights)
        return weights / weights.sum()

    def _combine_forecasts(
        self, models: List[Dict], weights: np.ndarray, series: pd.Series
    ) -> Dict:
        horizon = len(models[0]["values"])
        combined = np.zeros(horizon)
        for m, w in zip(models, weights):
            combined += w * m["values"]

        cv_resid = []
        min_train = max(self.config.cv_min_train_size, len(series) // 2)
        for train_end in range(min_train, len(series)):
            train = series.iloc[:train_end]
            test_val = series.iloc[train_end]

            ensemble_fc = 0.0
            for m, w in zip(models, weights):
                try:
                    fc = m["fit_function"](train, 1)[0]
                    ensemble_fc += w * fc
                except Exception:
                    pass
            cv_resid.append(test_val - ensemble_fc)

        cv_resid = np.array(cv_resid)
        if len(cv_resid) > 5:
            resid_std = cv_resid.std(ddof=1)
            h = np.arange(1, horizon + 1)
            ci_width = 1.96 * resid_std * np.sqrt(h)
            ci_lower = combined - ci_width
            ci_upper = combined + ci_width
        else:
            ci_lower = np.zeros(horizon)
            ci_upper = np.zeros(horizon)
            for m, w in zip(models, weights):
                ci_lower += w * m["ci_lower"]
                ci_upper += w * m["ci_upper"]

        return {
            "method": f"Ensemble({len(models)} models)",
            "values": combined,
            "years": models[0]["years"],
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "weights": weights.tolist(),
            "component_methods": [m["method"] for m in models],
            "cv_residual_std": float(cv_resid.std(ddof=1)) if len(cv_resid) > 1 else None,
        }

    def _apply_constraints(
        self, forecast: Dict, historical: pd.Series, metric_info: Dict
    ) -> Dict:
        """Apply constraints with growth caps in level space."""
        metric_name = historical.name
        is_log = self.config.use_log_transform.get(metric_name, False)
        
        vals = forecast["values"].copy()
        ci_low = forecast["ci_lower"].copy()
        ci_up = forecast["ci_upper"].copy()

        # Convert to level space if needed
        if is_log:
            vals_level = np.exp(vals)
            ci_low_level = np.exp(ci_low)
            ci_up_level = np.exp(ci_up)
            last_hist_level = np.exp(historical.iloc[-1])
        else:
            vals_level = vals.copy()
            ci_low_level = ci_low.copy()
            ci_up_level = ci_up.copy()
            last_hist_level = historical.iloc[-1]

        # Non-negative
        if self.config.enforce_non_negative and not metric_info.get("allow_negative", False):
            vals_level = np.maximum(vals_level, 0.0)
            ci_low_level = np.maximum(ci_low_level, 0.0)

        # Rate bounds
        bounds = metric_info.get("bounds")
        if bounds is not None:
            lower_b, upper_b = bounds
            vals_level = np.clip(vals_level, lower_b, upper_b)
            ci_low_level = np.clip(ci_low_level, lower_b, upper_b)
            ci_up_level = np.clip(ci_up_level, lower_b, upper_b)

        # Monotonic (population_total only)
        if metric_info.get("monotonic", False) and self.config.enforce_monotonic_population:
            for i in range(1, len(vals_level)):
                vals_level[i] = max(vals_level[i], vals_level[i - 1])

        # Growth caps in level space
        if self.config.enforce_growth_caps and metric_info.get("type") != "rate":
            growth_cap = GROWTH_CAPS.get(metric_name)
            if growth_cap:
                min_g, max_g = growth_cap["min"], growth_cap["max"]
                capped = 0
                prev = last_hist_level
                
                for i in range(len(vals_level)):
                    if prev <= 0:
                        prev = vals_level[i]
                        continue
                    g = (vals_level[i] - prev) / prev
                    if g > max_g:
                        vals_level[i] = prev * (1 + max_g)
                        capped += 1
                    elif g < min_g:
                        vals_level[i] = prev * (1 + min_g)
                        capped += 1
                    prev = vals_level[i]
                
                if capped > 0:
                    logger.debug(f"    {metric_name}: Capped {capped} years")
                    # Recalculate CIs
                    hist_vol = historical.pct_change().dropna().std()
                    if np.isfinite(hist_vol) and hist_vol > 0:
                        for i in range(len(vals_level)):
                            ci_w = 1.96 * vals_level[i] * hist_vol * np.sqrt(i + 1)
                            ci_low_level[i] = max(0, vals_level[i] - ci_w)
                            ci_up_level[i] = vals_level[i] + ci_w

        # Convert back to log space if needed
        if is_log:
            vals_level = np.maximum(vals_level, 1e-10)
            ci_low_level = np.maximum(ci_low_level, 1e-10)
            ci_up_level = np.maximum(ci_up_level, 1e-10)
            forecast["values"] = np.log(vals_level)
            forecast["ci_lower"] = np.log(ci_low_level)
            forecast["ci_upper"] = np.log(ci_up_level)
        else:
            forecast["values"] = vals_level
            forecast["ci_lower"] = ci_low_level
            forecast["ci_upper"] = ci_up_level

        forecast["constraints_applied"] = True
        return forecast

    def _prepare_break_dummies(
        self, series: pd.Series, breaks: Optional[Sequence[Dict]]
    ) -> Optional[np.ndarray]:
        if not breaks:
            return None
        years = series.index.astype(int)
        dummies = []
        for b in breaks:
            year = int(b["year"])
            btype = b.get("type", "level")
            if btype == "level":
                dummies.append((years >= year).astype(int))
            else:
                dummies.append(np.maximum(0, years - year))
        return np.column_stack(dummies) if dummies else None

    def _extend_exog_properly(
        self, exog: np.ndarray, idx: Sequence[int], horizon: int
    ) -> Optional[np.ndarray]:
        if exog is None:
            return None
        if exog.ndim == 1:
            exog = exog.reshape(-1, 1)
        years = np.array(idx, dtype=int)
        last_year = int(years[-1])
        future_years = np.arange(last_year + 1, last_year + horizon + 1)
        extended = []
        for _y in future_years:
            row = []
            for col in range(exog.shape[1]):
                col_data = exog[:, col]
                if np.all(np.isin(col_data, [0, 1])):
                    row.append(col_data[-1])
                else:
                    inc = col_data[-1] - col_data[-2] if len(col_data) > 1 else 1.0
                    row.append(col_data[-1] + inc)
            extended.append(row)
        res = np.array(extended)
        return res

    def _test_residuals(self, residuals: pd.Series) -> Dict:
        results: Dict[str, float] = {}
        if not HAVE_STATSMODELS or len(residuals) < 10:
            return results

        r = residuals.dropna()
        try:
            lb = acorr_ljungbox(r, lags=min(10, len(r) // 3))
            results["ljung_box_p"] = float(lb["lb_pvalue"].min())
            results["no_autocorr"] = results["ljung_box_p"] > 0.05
        except Exception:
            pass

        try:
            jb_stat, jb_p = jarque_bera(r)
            results["jarque_bera_p"] = float(jb_p)
            results["normal"] = jb_p > 0.05
        except Exception:
            pass

        return results

    def _arima_predict(
        self,
        series: pd.Series,
        horizon: int,
        order: Tuple[int, int, int],
        breaks: Optional[Sequence[Dict]],
    ) -> np.ndarray:
        try:
            exog = (
                self._prepare_break_dummies(series, breaks)
                if breaks
                else None
            )
            model = ARIMA(series, order=order, exog=exog)
            fitted = model.fit(method_kwargs={"warn_convergence": False})
            exog_fc = (
                self._extend_exog_properly(exog, series.index, horizon)
                if exog is not None
                else None
            )
            fc = fitted.forecast(steps=horizon, exog=exog_fc)
            return fc.values
        except Exception:
            return np.repeat(series.iloc[-1], horizon)

    def _linear_predict(self, series: pd.Series, horizon: int) -> np.ndarray:
        try:
            X = np.arange(len(series))
            y = series.values
            coeffs = np.polyfit(X, y, 1)
            Xf = np.arange(len(series), len(series) + horizon)
            return np.polyval(coeffs, Xf)
        except Exception:
            return np.repeat(series.iloc[-1], horizon)

    def _fallback_forecast(self, series: pd.Series, horizon: int) -> Dict:
        last_val = float(series.iloc[-1])
        if len(series) > 5:
            recent_trend = (series.iloc[-1] - series.iloc[-5]) / 5.0
        else:
            recent_trend = 0.0
        vals = np.array(
            [max(last_val + recent_trend * (i + 1), 0.0) for i in range(horizon)]
        )
        if len(series) > 2:
            std = series.std(ddof=1)
            h = np.arange(1, horizon + 1)
            ci_width = 1.96 * std * np.sqrt(h)
        else:
            ci_width = vals * 0.1
        ci_lower = vals - ci_width
        ci_upper = vals + ci_width

        last_year = int(series.index[-1])
        years = list(range(last_year + 1, last_year + horizon + 1))

        return {
            "method": "Fallback",
            "values": vals,
            "years": years,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }


# ===============================
# Derived Metrics Calculator (Separate)
# ===============================

class DerivedMetricsCalculator:
    """
    Calculate derived metrics from base forecasts.
    Outputs to separate table: gold.itl1_derived
    """

    @staticmethod
    def calculate_with_uncertainty(
        numerator: Tuple[float, float, float],
        denominator: Tuple[float, float, float],
        operation: str = "divide",
        correlation: float = 0.0,
        n_simulations: int = 10000,
    ) -> Tuple[float, float, float]:
        num_val, num_low, num_high = numerator
        den_val, den_low, den_high = denominator

        # Handle NaN CIs (historical components in mixed-vintage scenarios)
        if math.isnan(num_low) or math.isnan(num_high):
            num_se = max(abs(num_val) * 0.02, 1e-10)
        else:
            num_se = (num_high - num_low) / (2 * 1.96)

        if math.isnan(den_low) or math.isnan(den_high):
            den_se = max(abs(den_val) * 0.02, 1e-10)
        else:
            den_se = (den_high - den_low) / (2 * 1.96)

        eps = 1e-10
        num_se = max(num_se, eps)
        den_se = max(den_se, eps)

        if operation == "divide" and abs(den_val) < eps:
            return 0.0, 0.0, 0.0

        try:
            if abs(correlation) > eps:
                mean = [num_val, den_val]
                cov = [
                    [num_se**2 + eps, correlation * num_se * den_se],
                    [correlation * num_se * den_se, den_se**2 + eps],
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

        if operation == "divide":
            den_samples = np.where(np.abs(den_samples) < eps, eps, den_samples)
            res_samples = num_samples / den_samples
        else:
            res_samples = num_samples

        mean_res = np.mean(res_samples)
        std_res = np.std(res_samples)
        valid = res_samples[np.abs(res_samples - mean_res) < 5 * std_res]

        if len(valid) < 100:
            if operation == "divide":
                val = num_val / den_val if den_val != 0 else 0.0
            else:
                val = num_val

            if num_val != 0 and den_val != 0:
                rel_err = np.sqrt((num_se / abs(num_val)) ** 2 + (den_se / abs(den_val)) ** 2)
            else:
                rel_err = 0.1
            ci_width = 1.96 * abs(val) * rel_err
            return val, val - ci_width, val + ci_width

        val = float(np.mean(valid))
        low = float(np.percentile(valid, 2.5))
        high = float(np.percentile(valid, 97.5))

        # Ensure value is within CI
        val = max(low, min(high, val))

        return val, low, high

    @staticmethod
    def calculate_all_derived(base_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived metrics from base forecast data.
        
        Derived metrics:
        - productivity_gbp_per_job = nominal_gva_mn_gbp / emp_total_jobs × 1e6
        - income_per_worker_gbp = gdhi_total_mn_gbp / emp_total_jobs × 1e6
        """
        derived_rows = []
        
        for region_code in base_data["region_code"].unique():
            region_data = base_data[base_data["region_code"] == region_code]
            region_name = region_data["region"].iloc[0] if "region" in region_data.columns else region_code
            
            for year in sorted(region_data["year"].unique()):
                year_data = region_data[region_data["year"] == year]
                
                # Extract metrics with CIs
                metrics = {}
                for _, row in year_data.iterrows():
                    metric = row["metric"]
                    val = row["value"]
                    ci_l = row.get("ci_lower", np.nan)
                    ci_u = row.get("ci_upper", np.nan)
                    if pd.isna(ci_l):
                        ci_l = val * 0.95 if val else 0.0
                    if pd.isna(ci_u):
                        ci_u = val * 1.05 if val else 0.0
                    metrics[metric] = (float(val), float(ci_l), float(ci_u))
                
                data_type = year_data["data_type"].iloc[0] if "data_type" in year_data.columns else "forecast"
                
                # Productivity: GVA (GBP) / jobs
                if "nominal_gva_mn_gbp" in metrics and "emp_total_jobs" in metrics:
                    gva_mn = metrics["nominal_gva_mn_gbp"]
                    gva_gbp = (gva_mn[0] * 1e6, gva_mn[1] * 1e6, gva_mn[2] * 1e6)
                    
                    prod_val, prod_low, prod_high = DerivedMetricsCalculator.calculate_with_uncertainty(
                        gva_gbp,
                        metrics["emp_total_jobs"],
                        operation="divide",
                        correlation=0.8,
                    )
                    
                    derived_rows.append({
                        "region_code": region_code,
                        "region": region_name,
                        "year": int(year),
                        "metric": "productivity_gbp_per_job",
                        "value": prod_val,
                        "ci_lower": prod_low,
                        "ci_upper": prod_high,
                        "formula": "nominal_gva_mn_gbp / emp_total_jobs * 1e6",
                        "method": "derived_monte_carlo",
                        "data_type": data_type,
                    })
                
                # Income per worker: GDHI (GBP) / jobs
                if "gdhi_total_mn_gbp" in metrics and "emp_total_jobs" in metrics:
                    gdhi_mn = metrics["gdhi_total_mn_gbp"]
                    gdhi_gbp = (gdhi_mn[0] * 1e6, gdhi_mn[1] * 1e6, gdhi_mn[2] * 1e6)
                    
                    inc_val, inc_low, inc_high = DerivedMetricsCalculator.calculate_with_uncertainty(
                        gdhi_gbp,
                        metrics["emp_total_jobs"],
                        operation="divide",
                        correlation=0.7,
                    )
                    
                    derived_rows.append({
                        "region_code": region_code,
                        "region": region_name,
                        "year": int(year),
                        "metric": "income_per_worker_gbp",
                        "value": inc_val,
                        "ci_lower": inc_low,
                        "ci_upper": inc_high,
                        "formula": "gdhi_total_mn_gbp / emp_total_jobs * 1e6",
                        "method": "derived_monte_carlo",
                        "data_type": data_type,
                    })
                
                # GDHI per head: GDHI (GBP) / population
                if "gdhi_total_mn_gbp" in metrics and "population_total" in metrics:
                    gdhi_mn = metrics["gdhi_total_mn_gbp"]
                    gdhi_gbp = (gdhi_mn[0] * 1e6, gdhi_mn[1] * 1e6, gdhi_mn[2] * 1e6)
                    
                    val, low, high = DerivedMetricsCalculator.calculate_with_uncertainty(
                        gdhi_gbp,
                        metrics["population_total"],
                        operation="divide",
                        correlation=0.3,
                    )
                    
                    # FIX: Check data_type of components, not first row
                    # If EITHER component is forecast, then gdhi_per_head is forecast
                    gdhi_row = year_data[year_data["metric"] == "gdhi_total_mn_gbp"]
                    pop_row = year_data[year_data["metric"] == "population_total"]
                    gdhi_dtype = gdhi_row["data_type"].iloc[0] if not gdhi_row.empty and "data_type" in gdhi_row.columns else "forecast"
                    pop_dtype = pop_row["data_type"].iloc[0] if not pop_row.empty and "data_type" in pop_row.columns else "forecast"
                    per_head_dtype = "forecast" if (gdhi_dtype == "forecast" or pop_dtype == "forecast") else "historical"
                    
                    derived_rows.append({
                        "region_code": region_code,
                        "region": region_name,
                        "year": int(year),
                        "metric": "gdhi_per_head_gbp",
                        "value": val,
                        "ci_lower": low,
                        "ci_upper": high,
                        "formula": "gdhi_total_mn_gbp / population_total * 1e6",
                        "method": "derived_monte_carlo",
                        "data_type": per_head_dtype,
                    })
        
        return pd.DataFrame(derived_rows)


# ===============================
# Main ITL1 forecasting pipeline
# ===============================

class ITL1ForecasterV35:
    """
    Full ITL1 forecasting pipeline V3.5.
    """

    def __init__(self, config: ForecastConfig):
        self.config = config
        self.data_manager = DataManagerV35(config)
        self.forecaster = AdvancedForecastingV35(config)
        self.derived_calc = DerivedMetricsCalculator()

        if config.use_macro_anchoring and HAVE_DUCKDB:
            self.macro_manager = MacroAnchorManager(config.duckdb_path)
            if self.macro_manager.has_anchors():
                self.reconciler = TopDownReconciler(config, self.macro_manager)
                logger.info("✓ Macro anchors loaded – top-down reconciliation ENABLED")
            else:
                self.reconciler = None
                logger.warning("⚠ No macro anchors found – reconciliation DISABLED")
        else:
            self.macro_manager = None
            self.reconciler = None

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("=" * 70)
        logger.info("ITL1 FORECASTING ENGINE V3.5")
        logger.info("=" * 70)

        historical = self.data_manager.load_all_data()
        tasks = self._identify_tasks(historical)
        logger.info(f"Identified {len(tasks)} forecasting tasks")

        forecast_rows = self._run_forecasts(historical, tasks)

        base_data = pd.concat(
            [historical, pd.DataFrame(forecast_rows)], ignore_index=True
        )

        # Reconcile to UK macro totals (additive metrics only)
        if self.reconciler:
            base_data = self.reconciler.reconcile(base_data)

        # Apply sanity caps to base data, then re-reconcile in a cap-aware way
        # to keep UK totals exact while preserving caps.
        base_data = self._apply_sanity_caps(base_data)
        if self.reconciler:
            base_data = self.reconciler.reconcile(base_data, sanity_caps=SANITY_CAPS)

        logger.info("\nCalculating derived metrics...")
        derived_data = self.derived_calc.calculate_all_derived(base_data)
        logger.info(f"  Generated {len(derived_data)} derived metric rows")

        # Split gdhi_per_head_gbp - goes to BASE table, not derived
        gdhi_per_head = derived_data[derived_data["metric"] == "gdhi_per_head_gbp"].copy()
        derived_data = derived_data[derived_data["metric"] != "gdhi_per_head_gbp"]
        
        if not gdhi_per_head.empty:
            # Merge gdhi_per_head_gbp into base_data
            gdhi_per_head = gdhi_per_head.drop(columns=["formula"], errors="ignore")
            base_data = pd.concat([base_data, gdhi_per_head], ignore_index=True)
            logger.info(f"  ✓ Merged {len(gdhi_per_head)} gdhi_per_head_gbp rows into base forecast")

        logger.info("\nSaving outputs...")
        self._save_outputs(base_data, derived_data)

        logger.info("✅ ITL1 V3.5 forecasting completed")
        return base_data, derived_data

    def _identify_tasks(self, data: pd.DataFrame) -> List[Dict]:
        tasks = []
        for (region_code, region, metric), group in data.groupby(
            ["region_code", "region", "metric"]
        ):
            # Only forecast base metrics
            if metric not in self.config.metric_definitions:
                continue
                
            n = len(group)
            # NI-only jobs series is shorter historically; allow forecast with fewer years.
            # Fully dynamic: forecast still starts at last observed year + 1.
            min_years = 8 if metric == "emp_total_jobs_ni" else self.config.min_history_years
            if n < min_years:
                continue

            last_year = int(group["year"].max())
            horizon = min(self.config.target_year - last_year, 30)
            if horizon <= 0:
                continue

            metric_info = self.config.metric_definitions.get(metric, {})

            tasks.append(
                {
                    "region_code": region_code,
                    "region": region,
                    "metric": metric,
                    "horizon": horizon,
                    "last_year": last_year,
                    "history_len": n,
                    "metric_info": metric_info,
                }
            )
        return tasks

    def _run_forecasts(self, data: pd.DataFrame, tasks: List[Dict]) -> List[Dict]:
        results: List[Dict] = []
        processed_pairs = set()

        tasks_by_region: Dict[str, List[Dict]] = {}
        for t in tasks:
            tasks_by_region.setdefault(t["region_code"], []).append(t)

        for region_code, region_tasks in tasks_by_region.items():
            logger.info(f"\nRegion {region_code}:")

            region_metrics = {t["metric"] for t in region_tasks}

            # VAR/VECM systems first
            for sys_name, sys_metrics in self.config.var_systems.items():
                available = [m for m in sys_metrics if m in region_metrics]
                if len(available) < 2 or not self.config.use_var_systems:
                    continue

                horizons = [
                    t["horizon"] for t in region_tasks if t["metric"] in available
                ]
                if not horizons:
                    continue
                
                full_horizon = min(horizons)
                var_horizon = min(full_horizon, self.config.var_max_horizon)

                if var_horizon <= 0:
                    continue

                logger.info(f"  VAR/VECM system: {available}")

                sys_last_year = max(
                    t["last_year"] for t in region_tasks if t["metric"] in available
                )
                breaks = [
                    b
                    for b in self.config.structural_breaks
                    if sys_last_year - 20 <= b["year"] <= sys_last_year
                ]

                var_res = self.forecaster.var_forecaster.forecast_system(
                    data, region_code, available, var_horizon, breaks
                )
                
                if var_res is None:
                    logger.info("    VAR/VECM failed – fallback to univariate.")
                else:
                    for metric, fc in var_res.items():
                        processed_pairs.add((region_code, metric))
                        region_name = next(
                            t["region"] for t in region_tasks if t["metric"] == metric
                        )
                        task = next(t for t in region_tasks if t["metric"] == metric)
                        
                        # If VAR horizon < full horizon, blend with univariate
                        if var_horizon < full_horizon:
                            # Run univariate for full horizon
                            series_data = data[(data["region_code"] == region_code) & (data["metric"] == metric)].sort_values("year")
                            series = pd.Series(
                                series_data["value"].values,
                                index=series_data["year"].astype(int),
                                name=metric,
                            )
                            use_log = self.config.use_log_transform.get(metric, False)
                            working = np.log(series) if use_log and (series > 0).all() else series
                            
                            uni_fc = self.forecaster.forecast_univariate(
                                working, full_horizon, breaks, task["metric_info"]
                            )
                            
                            # Convert univariate to level space
                            uni_vals = np.asarray(uni_fc["values"])
                            uni_ci_lo = np.asarray(uni_fc["ci_lower"])
                            uni_ci_hi = np.asarray(uni_fc["ci_upper"])
                            if use_log:
                                uni_vals = np.exp(uni_vals)
                                uni_ci_lo = np.exp(uni_ci_lo)
                                uni_ci_hi = np.exp(uni_ci_hi)
                            
                            # Blend
                            blended = self._blend_var_univariate(
                                fc, 
                                {"values": uni_vals, "ci_lower": uni_ci_lo, "ci_upper": uni_ci_hi, "years": uni_fc["years"]},
                                full_horizon,
                                fc["method"],
                                uni_fc["method"]
                            )
                            
                            for i, year in enumerate(blended["years"]):
                                results.append({
                                    "region": region_name,
                                    "region_code": region_code,
                                    "metric": metric,
                                    "year": int(year),
                                    "value": float(blended["values"][i]),
                                    "ci_lower": float(blended["ci_lower"][i]),
                                    "ci_upper": float(blended["ci_upper"][i]),
                                    "method": blended["method"],
                                    "data_type": "forecast",
                                    "source": "model",
                                })
                        else:
                            for i, year in enumerate(fc["years"]):
                                results.append({
                                    "region": region_name,
                                    "region_code": region_code,
                                    "metric": metric,
                                    "year": int(year),
                                    "value": float(fc["values"][i]),
                                    "ci_lower": float(fc["ci_lower"][i]),
                                    "ci_upper": float(fc["ci_upper"][i]),
                                    "method": fc["method"],
                                    "data_type": "forecast",
                                    "source": "model",
                                })
                    
                    logger.info(f"    ✓ VAR completed for {len(var_res)} metrics.")

            # Univariate for remaining metrics
            for t in region_tasks:
                if (t["region_code"], t["metric"]) in processed_pairs:
                    continue

                metric = t["metric"]
                logger.info(f"  {metric}: univariate")

                subset = data[
                    (data["region_code"] == t["region_code"])
                    & (data["metric"] == metric)
                ].sort_values("year")

                series = pd.Series(
                    subset["value"].values.astype(float),
                    index=subset["year"].astype(int),
                    name=metric,
                )

                use_log = self.config.use_log_transform.get(metric, False)
                working = (
                    np.log(series)
                    if use_log and (series > 0).all()
                    else series.copy()
                )

                breaks = [
                    b
                    for b in self.config.structural_breaks
                    if t["last_year"] - 20 <= b["year"] <= t["last_year"]
                ]

                try:
                    fc = self.forecaster.forecast_univariate(
                        working, t["horizon"], breaks, t["metric_info"]
                    )

                    fc_vals = np.asarray(fc["values"])
                    fc_ci_lo = np.asarray(fc["ci_lower"])
                    fc_ci_hi = np.asarray(fc["ci_upper"])
                    
                    if use_log:
                        if fc.get("cv_residual_std"):
                            bc = np.exp(0.5 * fc["cv_residual_std"] ** 2)
                        else:
                            bc = 1.0
                        fc_vals = np.exp(fc_vals) * bc
                        fc_ci_lo = np.exp(fc_ci_lo)
                        fc_ci_hi = np.exp(fc_ci_hi)

                    for i, year in enumerate(fc["years"]):
                        results.append({
                            "region": t["region"],
                            "region_code": t["region_code"],
                            "metric": metric,
                            "year": int(year),
                            "value": float(fc_vals[i]),
                            "ci_lower": float(fc_ci_lo[i]),
                            "ci_upper": float(fc_ci_hi[i]),
                            "method": fc["method"],
                            "data_type": "forecast",
                            "source": "model",
                        })
                except Exception as e:
                    logger.error(f"  FAILED {t['region_code']}-{metric} univariate: {e}")

        return results

    def _blend_var_univariate(
        self,
        var_forecast: Dict,
        uni_forecast: Dict,
        full_horizon: int,
        var_method: str,
        uni_method: str
    ) -> Dict:
        """Smooth exponential blend from VAR → Univariate."""
        decay = self.config.var_blend_decay
        
        var_vals = np.asarray(var_forecast["values"])
        var_ci_lo = np.asarray(var_forecast["ci_lower"])
        var_ci_hi = np.asarray(var_forecast["ci_upper"])
        var_len = len(var_vals)
        
        uni_vals = np.asarray(uni_forecast["values"])
        uni_ci_lo = np.asarray(uni_forecast["ci_lower"])
        uni_ci_hi = np.asarray(uni_forecast["ci_upper"])
        
        # Extend VAR if shorter than full horizon
        if var_len < full_horizon:
            if var_len >= 3:
                trend = (var_vals[-1] - var_vals[-3]) / 2
            else:
                trend = 0
            
            extension_len = full_horizon - var_len
            var_extension = var_vals[-1] + trend * np.arange(1, extension_len + 1)
            var_vals = np.concatenate([var_vals, var_extension])
            
            last_ci_width = (var_ci_hi[-1] - var_ci_lo[-1]) / 2
            ext_factors = np.sqrt(np.arange(var_len + 1, full_horizon + 1))
            var_ci_lo = np.concatenate([var_ci_lo, var_extension - last_ci_width * ext_factors / np.sqrt(var_len)])
            var_ci_hi = np.concatenate([var_ci_hi, var_extension + last_ci_width * ext_factors / np.sqrt(var_len)])
        
        h = np.arange(1, full_horizon + 1)
        w_var = np.exp(-decay * h)
        w_uni = 1 - w_var
        
        w_total = w_var + w_uni
        w_var = w_var / w_total
        w_uni = w_uni / w_total
        
        var_vals = np.nan_to_num(var_vals, nan=uni_vals)
        var_ci_lo = np.nan_to_num(var_ci_lo, nan=uni_ci_lo)
        var_ci_hi = np.nan_to_num(var_ci_hi, nan=uni_ci_hi)
        
        blended_values = w_var * var_vals + w_uni * uni_vals
        blended_ci_lower = w_var * var_ci_lo + w_uni * uni_ci_lo
        blended_ci_upper = w_var * var_ci_hi + w_uni * uni_ci_hi
        
        return {
            "values": blended_values,
            "ci_lower": blended_ci_lower,
            "ci_upper": blended_ci_upper,
            "years": uni_forecast["years"],
            "method": f"{var_method}+{uni_method}+blend",
        }

    def _apply_sanity_caps(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        logger.info("\n  Applying sanity caps...")
        for metric, cap in SANITY_CAPS.items():
            mask = (data["metric"] == metric) & (data["value"] > cap)
            if mask.any():
                logger.warning(f"    {metric}: Capping {mask.sum()} values at {cap:,.0f}")
                data.loc[mask, "value"] = cap
                if "ci_upper" in data.columns:
                    data.loc[mask, "ci_upper"] = np.minimum(data.loc[mask, "ci_upper"], cap * 1.2)
        return data

    def _save_outputs(self, base_data: pd.DataFrame, derived_data: pd.DataFrame):
        # Base long CSV
        long_path = self.config.output_dir / "itl1_forecast_long.csv"
        base_data.to_csv(long_path, index=False)
        logger.info(f"✓ Base long: {long_path}")

        # Base wide CSV
        wide = base_data.pivot_table(
            index=["region", "region_code", "metric"],
            columns="year",
            values="value",
            aggfunc="first",
        ).reset_index()
        wide.columns.name = None
        wide_path = self.config.output_dir / "itl1_forecast_wide.csv"
        wide.to_csv(wide_path, index=False)
        logger.info(f"✓ Base wide: {wide_path}")

        # Derived CSV
        derived_path = self.config.output_dir / "itl1_derived.csv"
        derived_data.to_csv(derived_path, index=False)
        logger.info(f"✓ Derived: {derived_path}")

        # CI diagnostics
        if "ci_lower" in base_data.columns and "ci_upper" in base_data.columns:
            ci_data = base_data[
                ["region_code", "metric", "year", "value", "ci_lower", "ci_upper"]
            ].dropna(subset=["ci_lower", "ci_upper"])
            if not ci_data.empty:
                ci_data["ci_width"] = ci_data["ci_upper"] - ci_data["ci_lower"]
                ci_path = self.config.output_dir / "itl1_confidence_intervals.csv"
                ci_data.to_csv(ci_path, index=False)
                logger.info(f"✓ CIs: {ci_path}")

        # Metadata
        meta = {
            "run_timestamp": datetime.now().isoformat(),
            "version": "3.5",
            "base_metrics": list(self.config.metric_definitions.keys()),
            "derived_metrics": ["productivity_gbp_per_job", "income_per_worker_gbp"],
            "calculated_in_base": ["gdhi_per_head_gbp"],
            "config": {
                "target_year": self.config.target_year,
                "var_enabled": self.config.use_var_systems,
                "var_max_horizon": self.config.var_max_horizon,
                "var_blend_decay": self.config.var_blend_decay,
                "macro_anchoring": self.config.use_macro_anchoring,
            },
            "growth_caps": GROWTH_CAPS,
            "sanity_caps": SANITY_CAPS,
            "data_summary": {
                "regions": int(base_data["region_code"].nunique()),
                "base_metrics": int(base_data["metric"].nunique()),
                "base_forecast_obs": int((base_data["data_type"] == "forecast").sum()),
                "derived_obs": int(len(derived_data)),
            },
        }

        if hasattr(base_data, "attrs") and "reconciliation_log" in base_data.attrs:
            recon = base_data.attrs["reconciliation_log"]
            if recon:
                meta["reconciliation_summary"] = {
                    "adjustments": len(recon),
                    "avg_scale_factor": float(np.mean([r["scale_factor"] for r in recon])),
                }

        meta_path = self.config.output_dir / "itl1_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        logger.info(f"✓ Metadata: {meta_path}")

        # DuckDB - base table
        if HAVE_DUCKDB:
            try:
                df = base_data.copy()

                if "metric" in df.columns:
                    df["metric_id"] = df["metric"]
                if "year" in df.columns:
                    df["period"] = df["year"]
                if "region_name" not in df.columns and "region" in df.columns:
                    df["region_name"] = df["region"]
                if "region_level" not in df.columns:
                    df["region_level"] = "ITL1"
                if "unit" not in df.columns:
                    df["unit"] = df["metric_id"].map(
                        lambda m: self.config.metric_definitions.get(m, {}).get("unit", "unknown")
                    )
                if "freq" not in df.columns:
                    df["freq"] = "A"

                df["forecast_run_date"] = datetime.now().date()
                df["forecast_version"] = "3.5"

                cols = [
                    "region_code", "region_name", "region_level", "metric_id", "period",
                    "value", "unit", "freq", "data_type", "ci_lower", "ci_upper",
                    "forecast_run_date", "forecast_version",
                ]
                df_flat = df[[c for c in cols if c in df.columns]].reset_index(drop=True)

                con = duckdb.connect(str(self.config.duckdb_path))
                con.execute("CREATE SCHEMA IF NOT EXISTS gold")
                con.register("itl1_df", df_flat)
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
                
                logger.info(f"✓ DuckDB: gold.itl1_forecast ({len(df_flat)} rows)")

                # DuckDB - derived table
                if not derived_data.empty:
                    dfd = derived_data.copy()
                    dfd["metric_id"] = dfd["metric"]
                    dfd["period"] = dfd["year"]
                    dfd["region_name"] = dfd["region"]
                    dfd["region_level"] = "ITL1"
                    dfd["unit"] = "GBP"
                    dfd["freq"] = "A"
                    dfd["forecast_run_date"] = datetime.now().date()
                    dfd["forecast_version"] = "3.5"

                    derived_cols = [
                        "region_code", "region_name", "region_level", "metric_id", "period",
                        "value", "unit", "freq", "data_type", "ci_lower", "ci_upper",
                        "formula", "forecast_run_date", "forecast_version",
                    ]
                    dfd_flat = dfd[[c for c in derived_cols if c in dfd.columns]].reset_index(drop=True)

                    con.register("itl1_derived_df", dfd_flat)
                    con.execute("CREATE OR REPLACE TABLE gold.itl1_derived AS SELECT * FROM itl1_derived_df")
                    logger.info(f"✓ DuckDB: gold.itl1_derived ({len(dfd_flat)} rows)")

                con.close()
            except Exception as e:
                logger.warning(f"⚠ DuckDB save failed: {e}")


# ===============================
# Entry point
# ===============================

def main():
    try:
        config = ForecastConfig()

        logger.info("=" * 70)
        logger.info("ITL1 FORECAST V3.5")
        logger.info("=" * 70)
        logger.info(f"  Silver: {config.silver_path}")
        logger.info(f"  Target: {config.target_year}")
        logger.info(f"  VAR: {config.use_var_systems} (max horizon: {config.var_max_horizon})")
        logger.info(f"  Blend decay: {config.var_blend_decay}")
        logger.info(f"  Macro anchoring: {config.use_macro_anchoring}")
        logger.info(f"  Base metrics: {list(config.metric_definitions.keys())}")

        if config.use_macro_anchoring:
            if not HAVE_DUCKDB or not config.duckdb_path.exists():
                logger.warning("⚠ DuckDB missing – macro anchoring disabled.")
                config.use_macro_anchoring = False
            else:
                try:
                    tmp_mgr = MacroAnchorManager(config.duckdb_path)
                    if not tmp_mgr.has_anchors():
                        logger.warning("⚠ No UK macro forecasts found. Run macro engine first.")
                        config.use_macro_anchoring = False
                except Exception as e:
                    logger.warning(f"⚠ Could not load macro anchors: {e}")
                    config.use_macro_anchoring = False

        forecaster = ITL1ForecasterV35(config)
        base_results, derived_results = forecaster.run()

        logger.info("=" * 70)
        logger.info("✅ ITL1 V3.5 COMPLETED")
        logger.info(f"📊 Regions: {base_results['region_code'].nunique()}")
        logger.info(f"📊 Base metrics: {base_results['metric'].nunique()}")
        logger.info(f"📊 Base forecasts: {(base_results['data_type'] == 'forecast').sum()}")
        logger.info(f"📊 Derived rows: {len(derived_results)}")

        if config.use_macro_anchoring and hasattr(base_results, "attrs"):
            recon_log = base_results.attrs.get("reconciliation_log", [])
            logger.info(f"📊 Reconciliation adjustments: {len(recon_log)}")

        logger.info("\n📊 2050 SUMMARY (London):")
        london = base_results[(base_results["region_code"] == "E12000007") & (base_results["year"] == 2050)]
        for m in ["nominal_gva_mn_gbp", "gdhi_total_mn_gbp", "gdhi_per_head_gbp", "emp_total_jobs", "population_total", "population_16_64"]:
            v = london[london["metric"] == m]["value"]
            if not v.empty:
                if m == "gdhi_per_head_gbp":
                    logger.info(f"    {m}: £{v.iloc[0]:,.0f}")
                elif "gbp" in m:
                    logger.info(f"    {m}: £{v.iloc[0]:,.0f}m")
                else:
                    logger.info(f"    {m}: {v.iloc[0]:,.0f}")

        logger.info("=" * 70)
        return base_results, derived_results

    except Exception as e:
        logger.error(f"ITL1 pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()