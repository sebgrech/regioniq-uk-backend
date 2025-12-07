# scripts/forecast/population_itl1_forecast.py
"""
Region IQ — ITL1 population forecaster (totals + working-age)
Reads multi-metric wide CSV and writes forecast artifacts for dashboards.

Inputs
------
data/clean/population_ITL1_metrics_wide.csv
  Columns: region, region_code, metric, <year1>, <year2>, ...

Outputs
-------
data/forecast/population_ITL1_forecast_only.csv
  region, region_code, metric, year, forecast_value, method, ci_lower, ci_upper

data/forecast/population_ITL1_hist_forecast_long.csv
  region, region_code, metric, year, value, source, method?, ci_lower?, ci_upper?

data/forecast/population_ITL1_hist_forecast_wide.csv
  Pivoted wide: (region, region_code, metric) × years (history+forecast)

Notes
-----
- Tries small ARIMA order set; falls back to linear trend, then flat carry.
- Non-negative clamp.
- Works even if statsmodels is not installed (linear fallback).
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------- Config ----------------------
IN_WIDE = Path("data/clean/population_ITL1_metrics_wide.csv")
OUT_DIR = Path("data/forecast"); OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FORECAST_ONLY = OUT_DIR / "population_ITL1_forecast_only.csv"
OUT_HIST_FORECAST_LONG = OUT_DIR / "population_ITL1_hist_forecast_long.csv"
OUT_HIST_FORECAST_WIDE = OUT_DIR / "population_ITL1_hist_forecast_wide.csv"

# Prefer TARGET_YEAR; if None, use HORIZON steps
TARGET_YEAR = 2030
HORIZON = 6  # used only if TARGET_YEAR is None

# Candidate ARIMA orders to try (fast + safe for annual demography)
ARIMA_ORDERS = [(1,1,1), (0,1,1), (1,1,0), (0,1,0)]

# ------------------ Optional dependency ------------------
try:
    from statsmodels.tsa.arima.model import ARIMA
    HAVE_SM = True
except Exception:
    HAVE_SM = False


# ---------------------- Helpers ----------------------
def to_year_cols(columns) -> list[int]:
    years = []
    for c in columns:
        try:
            years.append(int(c))
        except Exception:
            pass
    return sorted(years)


def linear_trend_forecast(ts: pd.Series, steps: int) -> tuple[pd.Series, str]:
    """OLS line on (year,value); fallback to flat if too short."""
    x = ts.index.values.astype(float)
    y = ts.values.astype(float)
    if len(ts) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        future_x = np.arange(x[-1] + 1, x[-1] + 1 + steps, 1.0)
        yhat = intercept + slope * future_x
        out = pd.Series(yhat, index=future_x.astype(int))
        return out.clip(lower=0.0), "lintrend"
    # flat carry
    future_x = np.arange(ts.index[-1] + 1, ts.index[-1] + 1 + steps, 1, dtype=int)
    yhat = np.repeat(float(ts.iloc[-1]), steps)
    return pd.Series(yhat, index=future_x).clip(lower=0.0), "flat"


def arima_try(ts: pd.Series, steps: int) -> tuple[pd.Series, pd.DataFrame | None, str] | None:
    """Try a few ARIMA orders; return (mean, conf_int, method) or None if no fit."""
    if not HAVE_SM:
        return None
    for order in ARIMA_ORDERS:
        try:
            model = ARIMA(ts, order=order)
            fitted = model.fit(method_kwargs={"warn_convergence": False})
            fc = fitted.get_forecast(steps=steps)
            mean = fc.predicted_mean
            ci = fc.conf_int()
            # align index to integer future years
            future_years = np.arange(ts.index[-1] + 1, ts.index[-1] + 1 + steps, dtype=int)
            mean.index = future_years
            ci.index = future_years
            # clamp
            mean = mean.clip(lower=0.0)
            ci.iloc[:, 0] = ci.iloc[:, 0].clip(lower=0.0)
            ci.iloc[:, 1] = ci.iloc[:, 1].clip(lower=0.0)
            return mean.astype(float), ci.astype(float), f"arima{order}"
        except Exception:
            continue
    return None


def compute_steps(last_year: int, target_year: int | None, horizon: int) -> int:
    if target_year is not None:
        return max(0, int(target_year) - int(last_year))
    return int(horizon)


# ---------------------- Main ----------------------
def main():
    if not IN_WIDE.exists():
        raise FileNotFoundError(f"Input not found: {IN_WIDE}. Did you run the cleaner?")

    df = pd.read_csv(IN_WIDE)

    id_cols = ["region", "region_code", "metric"]
    year_cols = [c for c in df.columns if c not in id_cols]
    years = to_year_cols(year_cols)
    if not years:
        raise ValueError("No year columns detected in input wide file.")

    # Ensure numeric matrix for year columns
    df[year_cols] = df[year_cols].apply(pd.to_numeric, errors="coerce")

    # Storage
    rows_fc_only = []          # forecasts only
    rows_hist_forecast = []    # history + forecasts w/ source & optional CI

    # Iterate each (region, metric) row
    for _, row in df.iterrows():
        region = row["region"]
        code = row["region_code"]
        metric = row["metric"]

        series = pd.Series(row[year_cols].values.astype(float), index=years)
        ts = series.dropna()
        if ts.empty:
            # no history available
            continue

        last_year = int(ts.index.max())
        steps = compute_steps(last_year, TARGET_YEAR, HORIZON)

        # --- always write history to hist+forecast long
        hist_df = pd.DataFrame({
            "region": region,
            "region_code": code,
            "metric": metric,
            "year": ts.index.astype(int),
            "value": ts.values.astype(float),
            "source": "historical",
            "method": "observed",
            "ci_lower": pd.NA,
            "ci_upper": pd.NA,
        })
        rows_hist_forecast.append(hist_df)

        if steps <= 0:
            # Nothing to forecast for this series
            continue

        # --- Forecast: ARIMA -> linear -> flat
        arima_out = arima_try(ts, steps)
        if arima_out is not None:
            mean, ci, method = arima_out
        else:
            mean, method = linear_trend_forecast(ts, steps)
            ci = None

        # Build forecast rows
        fc_years = mean.index.astype(int)
        fc_vals = mean.values.astype(float)

        fc_only_df = pd.DataFrame({
            "region": region,
            "region_code": code,
            "metric": metric,
            "year": fc_years,
            "forecast_value": fc_vals,
            "method": method,
            "ci_lower": ci.iloc[:, 0].values if ci is not None else pd.NA,
            "ci_upper": ci.iloc[:, 1].values if ci is not None else pd.NA,
        })
        rows_fc_only.append(fc_only_df)

        fc_long_df = pd.DataFrame({
            "region": region,
            "region_code": code,
            "metric": metric,
            "year": fc_years,
            "value": fc_vals,
            "source": "forecast",
            "method": method,
            "ci_lower": ci.iloc[:, 0].values if ci is not None else pd.NA,
            "ci_upper": ci.iloc[:, 1].values if ci is not None else pd.NA,
        })
        rows_hist_forecast.append(fc_long_df)

    # Concatenate and save
    if rows_fc_only:
        out_fc_only = pd.concat(rows_fc_only, ignore_index=True).sort_values(
            ["region_code", "metric", "year"]
        )
    else:
        out_fc_only = pd.DataFrame(columns=[
            "region","region_code","metric","year","forecast_value","method","ci_lower","ci_upper"
        ])

    out_hist_forecast = pd.concat(rows_hist_forecast, ignore_index=True).sort_values(
        ["region_code", "metric", "year"]
    )

    # Write CSVs
    out_fc_only.to_csv(OUT_FORECAST_ONLY, index=False)
    out_hist_forecast.to_csv(OUT_HIST_FORECAST_LONG, index=False)

    # Wide for convenience (history+forecast together)
    wide = out_hist_forecast.pivot_table(
        index=["region", "region_code", "metric"],
        columns="year",
        values="value",
        aggfunc="first"
    ).reset_index()
    wide.columns.name = None
    # Reorder id cols + years
    idc = ["region", "region_code", "metric"]
    year_cols_out = [c for c in wide.columns if c not in idc]
    year_cols_out_sorted = sorted([int(c) for c in year_cols_out])
    wide = wide[idc + year_cols_out_sorted]
    wide.to_csv(OUT_HIST_FORECAST_WIDE, index=False)

    # QA summary
    used_methods = (
        out_fc_only["method"].value_counts().to_dict()
        if not out_fc_only.empty else {}
    )
    print(f"💾 Saved forecasts only → {OUT_FORECAST_ONLY}")
    print(f"💾 Saved hist+forecast (long) → {OUT_HIST_FORECAST_LONG}")
    print(f"💾 Saved hist+forecast (wide) → {OUT_HIST_FORECAST_WIDE}")
    print(f"Models used: {used_methods}")
    if not out_fc_only.empty:
        print(f"Regions: {out_fc_only['region'].nunique()}  |  Metrics: {out_fc_only['metric'].nunique()}")
        print(f"Forecast span: {int(out_fc_only['year'].min())}–{int(out_fc_only['year'].max())}")
        print(f"ARIMA available: {HAVE_SM}")

if __name__ == "__main__":
    main()