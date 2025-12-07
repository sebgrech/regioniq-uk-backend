# scripts/forecast/income_GDHI_ITL1_forecast.py
"""
Region IQ — ITL1 GDHI income forecaster (per-head → total derived from population)

What it does
------------
• Forecasts GDHI per head (£) in log-space (ARIMA grid → linear → flat).
• Derives GDHI total (£m) using population hist+forecast:
      gdhi_total_mn_gbp = gdhi_per_head_gbp * total_population / 1e6
• Writes forecasts-only, hist+forecast (long), and hist+forecast (wide) artifacts.

Inputs
------
data/clean/income_GDHI_ITL1_wide.csv
  Columns: region, region_code, metric (gdhi_per_head_gbp, gdhi_total_mn_gbp), <year1>, <year2>, ...
data/forecast/population_ITL1_hist_forecast_wide.csv
  From population forecaster; must include metric == 'total_population'.

Outputs
-------
data/forecast/income_GDHI_ITL1_forecast_only.csv
data/forecast/income_GDHI_ITL1_hist_forecast_long.csv
data/forecast/income_GDHI_ITL1_hist_forecast_wide.csv
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import itertools

warnings.filterwarnings("ignore")

# ---------------------- Config ----------------------
IN_GDHI_WIDE = Path("data/clean/incomes/income_GDHI_ITL1_wide.csv")
IN_POP_WIDE  = Path("data/forecast/population_ITL1_hist_forecast_wide.csv")

OUT_DIR = Path("data/forecast"); OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FC_ONLY = OUT_DIR / "income_GDHI_ITL1_forecast_only.csv"
OUT_HF_LONG = OUT_DIR / "income_GDHI_ITL1_hist_forecast_long.csv"
OUT_HF_WIDE = OUT_DIR / "income_GDHI_ITL1_hist_forecast_wide.csv"

TARGET_YEAR = 2030          # if None, use HORIZON
HORIZON     = 6

# ARIMA grid (small for speed on annual data)
P_RANGE = range(0, 3)
D_RANGE = range(0, 3)
Q_RANGE = range(0, 3)

# Optional dependency
try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.stats.diagnostic import acorr_ljungbox
    HAVE_SM = True
except Exception:
    HAVE_SM = False


# ---------------------- Helpers ----------------------
def to_year_cols(columns):
    """Return sorted list of columns that can be parsed as INT years."""
    years = []
    for c in columns:
        try:
            years.append(int(c))
        except Exception:
            pass
    return sorted(years)

def compute_steps(last_year, target_year, horizon):
    return max(0, int(target_year) - int(last_year)) if target_year is not None else int(horizon)

def arima_log_forecast(ts_log: pd.Series, steps: int):
    """Grid-search ARIMA(p,d,q) in 0..2; pick min-AIC; record Ljung-Box p."""
    if not HAVE_SM:
        return None
    best = {"aic": np.inf, "order": None, "fit": None, "lb_p": None}
    for p, d, q in itertools.product(P_RANGE, D_RANGE, Q_RANGE):
        try:
            fit = ARIMA(ts_log, order=(p, d, q)).fit(method_kwargs={"warn_convergence": False})
            aic = fit.aic
            if aic < best["aic"]:
                lb = acorr_ljungbox(fit.resid, lags=min(8, max(1, len(ts_log)//3)), return_df=True)
                lb_p = float(lb["lb_pvalue"].iloc[-1])
                best = {"aic": aic, "order": (p, d, q), "fit": fit, "lb_p": lb_p}
        except Exception:
            continue
    if best["fit"] is None:
        return None
    fc = best["fit"].get_forecast(steps=steps)
    mean_log = fc.predicted_mean
    ci_log   = fc.conf_int()

    fut = np.arange(ts_log.index[-1] + 1, ts_log.index[-1] + 1 + steps, dtype=int)
    mean_log.index = fut
    ci_log.index   = fut

    mean = np.exp(mean_log).clip(lower=0.0)
    ci = pd.DataFrame({
        "lower": np.exp(ci_log.iloc[:, 0]).clip(lower=0.0),
        "upper": np.exp(ci_log.iloc[:, 1]).clip(lower=0.0),
    }, index=fut)
    tag = f"arima{best['order']}_log_lb_p={best['lb_p']:.3f}"
    return mean.astype(float), ci.astype(float), tag

def linear_log_forecast(ts_log: pd.Series, steps: int):
    """Linear trend in log-space; CIs if statsmodels present; else no CI."""
    x = ts_log.index.values.astype(float)
    y = ts_log.values.astype(float)

    # OLS with prediction intervals
    if HAVE_SM and len(ts_log) >= 2:
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        fx = np.arange(x[-1] + 1, x[-1] + 1 + steps, 1.0)
        FX = sm.add_constant(fx)
        pred = model.get_prediction(FX).summary_frame(alpha=0.05)  # 95% PI
        mean = np.exp(pred["mean"]).clip(lower=0.0)
        ci = pd.DataFrame({
            "lower": np.exp(pred["obs_ci_lower"]).clip(lower=0.0),
            "upper": np.exp(pred["obs_ci_upper"]).clip(lower=0.0),
        }, index=fx.astype(int))
        return pd.Series(mean.values, index=fx.astype(int)), ci, "lintrend_log"

    # Numpy fallback (no CI)
    if len(ts_log) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        fx = np.arange(x[-1] + 1, x[-1] + 1 + steps, 1.0)
        yhat = np.exp(intercept + slope * fx)
        return pd.Series(yhat, index=fx.astype(int)).clip(lower=0.0), None, "lintrend_log"

    # Flat carry if too short
    fx = np.arange(ts_log.index[-1] + 1, ts_log.index[-1] + 1 + steps, 1, dtype=int)
    yhat = np.repeat(float(np.exp(ts_log.iloc[-1])), steps)
    return pd.Series(yhat, index=fx).clip(lower=0.0), None, "flat_log"


# ---------------------- Main ----------------------
def main():
    if not IN_GDHI_WIDE.exists():
        raise FileNotFoundError(f"Missing GDHI wide: {IN_GDHI_WIDE}")
    if not IN_POP_WIDE.exists():
        raise FileNotFoundError(f"Missing population hist+forecast wide: {IN_POP_WIDE}")

    gdhi = pd.read_csv(IN_GDHI_WIDE).drop_duplicates()
    pop  = pd.read_csv(IN_POP_WIDE).drop_duplicates()

    id_cols = ["region", "region_code", "metric"]
    ycols_gdhi = [c for c in gdhi.columns if c not in id_cols]
    ycols_pop  = [c for c in pop.columns  if c not in id_cols]

    # Parse year columns as ints, but keep parallel string lists for DataFrame indexing
    years_g = to_year_cols(ycols_gdhi)
    years_p = to_year_cols(ycols_pop)
    if not years_g:
        raise ValueError("No year columns detected in GDHI wide file.")
    if not years_p:
        raise ValueError("No year columns detected in population hist+forecast wide file.")
    years_g_str = [str(y) for y in years_g]
    years_p_str = [str(y) for y in years_p]

    # Coerce numeric for the year columns
    gdhi[years_g_str] = gdhi[years_g_str].apply(pd.to_numeric, errors="coerce")
    pop[years_p_str]  = pop[years_p_str].apply(pd.to_numeric,  errors="coerce")

    # Subsets
    per_head = gdhi[gdhi["metric"] == "gdhi_per_head_gbp"].copy()
    total_h  = gdhi[gdhi["metric"] == "gdhi_total_mn_gbp"].copy()
    pop_tot  = pop[pop["metric"] == "total_population"].copy()
    if per_head.empty:
        raise ValueError("gdhi_per_head_gbp not found in GDHI wide input.")
    if pop_tot.empty:
        raise ValueError("total_population not found in population hist+forecast wide input.")

    # Keys by region_code (robust)
    keys = per_head[["region_code", "region"]].drop_duplicates().merge(
        pop_tot[["region_code"]].drop_duplicates(), on="region_code"
    )

    rows_fc_only = []
    rows_hf = []

    for _, k in keys.iterrows():
        code = k["region_code"]

        # --- per-head history
        ph_row = per_head.loc[per_head["region_code"] == code]
        if ph_row.empty:
            continue
        ph_row = ph_row.iloc[0]
        region = ph_row["region"]

        vals = ph_row[years_g_str].values.astype(float)
        ts_ph = pd.Series(vals, index=years_g).dropna()
        if ts_ph.empty:
            continue

        rows_hf.append(pd.DataFrame({
            "region": region, "region_code": code, "metric": "gdhi_per_head_gbp",
            "year": ts_ph.index.astype(int), "value": ts_ph.values.astype(float),
            "source": "historical", "method": "observed", "ci_lower": pd.NA, "ci_upper": pd.NA
        }))

        # --- total history (if present)
        tot_row = total_h.loc[total_h["region_code"] == code]
        if not tot_row.empty:
            tvals = tot_row.iloc[0][years_g_str].values.astype(float)
            ts_tot = pd.Series(tvals, index=years_g).dropna()
            if not ts_tot.empty:
                rows_hf.append(pd.DataFrame({
                    "region": region, "region_code": code, "metric": "gdhi_total_mn_gbp",
                    "year": ts_tot.index.astype(int), "value": ts_tot.values.astype(float),
                    "source": "historical", "method": "observed", "ci_lower": pd.NA, "ci_upper": pd.NA
                }))

        # --- forecast horizon
        steps = compute_steps(int(ts_ph.index.max()), TARGET_YEAR, HORIZON)
        if steps <= 0:
            continue

        # --- forecast per-head in log-space
        z = np.log(ts_ph.clip(lower=1e-9))
        ar = arima_log_forecast(z, steps)
        if ar is None:
            mean_ph, ci_ph, method = linear_log_forecast(z, steps)
        else:
            mean_ph, ci_ph, method = ar

        fc_years = mean_ph.index.astype(int)

        # forecasts-only (per-head)
        rows_fc_only.append(pd.DataFrame({
            "region": region, "region_code": code, "metric": "gdhi_per_head_gbp",
            "year": fc_years, "forecast_value": mean_ph.values.astype(float),
            "method": method,
            "ci_lower": (ci_ph["lower"].values if ci_ph is not None else pd.NA),
            "ci_upper": (ci_ph["upper"].values if ci_ph is not None else pd.NA),
        }))
        # hist+forecast (per-head)
        rows_hf.append(pd.DataFrame({
            "region": region, "region_code": code, "metric": "gdhi_per_head_gbp",
            "year": fc_years, "value": mean_ph.values.astype(float),
            "source": "forecast", "method": method,
            "ci_lower": (ci_ph["lower"].values if ci_ph is not None else pd.NA),
            "ci_upper": (ci_ph["upper"].values if ci_ph is not None else pd.NA),
        }))

        # --- derive totals for years covered by population
        pop_row = pop_tot.loc[pop_tot["region_code"] == code]
        if pop_row.empty:
            continue
        # Make sure we index population using string year cols
        fc_years_str = [str(y) for y in fc_years]
        # Only keep years that exist in population columns
        fc_years_keep = [y for y in fc_years if str(y) in years_p_str]
        if not fc_years_keep:
            continue

        pop_vals = pop_row.iloc[0][[str(y) for y in fc_years_keep]].values.astype(float)
        per_head_vals = mean_ph.reindex(fc_years_keep).values.astype(float)

        tot_vals = (per_head_vals * pop_vals) / 1_000_000.0

        rows_fc_only.append(pd.DataFrame({
            "region": region, "region_code": code, "metric": "gdhi_total_mn_gbp",
            "year": fc_years_keep, "forecast_value": tot_vals,
            "method": f"{method}_derived", "ci_lower": pd.NA, "ci_upper": pd.NA
        }))
        rows_hf.append(pd.DataFrame({
            "region": region, "region_code": code, "metric": "gdhi_total_mn_gbp",
            "year": fc_years_keep, "value": tot_vals,
            "source": "forecast", "method": f"{method}_derived",
            "ci_lower": pd.NA, "ci_upper": pd.NA
        }))

    # ---------------------- Save artifacts ----------------------
    fc_only = (pd.concat(rows_fc_only, ignore_index=True).sort_values(["region_code","metric","year"])
               if rows_fc_only else pd.DataFrame(columns=["region","region_code","metric","year","forecast_value","method","ci_lower","ci_upper"]))
    hf_long = (pd.concat(rows_hf, ignore_index=True).sort_values(["region_code","metric","year"])
               if rows_hf else pd.DataFrame(columns=["region","region_code","metric","year","value","source","method","ci_lower","ci_upper"]))

    fc_only.to_csv(OUT_FC_ONLY, index=False)
    hf_long.to_csv(OUT_HF_LONG, index=False)

    wide = hf_long.pivot_table(index=["region","region_code","metric"], columns="year", values="value", aggfunc="first").reset_index()
    wide.columns.name = None
    idc = ["region","region_code","metric"]
    year_cols_out = [c for c in wide.columns if c not in idc]
    # Ensure numeric sort even if columns came back as strings
    year_cols_out_sorted = sorted([int(c) for c in year_cols_out]) if year_cols_out else []
    wide = wide[idc + year_cols_out_sorted] if year_cols_out_sorted else wide[idc]
    wide.to_csv(OUT_HF_WIDE, index=False)

    used = fc_only["method"].value_counts().to_dict() if not fc_only.empty else {}
    print(f"💾 Saved forecasts only → {OUT_FC_ONLY}")
    print(f"💾 Saved hist+forecast (long) → {OUT_HF_LONG}")
    print(f"💾 Saved hist+forecast (wide) → {OUT_HF_WIDE}")
    print(f"Models used: {used}")
    print(f"ARIMA available: {HAVE_SM}")

if __name__ == "__main__":
    main()
