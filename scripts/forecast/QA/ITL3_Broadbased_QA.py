#!/usr/bin/env python3
"""
ITL3 Forecast QA Script V7.1
============================
Aligned with ITL2 Forecast QA architecture and LAD gate.

BASE TABLE: gold.itl3_forecast (8 base metrics)
  - Additive (5):
        nominal_gva_mn_gbp
        gdhi_total_mn_gbp
        emp_total_jobs
        population_total
        population_16_64
  - Rates (2):
        employment_rate_pct
        unemployment_rate_pct
  - Calculated, stored in base (1):
        gdhi_per_head_gbp

DERIVED TABLE: gold.itl3_derived (2 metrics, Monte Carlo)
  - productivity_gbp_per_job
  - income_per_worker_gbp

RECONCILIATION: ITL3 → ITL2 (additive metrics only)

LAD GATING:
  - This script validates ITL3 before LAD forecasts can run.
  - If this script fails, LAD MUST NOT run.

CASCADE ARCHITECTURE:
    UK → ITL1 → ITL2 → ITL3 → LAD

Exit codes:
    0: All critical checks passed — safe to run LAD
    1: Fatal errors — do NOT proceed to LAD

Usage:
    python3 scripts/forecast/QA/ITL3_Broadbased_QA.py

V7.1 Changes:
  - Fixed check_lad_readiness to check ALL base metrics for year gaps,
    not just additive metrics. This ensures gdhi_per_head_gbp and rate
    metrics are also validated for continuous coverage before LAD.
"""

import sys
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict
import json


class ITL3ForecastQA:
    """ITL3 Forecast QA V7.1 — LAD Gatekeeper, aligned with V4+ ITL2 QA."""

    def __init__(
        self,
        db_path: str = "data/lake/warehouse.duckdb",
        tolerance: float = 0.01
    ):
        self.db_path = db_path
        self.tolerance = tolerance  # 1% default
        self.conn = duckdb.connect(db_path, read_only=True)

        # ----------------------------------------------------------------------
        # METRIC DEFINITIONS
        # ----------------------------------------------------------------------

        # Additive (sum over children = parent)
        self.additive_metrics = [
            "nominal_gva_mn_gbp",
            "gdhi_total_mn_gbp",
            "emp_total_jobs",
            "population_total",
            "population_16_64",
        ]

        # Rate metrics – primary from NOMIS, not reconciled
        self.rate_metrics = {
            "employment_rate_pct": {
                "min": 0.0,
                "max": 100.0,
                "expected_range": (50, 90),
            },
            "unemployment_rate_pct": {
                "min": 0.0,
                "max": 100.0,
                "expected_range": (1, 15),
            },
        }

        # Calculated metrics that *belong in base* (primary KPI)
        self.calculated_base_metrics = [
            "gdhi_per_head_gbp",
        ]

        # Final expected base metrics (8)
        self.expected_base_metrics = (
            self.additive_metrics
            + list(self.rate_metrics.keys())
            + self.calculated_base_metrics
        )

        # Derived metrics (Monte Carlo, in gold.itl3_derived)
        self.derived_metrics = [
            "productivity_gbp_per_job",
            "income_per_worker_gbp",
        ]

        # Geography expectations
        # The pipeline now includes NI LAD equivalents (N09...) in the LAD layer,
        # so ITL3 coverage should match the full bottom-up ITL3 universe implied
        # by LAD children (GB + NI).
        self.expected_itl3_count = self._expected_bottomup_itl3_count()
        self.expected_itl2_count = 46

        # LAD requirements
        self.required_lad_horizon = 2050
        self.forecast_years = [2024, 2025, 2030, 2040, 2050]

        # Storage for QA issues
        self.issues: List[str] = []
        self.warnings: List[str] = []

        # Mapping ITL3 -> ITL2
        self._load_geography_mapping()

    def _expected_bottomup_itl3_count(self) -> int:
        """Compute expected bottom-up ITL3 count from the canonical lookup (ITL3s with LAD/LGD children)."""
        try:
            lookup = pd.read_csv("data/reference/master_2025_geography_lookup.csv")
            lookup.columns = [c.replace("\ufeff", "") for c in lookup.columns]
            lad = lookup["LAD25CD"].dropna().astype(str)
            bottomup_mask = lad.str[0].isin(["E", "W", "S", "N"])
            return int(lookup.loc[bottomup_mask, "ITL325CD"].nunique())
        except Exception:
            # Fallback to historical constant (pre-fix behavior).
            return 182

    # ======================================================================
    # DATA LOADERS
    # ======================================================================

    def _load_geography_mapping(self) -> None:
        """Load ITL3→ITL2 mapping from reference CSV."""
        try:
            lookup = pd.read_csv("data/reference/master_2025_geography_lookup.csv")
            lookup.columns = [c.replace("\ufeff", "") for c in lookup.columns]

            self.itl3_to_itl2 = lookup[["ITL325CD", "ITL225CD"]].drop_duplicates()
            self.itl3_to_itl2.columns = ["itl3_code", "itl2_code"]
        except Exception as e:
            self.issues.append(f"Failed to load geography mapping: {e}")
            self.itl3_to_itl2 = pd.DataFrame(columns=["itl3_code", "itl2_code"])

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load ITL3 base, ITL3 derived, ITL2 parent, and CI subset."""

        base_query = """
        SELECT 
            region_code, region_name, metric_id, period,
            value, data_type, ci_lower, ci_upper, method
        FROM gold.itl3_forecast
        WHERE period >= 2020
        ORDER BY region_code, metric_id, period
        """
        base_df = self.conn.execute(base_query).df()

        # Derived forecasts (productivity + income per worker)
        try:
            derived_query = """
            SELECT 
                region_code, region_name, metric_id, period,
                value, data_type, ci_lower, ci_upper
            FROM gold.itl3_derived
            WHERE period >= 2020
            ORDER BY region_code, metric_id, period
            """
            derived_df = self.conn.execute(derived_query).df()
        except Exception:
            derived_df = pd.DataFrame()

        itl2_query = """
        SELECT region_code, region_name, metric_id, period, value, data_type
        FROM gold.itl2_forecast
        WHERE period >= 2020
        ORDER BY region_code, metric_id, period
        """
        itl2_df = self.conn.execute(itl2_query).df()

        ci_df = base_df[
            (base_df["period"] >= 2025) & (base_df["ci_lower"].notna())
        ][["region_code", "metric_id", "period", "value", "ci_lower", "ci_upper"]].copy()

        return base_df, derived_df, itl2_df, ci_df

    # ======================================================================
    # METRIC COVERAGE
    # ======================================================================

    def check_base_metric_coverage(self, base_df: pd.DataFrame) -> None:
        """Ensure all 8 expected base metrics are present in gold.itl3_forecast."""
        print("\n" + "=" * 80)
        print("BASE TABLE METRIC COVERAGE (gold.itl3_forecast)")
        print("=" * 80)

        present = set(base_df["metric_id"].unique())
        expected = set(self.expected_base_metrics)

        missing = expected - present
        extra = present - expected

        print(f"\n📊 Expected: {len(expected)} | Present: {len(present)}")

        if missing:
            self.issues.append(f"CRITICAL: Missing base metrics: {missing}")
            print(f"\n  ❌ CRITICAL: Missing {len(missing)} metrics:")
            for m in sorted(missing):
                print(f"     - {m}")
        else:
            print("\n  ✅ All expected base metrics present")

        if extra:
            print(f"\n  ℹ️  Extra metrics (not expected):")
            for m in sorted(extra):
                print(f"     - {m}")

    def check_derived_table(self, derived_df: pd.DataFrame) -> None:
        """Check that derived table has productivity + income_per_worker."""
        print("\n" + "=" * 80)
        print("DERIVED TABLE (gold.itl3_derived)")
        print("=" * 80)

        if derived_df.empty:
            self.warnings.append("Derived table empty or missing")
            print("\n  ⚠️  gold.itl3_derived is empty or doesn't exist")
            return

        present = set(derived_df["metric_id"].unique())
        expected = set(self.derived_metrics)

        missing = expected - present

        print(f"\n📊 Expected: {len(expected)} | Present: {len(present)}")

        if missing:
            self.warnings.append(f"Missing derived metrics: {missing}")
            print("\n  ⚠️  Missing derived metrics:")
            for m in sorted(missing):
                print(f"     - {m}")
        else:
            print("\n  ✅ All derived metrics present")

        # Basic sanity: NaNs / negatives
        for metric in self.derived_metrics:
            m_data = derived_df[derived_df["metric_id"] == metric]
            if m_data.empty:
                continue
            nan_count = m_data["value"].isna().sum()
            neg_count = (m_data["value"] < 0).sum()

            if nan_count > 0:
                self.warnings.append(f"{metric}: {nan_count} NaN values")
            if neg_count > 0:
                self.warnings.append(f"{metric}: {neg_count} negative values")

        print(f"  📊 Derived rows: {len(derived_df):,}")

    # ======================================================================
    # DATA QUALITY
    # ======================================================================

    def check_data_quality(self, base_df: pd.DataFrame) -> None:
        """Region count, nulls, and negatives."""
        print("\n" + "=" * 80)
        print("DATA QUALITY CHECKS")
        print("=" * 80)

        # Region count (only for the expected base metric set).
        # Extra metrics (e.g. NI-only series) should not fail the global region count gate.
        base_expected = base_df[base_df["metric_id"].isin(self.expected_base_metrics)]
        n_regions = base_expected["region_code"].nunique()
        print(f"\n📊 Region count: {n_regions} (expected: {self.expected_itl3_count})")

        if n_regions != self.expected_itl3_count:
            self.issues.append(
                f"Region count mismatch: {n_regions} vs {self.expected_itl3_count}"
            )
            print("  ❌ Region count mismatch")
        else:
            print("  ✅ Region count correct")

        # Null forecast values
        fcst_df = base_df[base_df["data_type"] == "forecast"]
        nulls = fcst_df["value"].isna().sum()
        print(f"\n📊 Null forecast values: {nulls}")

        if nulls > 0:
            self.issues.append(f"{nulls} null forecast values")
            print(f"  ❌ {nulls} null values found")
        else:
            print("  ✅ No null forecast values")

        # Negatives in additive + calculated base metrics
        print("\n📊 Negative value check:")
        for m in self.additive_metrics + self.calculated_base_metrics:
            d = fcst_df[fcst_df["metric_id"] == m]
            nneg = (d["value"] < 0).sum()
            if nneg > 0:
                self.issues.append(f"{m}: {nneg} negative values")
                print(f"  ❌ {m}: {nneg} negative values")
            else:
                print(f"  ✅ {m}: no negatives")

    # ======================================================================
    # LAD READINESS (ITL3 → LAD GATE)
    # ======================================================================

    def check_lad_readiness(self, base_df: pd.DataFrame) -> None:
        """
        LAD GATEKEEPER:
          • Forecast horizon to required LAD year
          • Continuous coverage for ALL base metrics (no year gaps)
        
        V7.1 FIX: Now checks all expected_base_metrics, not just additive_metrics.
        This ensures gdhi_per_head_gbp and rate metrics are validated for
        continuous year coverage before LAD can proceed.
        """
        print("\n" + "=" * 80)
        print("LAD READINESS CHECK (ITL3 → LAD GATE)")
        print("=" * 80)

        fcst_df = base_df[base_df["data_type"] == "forecast"]

        # Horizon
        fcst_years = fcst_df["period"].unique()
        print("\n📊 Forecast horizon check")
        if fcst_years.size == 0:
            self.issues.append("No ITL3 forecast years found.")
            print("  ❌ No forecast years found")
        else:
            max_year = int(fcst_years.max())
            if max_year < self.required_lad_horizon:
                self.issues.append(
                    f"ITL3 horizon ends at {max_year}, LAD needs {self.required_lad_horizon}"
                )
                print(
                    f"  ❌ Horizon too short: ends {max_year}, need {self.required_lad_horizon}"
                )
            else:
                print(f"  ✅ Forecast horizon OK → {max_year}")

        # V7.1 FIX: Check ALL base metrics for continuous coverage, not just additive
        print("\n📊 Continuous year coverage (all base metrics):")
        for metric in self.expected_base_metrics:
            df_m = base_df[base_df["metric_id"] == metric].copy()
            if df_m.empty:
                self.issues.append(f"{metric}: no data for gap check")
                print(f"  ❌ {metric}: no data for gap check")
                continue

            # Use series groupby to avoid FutureWarning
            grouped = (
                df_m[["region_code", "period"]]
                .drop_duplicates()
                .groupby("region_code")["period"]
            )

            def _missing_years(periods: pd.Series) -> set:
                years = periods.astype(int).tolist()
                return set(range(min(years), max(years) + 1)) - set(years)

            gaps = grouped.apply(_missing_years)
            n_with_gaps = sum(1 for g in gaps if len(g) > 0)

            if n_with_gaps > 0:
                self.issues.append(
                    f"{metric}: {n_with_gaps} ITL3 regions have year gaps (breaks LAD)"
                )
                print(f"  ❌ {metric}: {n_with_gaps} regions have gaps")
            else:
                print(f"  ✅ {metric}: continuous years for all regions")

    # ======================================================================
    # RECONCILIATION: ITL3 → ITL2
    # ======================================================================

    def check_reconciliation(self, base_df: pd.DataFrame, itl2_df: pd.DataFrame) -> None:
        """Check that ITL3 additive metrics aggregate to ITL2 anchors."""
        print("\n" + "=" * 80)
        print("RECONCILIATION: ITL3 → ITL2 (ADDITIVE METRICS)")
        print("=" * 80)

        if self.itl3_to_itl2.empty:
            self.issues.append("No ITL3→ITL2 mapping available")
            print("\n  ❌ Cannot check reconciliation - no mapping")
            return

        # Merge parent codes
        base_with_parent = base_df.merge(
            self.itl3_to_itl2,
            left_on="region_code",
            right_on="itl3_code",
            how="left",
        )

        for metric in self.additive_metrics:
            print(f"\n📊 {metric}")
            print("-" * 80)

            itl2_vals = (
                itl2_df[
                    (itl2_df["metric_id"] == metric)
                    & (itl2_df["data_type"] == "forecast")
                ]
                .set_index(["region_code", "period"])["value"]
            )

            if itl2_vals.empty:
                self.warnings.append(f"{metric}: No ITL2 anchor found")
                print("  ⚠️  No ITL2 anchor found")
                continue

            itl3_fcst = base_with_parent[
                (base_with_parent["metric_id"] == metric)
                & (base_with_parent["data_type"] == "forecast")
            ]
            itl3_sums = (
                itl3_fcst.groupby(["itl2_code", "period"])["value"].sum().sort_index()
            )

            failures = 0

            for year in self.forecast_years:
                year_errors = []

                for itl2_code in itl3_sums.index.get_level_values("itl2_code").unique():
                    if pd.isna(itl2_code):
                        continue

                    try:
                        itl3_sum = itl3_sums.loc[(itl2_code, year)]
                        itl2_val = itl2_vals.loc[(itl2_code, year)]
                    except KeyError:
                        continue

                    diff = itl3_sum - itl2_val
                    pct_diff = (diff / itl2_val * 100) if itl2_val != 0 else 0

                    if abs(pct_diff) > self.tolerance * 100:
                        year_errors.append((itl2_code, itl3_sum, itl2_val, pct_diff))
                        failures += 1

                total_itl3 = itl3_sums[
                    itl3_sums.index.get_level_values("period") == year
                ].sum()
                total_itl2 = itl2_vals[
                    itl2_vals.index.get_level_values("period") == year
                ].sum()
                delta = total_itl3 - total_itl2
                pct = (delta / total_itl2 * 100) if total_itl2 != 0 else 0

                status = "❌ FAIL" if year_errors else "✅ PASS"
                print(
                    f"  {year}: "
                    f"ITL3={total_itl3:>15,.0f} | "
                    f"ITL2={total_itl2:>15,.0f} | "
                    f"Δ={delta:+12,.0f} ({pct:+7.3f}%)"
                )

            if failures > 0:
                self.issues.append(
                    f"{metric}: {failures} ITL2 parents with >{self.tolerance*100:.0f}% deviation"
                )

    # ======================================================================
    # CALCULATED METRICS: GDHI PER HEAD
    # ======================================================================

    def check_gdhi_per_head(self, base_df: pd.DataFrame) -> None:
        """Validate gdhi_per_head_gbp = gdhi_total / population_total × 1e6."""
        print("\n" + "=" * 80)
        print("GDHI PER HEAD CALCULATION CHECK")
        print("=" * 80)

        fcst = base_df[base_df["data_type"] == "forecast"]

        pivot = (
            fcst.pivot_table(
                index=["region_code", "region_name", "period"],
                columns="metric_id",
                values="value",
            )
            .reset_index()
        )

        required = ["gdhi_total_mn_gbp", "population_total", "gdhi_per_head_gbp"]
        if not all(c in pivot.columns for c in required):
            self.warnings.append("Cannot validate gdhi_per_head - missing components")
            print("\n  ⚠️  Missing required columns for validation")
            return

        pivot["expected"] = (
            pivot["gdhi_total_mn_gbp"] / pivot["population_total"] * 1e6
        )
        pivot["diff_pct"] = (
            (pivot["gdhi_per_head_gbp"] - pivot["expected"]) / pivot["expected"] * 100
        )

        for year in [2025, 2030, 2050]:
            yr_data = pivot[pivot["period"] == year]
            if yr_data.empty:
                continue

            max_err = yr_data["diff_pct"].abs().max()

            if max_err > self.tolerance * 100:
                worst = yr_data.loc[yr_data["diff_pct"].abs().idxmax()]
                self.issues.append(
                    f"gdhi_per_head {year}: {worst['region_name']} error={max_err:.2f}%"
                )
                print(f"\n  {year}: ❌ Max error: {max_err:.3f}%")
                print(
                    f"       Worst: {worst['region_name']} "
                    f"(actual={worst['gdhi_per_head_gbp']:.0f}, "
                    f"expected={worst['expected']:.0f})"
                )
            else:
                print(f"\n  {year}: ✅ Max error: {max_err:.3f}%")

    # ======================================================================
    # RATE BOUNDS
    # ======================================================================

    def check_rate_bounds(self, base_df: pd.DataFrame) -> None:
        """Validate rate metrics are within 0–100% and plausible ranges."""
        print("\n" + "=" * 80)
        print("RATE METRIC BOUNDS")
        print("=" * 80)

        for rate, bounds in self.rate_metrics.items():
            data = base_df[base_df["metric_id"] == rate]

            if data.empty:
                self.warnings.append(f"{rate}: No data found")
                print(f"\n📊 {rate}: ⚠️  No data")
                continue

            print(f"\n📊 {rate}")
            print("-" * 80)
            print(f"  Regions: {data['region_code'].nunique()}")

            below = data[data["value"] < bounds["min"]]
            above = data[data["value"] > bounds["max"]]

            if not below.empty:
                self.issues.append(
                    f"{rate}: {len(below)} values < {bounds['min']}%"
                )
                print(f"  ❌ {len(below)} below {bounds['min']}%")

            if not above.empty:
                self.issues.append(
                    f"{rate}: {len(above)} values > {bounds['max']}%"
                )
                print(f"  ❌ {len(above)} above {bounds['max']}%")

            if below.empty and above.empty:
                print(
                    f"  ✅ All within [{bounds['min']}, {bounds['max']}]%"
                )

            # Expected range by region (forecast only)
            fcst = data[data["data_type"] == "forecast"]
            if not fcst.empty:
                exp_min, exp_max = bounds["expected_range"]
                stats = fcst.groupby("region_code")["value"].agg(["min", "max"])
                outliers = stats[
                    (stats["min"] < exp_min) | (stats["max"] > exp_max)
                ]

                if not outliers.empty:
                    self.warnings.append(
                        f"{rate}: {len(outliers)} regions outside typical range"
                    )
                    print(
                        f"  ⚠️  {len(outliers)} regions outside "
                        f"[{exp_min}, {exp_max}]%"
                    )

    # ======================================================================
    # HISTORICAL CONTINUITY
    # ======================================================================

    def check_historical_continuity(self, base_df: pd.DataFrame) -> None:
        """Check smooth transition at the historical→forecast boundary."""
        print("\n" + "=" * 80)
        print("HISTORICAL CONTINUITY")
        print("=" * 80)

        hist = base_df[base_df["data_type"] == "historical"]
        fcst = base_df[base_df["data_type"] == "forecast"]

        if hist.empty or fcst.empty:
            print("\n  ⚠️  Cannot check continuity (missing hist or forecast)")
            return

        last_hist = hist["period"].max()
        first_fcst = fcst["period"].min()

        print(
            f"\n🔗 Transition: {int(last_hist)} → {int(first_fcst)}"
        )
        print("-" * 80)

        for metric in self.expected_base_metrics:
            trans = base_df[
                (base_df["metric_id"] == metric)
                & (base_df["period"].isin([last_hist, first_fcst]))
            ].pivot_table(
                index="region_code",
                columns="period",
                values="value",
            )

            if (
                trans.empty
                or last_hist not in trans.columns
                or first_fcst not in trans.columns
            ):
                continue

            if metric in self.rate_metrics:
                trans["change"] = trans[first_fcst] - trans[last_hist]
                extreme = trans[trans["change"].abs() > 5]
                threshold = ">5pp"
            else:
                trans["change"] = (
                    trans[first_fcst] / trans[last_hist] - 1
                ) * 100
                extreme = trans[
                    (trans["change"] > 20) | (trans["change"] < -10)
                ]
                threshold = ">20% or <-10%"

            if not extreme.empty:
                self.warnings.append(
                    f"{metric}: {len(extreme)} extreme jumps"
                )
                print(
                    f"  ⚠️  {metric}: {len(extreme)} jumps ({threshold})"
                )
            else:
                print(f"  ✅ {metric}")

    # ======================================================================
    # CONFIDENCE INTERVALS
    # ======================================================================

    def check_confidence_intervals(self, ci_df: pd.DataFrame) -> None:
        """Validate CI ordering and that points lie within bands."""
        print("\n" + "=" * 80)
        print("CONFIDENCE INTERVALS")
        print("=" * 80)

        if ci_df.empty:
            self.warnings.append("No CIs found")
            print("\n  ⚠️  No confidence intervals")
            return

        print(f"\n📊 {len(ci_df):,} CI records")

        invalid = ci_df[ci_df["ci_lower"] > ci_df["ci_upper"]]
        if not invalid.empty:
            self.issues.append(
                f"{len(invalid)} CIs with lower > upper"
            )
            print(f"  ❌ {len(invalid)} invalid ordering")
        else:
            print("  ✅ All CIs properly ordered")

        outside = ci_df[
            (ci_df["value"] < ci_df["ci_lower"])
            | (ci_df["value"] > ci_df["ci_upper"])
        ]
        if not outside.empty:
            self.warnings.append(
                f"{len(outside)} points outside CIs"
            )
            print(f"  ⚠️  {len(outside)} points outside CIs")
        else:
            print("  ✅ All points within CIs")

    def check_ci_diagnostics_per_metric(self, base_df: pd.DataFrame) -> None:
        """CI width statistics per metric and horizon."""
        print("\n" + "=" * 80)
        print("CI WIDTH DIAGNOSTICS BY METRIC")
        print("=" * 80)

        ci_data = base_df[
            (base_df["data_type"] == "forecast")
            & (base_df["ci_lower"].notna())
            & (base_df["ci_upper"].notna())
        ].copy()

        if ci_data.empty:
            print("\n  ⚠️  No CI data for diagnostics")
            return

        ci_data["ci_width"] = ci_data["ci_upper"] - ci_data["ci_lower"]
        ci_data["ci_width_pct"] = (
            ci_data["ci_width"] / ci_data["value"].abs()
        ) * 100

        for metric in self.expected_base_metrics:
            m_data = ci_data[ci_data["metric_id"] == metric]
            if m_data.empty:
                continue

            print(f"\n📊 {metric}")
            print("-" * 80)

            for year in [2025, 2030, 2040, 2050]:
                yr = m_data[m_data["period"] == year]
                if yr.empty:
                    continue

                width_pct = yr["ci_width_pct"]
                print(
                    f"  {year}: median={width_pct.median():>6.1f}% | "
                    f"p90={width_pct.quantile(0.9):>6.1f}% | "
                    f"max={width_pct.max():>6.1f}%"
                )

                extreme = yr[yr["ci_width_pct"] > 50]
                if not extreme.empty:
                    self.warnings.append(
                        f"{metric} {year}: {len(extreme)} CIs >50% width"
                    )

            if (
                2025 in m_data["period"].values
                and 2050 in m_data["period"].values
            ):
                early = m_data[m_data["period"] == 2025][
                    "ci_width_pct"
                ].median()
                late = m_data[m_data["period"] == 2050][
                    "ci_width_pct"
                ].median()

                if early > 0 and late > early * 5:
                    self.warnings.append(
                        f"{metric}: CI width expands >5x by 2050"
                    )
                    print(
                        f"  ⚠️  CI expands {late/early:.1f}x from 2025→2050"
                    )

    # ======================================================================
    # GROWTH RATE BOUNDS (BASE METRICS ONLY)
    # ======================================================================

    def check_growth_rate_bounds(self, base_df: pd.DataFrame) -> None:
        """YoY growth rate bounds for base metrics."""
        print("\n" + "=" * 80)
        print("YEAR-ON-YEAR GROWTH RATE VALIDATION")
        print("=" * 80)

        growth_bounds: Dict[str, Tuple[float, float]] = {
            "nominal_gva_mn_gbp": (-10, 15),
            "gdhi_total_mn_gbp": (-10, 15),
            "gdhi_per_head_gbp": (-8, 12),
            "emp_total_jobs": (-8, 8),
            "population_total": (-3, 5),
            "population_16_64": (-3, 5),
            "employment_rate_pct": (-6, 6),  # pp change
            "unemployment_rate_pct": (-5, 8),  # pp change
        }

        fcst = base_df[base_df["data_type"] == "forecast"].copy()

        for metric, (min_g, max_g) in growth_bounds.items():
            m_data = fcst[fcst["metric_id"] == metric]
            if m_data.empty:
                continue

            print(f"\n📊 {metric} (bounds: {min_g}% to {max_g}%)")
            print("-" * 80)

            violations = []

            for region in m_data["region_code"].unique():
                r_data = m_data[m_data["region_code"] == region].sort_values(
                    "period"
                )

                r_data = r_data.copy()
                if metric in self.rate_metrics:
                    r_data["growth"] = r_data["value"].diff()
                else:
                    r_data["growth"] = (
                        r_data["value"].pct_change() * 100
                    )

                extreme = r_data[
                    (r_data["growth"] < min_g)
                    | (r_data["growth"] > max_g)
                ]
                if not extreme.empty:
                    region_name = (
                        r_data["region_name"].iloc[0]
                        if "region_name" in r_data.columns
                        else region
                    )
                    for _, row in extreme.iterrows():
                        violations.append(
                            {
                                "region": region_name,
                                "year": int(row["period"]),
                                "growth": row["growth"],
                            }
                        )

            if violations:
                self.warnings.append(
                    f"{metric}: {len(violations)} YoY growth violations"
                )
                print(f"  ⚠️  {len(violations)} violations")
                for v in sorted(
                    violations,
                    key=lambda x: abs(x["growth"]),
                    reverse=True,
                )[:3]:
                    print(
                        f"     {v['region']} {v['year']}: {v['growth']:+.1f}%"
                    )
            else:
                print("  ✅ All regions within bounds")

    # ======================================================================
    # CROSS-METRIC COHERENCE (INCLUDING DERIVED)
    # ======================================================================

    def check_cross_metric_coherence(
        self, base_df: pd.DataFrame, derived_df: pd.DataFrame
    ) -> None:
        """Cross-metric checks: GDHI/head, productivity, income/worker."""
        print("\n" + "=" * 80)
        print("CROSS-METRIC COHERENCE")
        print("=" * 80)

        fcst = base_df[base_df["data_type"] == "forecast"]

        pivot = (
            fcst.pivot_table(
                index=["region_code", "region_name", "period"],
                columns="metric_id",
                values="value",
            )
            .reset_index()
        )

        # GDHI/head consistency
        if all(
            c in pivot.columns
            for c in ["gdhi_total_mn_gbp", "population_total", "gdhi_per_head_gbp"]
        ):
            print("\n🔗 GDHI per head consistency")
            print("-" * 80)

            pivot["gdhi_calc"] = (
                pivot["gdhi_total_mn_gbp"] / pivot["population_total"] * 1e6
            )
            pivot["gdhi_diff_pct"] = (
                (pivot["gdhi_per_head_gbp"] - pivot["gdhi_calc"])
                / pivot["gdhi_calc"]
                * 100
            )

            max_diff = pivot["gdhi_diff_pct"].abs().max()
            if max_diff > 1:
                self.warnings.append(
                    f"GDHI/head deviation up to {max_diff:.1f}%"
                )
                print(f"  ⚠️  Max deviation: {max_diff:.2f}%")
            else:
                print("  ✅ Stored vs calculated within 1%")

        # Productivity (from derived table)
        if not derived_df.empty:
            print("\n🔗 Productivity consistency")
            print("-" * 80)

            prod = derived_df[
                (derived_df["metric_id"] == "productivity_gbp_per_job")
                & (derived_df["data_type"] == "forecast")
            ][["region_code", "period", "value"]].rename(
                columns={"value": "prod_stored"}
            )

            if (
                not prod.empty
                and "nominal_gva_mn_gbp" in pivot.columns
                and "emp_total_jobs" in pivot.columns
            ):
                merged = pivot.merge(
                    prod, on=["region_code", "period"], how="inner"
                )
                if not merged.empty:
                    merged["prod_calc"] = (
                        merged["nominal_gva_mn_gbp"]
                        / merged["emp_total_jobs"]
                        * 1e6
                    )
                    merged["prod_diff_pct"] = (
                        (merged["prod_stored"] - merged["prod_calc"])
                        / merged["prod_calc"]
                        * 100
                    )
                    max_diff = merged["prod_diff_pct"].abs().max()
                    if max_diff > 1:
                        self.warnings.append(
                            f"Productivity deviation up to {max_diff:.1f}%"
                        )
                        print(
                            f"  ⚠️  Max deviation: {max_diff:.2f}%"
                        )
                    else:
                        print("  ✅ Stored vs calculated within 1%")
            else:
                print("  ℹ️  No productivity data to validate")

            # Income per worker
            print("\n🔗 Income per worker consistency")
            print("-" * 80)

            income = derived_df[
                (derived_df["metric_id"] == "income_per_worker_gbp")
                & (derived_df["data_type"] == "forecast")
            ][["region_code", "period", "value"]].rename(
                columns={"value": "inc_stored"}
            )

            if (
                not income.empty
                and "gdhi_total_mn_gbp" in pivot.columns
                and "emp_total_jobs" in pivot.columns
            ):
                merged = pivot.merge(
                    income, on=["region_code", "period"], how="inner"
                )
                if not merged.empty:
                    merged["inc_calc"] = (
                        merged["gdhi_total_mn_gbp"]
                        / merged["emp_total_jobs"]
                        * 1e6
                    )
                    merged["inc_diff_pct"] = (
                        (merged["inc_stored"] - merged["inc_calc"])
                        / merged["inc_calc"]
                        * 100
                    )
                    max_diff = merged["inc_diff_pct"].abs().max()
                    if max_diff > 1:
                        self.warnings.append(
                            f"Income/worker deviation up to {max_diff:.1f}%"
                        )
                        print(
                            f"  ⚠️  Max deviation: {max_diff:.2f}%"
                        )
                    else:
                        print("  ✅ Stored vs calculated within 1%")
            else:
                print("  ℹ️  No income_per_worker data to validate")
        else:
            print("\nℹ️  Derived table empty — skipping productivity/income checks")

    # ======================================================================
    # TERMINAL YEAR DIAGNOSTICS (2050)
    # ======================================================================

    def check_terminal_year_diagnostics(
        self, base_df: pd.DataFrame, derived_df: pd.DataFrame
    ) -> None:
        """2050 snapshot: UK totals, top regions, sanity checks."""
        print("\n" + "=" * 80)
        print("TERMINAL YEAR DIAGNOSTICS (2050)")
        print("=" * 80)

        fcst_2050 = base_df[
            (base_df["data_type"] == "forecast") & (base_df["period"] == 2050)
        ]
        if fcst_2050.empty:
            print("\n  ⚠️  No 2050 forecasts found")
            return

        pivot = (
            fcst_2050.pivot_table(
                index=["region_code", "region_name"],
                columns="metric_id",
                values="value",
            )
            .reset_index()
        )

        print(f"\n📊 2050 Snapshot ({len(pivot)} regions)")
        print("-" * 80)

        print("\n  UK Totals (sum of ITL3):")
        for metric in self.additive_metrics:
            if metric in pivot.columns:
                total = pivot[metric].sum()
                if "gva" in metric or "gdhi" in metric:
                    print(f"    {metric}: £{total:,.0f}m")
                else:
                    print(f"    {metric}: {total:,.0f}")

        print("\n  📍 Top 5 by GVA (if available):")
        if "nominal_gva_mn_gbp" in pivot.columns:
            top_gva = pivot.nlargest(5, "nominal_gva_mn_gbp")
            for _, row in top_gva.iterrows():
                print(
                    f"    {row['region_name']}: £{row['nominal_gva_mn_gbp']:,.0f}m"
                )

        print("\n  📍 Top 5 by GDHI/head (if available):")
        if "gdhi_per_head_gbp" in pivot.columns:
            top_gdhi = pivot.nlargest(5, "gdhi_per_head_gbp")
            for _, row in top_gdhi.iterrows():
                print(
                    f"    {row['region_name']}: £{row['gdhi_per_head_gbp']:,.0f}"
                )

        print("\n  🚨 Sanity Checks:")
        issues_2050: List[str] = []

        if "nominal_gva_mn_gbp" in pivot.columns:
            max_gva = pivot["nominal_gva_mn_gbp"].max()
            if max_gva > 200_000:
                issues_2050.append(
                    f"Max regional GVA >£200bn: £{max_gva:,.0f}m"
                )

        if "gdhi_per_head_gbp" in pivot.columns:
            max_gdhi = pivot["gdhi_per_head_gbp"].max()
            min_gdhi = pivot["gdhi_per_head_gbp"].min()
            if max_gdhi > 200_000:
                issues_2050.append(
                    f"Max GDHI/head >£200k: £{max_gdhi:,.0f}"
                )
            if min_gdhi < 5_000:
                issues_2050.append(
                    f"Min GDHI/head <£5k: £{min_gdhi:,.0f}"
                )

        if "population_total" in pivot.columns:
            max_pop = pivot["population_total"].max()
            if max_pop > 2_000_000:
                issues_2050.append(
                    f"Max regional pop >2m: {max_pop:,.0f}"
                )

        if issues_2050:
            for issue in issues_2050:
                self.warnings.append(f"2050: {issue}")
                print(f"    ⚠️  {issue}")
        else:
            print("    ✅ All 2050 values within sanity bounds")

    # ======================================================================
    # REGIONAL PLAUSIBILITY
    # ======================================================================

    def check_regional_plausibility(self, base_df: pd.DataFrame) -> None:
        """Regional sanity: CAGRs and sample values."""
        print("\n" + "=" * 80)
        print("REGIONAL PLAUSIBILITY")
        print("=" * 80)

        print("\n📍 Regional CAGRs (2025–2030) - extreme check")

        for metric in ["nominal_gva_mn_gbp", "emp_total_jobs", "population_total"]:
            m_data = base_df[
                (base_df["metric_id"] == metric)
                & (base_df["data_type"] == "forecast")
                & (base_df["period"].isin([2025, 2030]))
            ].pivot_table(
                index="region_code", columns="period", values="value"
            )

            if (
                m_data.empty
                or 2025 not in m_data.columns
                or 2030 not in m_data.columns
            ):
                continue

            m_data["cagr"] = ((m_data[2030] / m_data[2025]) ** 0.2 - 1) * 100
            extreme = m_data[(m_data["cagr"] > 10) | (m_data["cagr"] < -5)]

            if not extreme.empty:
                self.warnings.append(
                    f"{metric}: {len(extreme)} extreme CAGRs"
                )
                print(f"  ⚠️  {metric}: {len(extreme)} extreme")
            else:
                print(f"  ✅ {metric}")

        print("\n📍 Sample regional 2025 values")
        fcst_2025 = base_df[
            (base_df["data_type"] == "forecast") & (base_df["period"] == 2025)
        ]

        if "gdhi_per_head_gbp" in fcst_2025["metric_id"].values:
            gdhi_data = fcst_2025[
                fcst_2025["metric_id"] == "gdhi_per_head_gbp"
            ]
            top_3 = gdhi_data.nlargest(3, "value")
            for _, row in top_3.iterrows():
                val = row["value"]
                if val < 10_000:
                    self.warnings.append(
                        f"{row['region_name']} 2025 GDHI/head suspicious: £{val:,.0f}"
                    )
                    print(f"  ⚠️  {row['region_name']}: £{val:,.0f}")
                else:
                    print(f"  ✅ {row['region_name']}: £{val:,.0f}")

    # ======================================================================
    # REPORT
    # ======================================================================

    def generate_report(self) -> int:
        """Summarise issues + warnings and return exit code."""
        print("\n" + "=" * 80)
        print("QA SUMMARY")
        print("=" * 80)

        print(f"\n🔍 Critical issues: {len(self.issues)}")
        for issue in self.issues[:15]:
            print(f"  ❌ {issue}")
        if len(self.issues) > 15:
            print(f"  ... and {len(self.issues) - 15} more")

        print(f"\n⚠️  Warnings: {len(self.warnings)}")
        for w in self.warnings[:10]:
            print(f"  ⚠️  {w}")
        if len(self.warnings) > 10:
            print(f"  ... and {len(self.warnings) - 10} more")

        print("\n" + "=" * 80)
        if len(self.issues) == 0:
            print("✅ ITL3 FORECAST QA V7.1: PASSED — Safe for LAD")
            print("=" * 80)
            print("\nReady to proceed to LAD.")
            exit_code = 0
        else:
            print("❌ ITL3 FORECAST QA V7.1: FAILED — DO NOT RUN LAD")
            print("=" * 80)
            print(f"\n{len(self.issues)} critical issues. Fix before LAD.")
            exit_code = 1

        summary = {
            "level": "ITL3",
            "version": "7.1",
            "timestamp": datetime.now().isoformat(),
            "status": "PASSED" if exit_code == 0 else "FAILED",
            "exit_code": exit_code,
            "architecture": {
                "base_table": "gold.itl3_forecast",
                "base_metrics": self.expected_base_metrics,
                "derived_table": "gold.itl3_derived",
                "derived_metrics": self.derived_metrics,
                "reconciles_to": "gold.itl2_forecast",
            },
            "lad_gate": {
                "required_horizon": self.required_lad_horizon,
                "expected_regions": self.expected_itl3_count,
                "can_run_lad": exit_code == 0,
            },
            "stats": {
                "base_metrics_expected": len(self.expected_base_metrics),
                "derived_metrics_expected": len(self.derived_metrics),
                "critical_issues": len(self.issues),
                "warnings": len(self.warnings),
            },
            "critical_issues": self.issues,
            "warnings": self.warnings,
        }

        output_path = Path("data/qa/itl3_qa_summary.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n📄 Summary: {output_path}")
        print(f"📤 Exit code: {exit_code}")

        return exit_code

    # ======================================================================
    # RUNNER
    # ======================================================================

    def run_all_checks(self) -> int:
        """Execute full QA suite."""
        print("\n" + "=" * 80)
        print("ITL3 FORECAST QA V7.1 — LAD GATEKEEPER")
        print("=" * 80)
        print(f"Database: {self.db_path}")
        print(f"Tolerance: {self.tolerance * 100:.1f}%")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nArchitecture:")
        print(
            f"  Base table: gold.itl3_forecast ({len(self.expected_base_metrics)} metrics)"
        )
        print(
            f"  Derived table: gold.itl3_derived ({len(self.derived_metrics)} metrics)"
        )
        print("  Reconciles to: gold.itl2_forecast")
        print(f"  Gates: LAD (requires {self.required_lad_horizon} horizon)")

        print("\n📥 Loading data...")
        base_df, derived_df, itl2_df, ci_df = self.load_data()
        print(f"  ✅ Base (ITL3): {len(base_df):,} rows")
        print(f"  ✅ Derived: {len(derived_df):,} rows")
        print(f"  ✅ ITL2 parent: {len(itl2_df):,} rows")
        print(f"  ✅ CIs: {len(ci_df):,} rows")

        # Core checks
        self.check_base_metric_coverage(base_df)
        self.check_derived_table(derived_df)
        self.check_data_quality(base_df)
        self.check_lad_readiness(base_df)
        self.check_reconciliation(base_df, itl2_df)
        self.check_gdhi_per_head(base_df)
        self.check_rate_bounds(base_df)
        self.check_historical_continuity(base_df)
        self.check_confidence_intervals(ci_df)
        self.check_ci_diagnostics_per_metric(base_df)
        self.check_growth_rate_bounds(base_df)
        self.check_cross_metric_coherence(base_df, derived_df)
        self.check_terminal_year_diagnostics(base_df, derived_df)
        self.check_regional_plausibility(base_df)

        exit_code = self.generate_report()
        self.conn.close()
        return exit_code


def main():
    qa = ITL3ForecastQA(
        db_path="data/lake/warehouse.duckdb",
        tolerance=0.01,
    )
    exit_code = qa.run_all_checks()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()