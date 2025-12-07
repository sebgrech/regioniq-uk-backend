#!/usr/bin/env python3
"""
UK Macro Forecast QA Script V3.5
=================================
Validates UK-level macro forecasts including labour market rates.
Serves as the anchor for the entire ITL cascade.

V3.5 CHANGES:
- uk_population_16_64 (working age) added to expected metrics
- Derived metrics (productivity, income_per_worker) now in gold.uk_macro_derived
- GDHI per head validation (derived from total/pop in base table)
- Dual-table architecture validation
- Per-metric continuity checks (handles mixed-vintage data)

Part of the weekly automated cascade: Macro → ITL1 → ITL2 → ITL3

Exit codes:
    0: All critical checks passed (warnings allowed) - proceed to ITL1
    1: Fatal errors detected - do not proceed to ITL1

Usage:
    python3 scripts/forecast/QA/Macro_Broadbased_QA.py
    
Output:
    - Console: Human-readable report
    - JSON: data/qa/macro_qa_summary.json (for automation)
"""

import sys
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json


class MacroForecastQA:
    """Comprehensive QA for UK macro forecasts V3.5"""
    
    def __init__(self, 
                 db_path: str = "data/lake/warehouse.duckdb",
                 tolerance: float = 0.01):
        self.db_path = db_path
        self.tolerance = tolerance
        self.conn = duckdb.connect(db_path, read_only=True)
        
        # Expected UK metrics in BASE table (V3.5)
        self.expected_base_metrics = [
            'uk_nominal_gva_mn_gbp',
            'uk_gdhi_total_mn_gbp',
            'uk_gdhi_per_head_gbp',
            'uk_emp_total_jobs',
            'uk_population_total',
            'uk_population_16_64',
            'uk_employment_rate_pct',
            'uk_unemployment_rate_pct'
        ]
        
        # Expected metrics in DERIVED table (V3.5 - separate table)
        self.expected_derived_metrics = [
            'uk_productivity_gbp_per_job',
            'uk_income_per_worker_gbp'
        ]
        
        # VAR system definitions
        self.var_systems = {
            'gva_employment': {
                'metrics': ['uk_nominal_gva_mn_gbp', 'uk_emp_total_jobs'],
                'min_correlation': 0.5,
                'name': 'GVA-Employment'
            },
            'labour_market_rates': {
                'metrics': ['uk_employment_rate_pct', 'uk_unemployment_rate_pct'],
                'min_correlation': 0.3,
                'name': 'Employment-Unemployment Rates'
            }
        }
        
        # Derived metric formulas (for validation)
        self.derived_formulas = {
            'uk_productivity_gbp_per_job': {
                'numerator': 'uk_nominal_gva_mn_gbp',
                'denominator': 'uk_emp_total_jobs',
                'scale': 1_000_000,
                'table': 'derived'
            },
            'uk_income_per_worker_gbp': {
                'numerator': 'uk_gdhi_total_mn_gbp',
                'denominator': 'uk_emp_total_jobs',
                'scale': 1_000_000,
                'table': 'derived'
            },
            'uk_gdhi_per_head_gbp': {
                'numerator': 'uk_gdhi_total_mn_gbp',
                'denominator': 'uk_population_total',
                'scale': 1_000_000,
                'table': 'base'
            }
        }
        
        # Rate metrics (must be bounded 0-100%)
        self.rate_metrics = {
            'uk_employment_rate_pct': {'min': 0.0, 'max': 100.0, 'expected_range': (60, 80)},
            'uk_unemployment_rate_pct': {'min': 0.0, 'max': 100.0, 'expected_range': (2, 10)}
        }
        
        # Growth caps (must match forecast script)
        self.growth_caps = {
            'uk_nominal_gva_mn_gbp': {'min': -0.03, 'max': 0.045},
            'uk_gdhi_total_mn_gbp': {'min': -0.03, 'max': 0.045},
            'uk_gdhi_per_head_gbp': {'min': -0.03, 'max': 0.04},
            'uk_emp_total_jobs': {'min': -0.02, 'max': 0.015},
            'uk_population_total': {'min': 0.0, 'max': 0.008},
            'uk_population_16_64': {'min': -0.005, 'max': 0.006},
        }
        
        # Sanity caps
        self.sanity_caps = {
            'uk_nominal_gva_mn_gbp': 12_000_000,
            'uk_gdhi_total_mn_gbp': 10_000_000,
            'uk_gdhi_per_head_gbp': 80_000,
            'uk_emp_total_jobs': 42_000_000,
            'uk_population_total': 85_000_000,
            'uk_population_16_64': 55_000_000,
        }
        
        self.forecast_years = [2025, 2030, 2040, 2050]
        self.issues = []
        self.warnings = []
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load UK macro forecasts from both tables"""
        
        # Load base forecasts
        base_query = """
        SELECT 
            region_code,
            metric_id,
            period,
            value,
            data_type,
            ci_lower,
            ci_upper
        FROM gold.uk_macro_forecast
        WHERE period >= 2000
        ORDER BY metric_id, period
        """
        base_df = self.conn.execute(base_query).df()
        
        # Load derived metrics (V3.5 - separate table)
        derived_df = pd.DataFrame()
        try:
            derived_query = """
            SELECT 
                region_code,
                metric_id,
                period,
                value,
                data_type,
                ci_lower,
                ci_upper,
                method,
                formula
            FROM gold.uk_macro_derived
            WHERE period >= 2000
            ORDER BY metric_id, period
            """
            derived_df = self.conn.execute(derived_query).df()
        except Exception as e:
            self.warnings.append(f"Could not load gold.uk_macro_derived: {e}")
        
        # Extract CIs from base table
        ci_df = base_df[
            (base_df['data_type'] == 'forecast') & 
            (base_df['ci_lower'].notna())
        ][['metric_id', 'period', 'ci_lower', 'ci_upper']].copy()
        
        return base_df, derived_df, ci_df
    
    def check_table_structure(self) -> None:
        """Validate V3.5 dual-table architecture exists"""
        
        print("\n" + "="*80)
        print("TABLE STRUCTURE (V3.5 Architecture)")
        print("="*80)
        
        # Check base table
        try:
            base_count = self.conn.execute(
                "SELECT COUNT(*) FROM gold.uk_macro_forecast"
            ).fetchone()[0]
            print(f"\n✅ gold.uk_macro_forecast: {base_count:,} rows")
        except Exception as e:
            self.issues.append(f"CRITICAL: Base table missing - {e}")
            print(f"\n❌ gold.uk_macro_forecast: MISSING")
        
        # Check derived table
        try:
            derived_count = self.conn.execute(
                "SELECT COUNT(*) FROM gold.uk_macro_derived"
            ).fetchone()[0]
            print(f"✅ gold.uk_macro_derived: {derived_count:,} rows")
        except Exception as e:
            self.warnings.append(f"Derived table not found (may be expected if no derived metrics)")
            print(f"⚠️  gold.uk_macro_derived: Not found")
        
        # Check view
        try:
            self.conn.execute("SELECT 1 FROM gold.uk_macro_forecast_only LIMIT 1")
            print(f"✅ gold.uk_macro_forecast_only view exists")
        except:
            self.warnings.append("Forecast-only view missing")
            print(f"⚠️  gold.uk_macro_forecast_only: Missing")
    
    def check_data_quality(self, base_df: pd.DataFrame) -> None:
        """Check for basic data quality issues"""
        
        print("\n" + "="*80)
        print("DATA QUALITY CHECKS")
        print("="*80)
        
        uk_code = 'K02000001'
        unique_regions = base_df['region_code'].unique()
        
        print(f"\n🇬🇧 Region codes: {unique_regions}")
        if uk_code not in unique_regions:
            self.issues.append(f"CRITICAL: UK region code {uk_code} not found")
            print(f"  ❌ CRITICAL: UK code missing")
        elif len(unique_regions) > 1:
            self.warnings.append(f"Multiple region codes found: {unique_regions}")
            print(f"  ⚠️  Multiple regions (expected only UK)")
        else:
            print(f"  ✅ Correct UK region code")
        
        # Null checks
        required_cols = ['metric_id', 'period', 'value', 'data_type']
        null_counts = base_df[required_cols].isna().sum()
        total_nulls = null_counts.sum()
        
        print(f"\n📋 Null values: {total_nulls}")
        if total_nulls > 0:
            self.issues.append(f"CRITICAL: {total_nulls} null values in core columns")
            print(f"  ❌ CRITICAL: Nulls detected")
            for col, count in null_counts[null_counts > 0].items():
                print(f"     {col}: {count}")
        else:
            print(f"  ✅ No nulls in core columns")
        
        # Duplicates
        dups = base_df.duplicated(subset=['metric_id', 'period'], keep=False)
        n_dups = dups.sum()
        
        print(f"\n📑 Duplicates: {n_dups}")
        if n_dups > 0:
            self.issues.append(f"CRITICAL: {n_dups} duplicate rows")
            print(f"  ❌ CRITICAL: Duplicates found")
        else:
            print(f"  ✅ No duplicates")
        
        # Negative values
        negatives = base_df[base_df['value'] < 0]
        n_neg = len(negatives)
        
        print(f"\n🔢 Negative values: {n_neg}")
        if n_neg > 0:
            neg_by_metric = negatives.groupby('metric_id').size()
            self.issues.append(f"CRITICAL: {n_neg} negative values")
            print(f"  ❌ CRITICAL: Negatives detected:")
            for m, c in neg_by_metric.items():
                print(f"     {m}: {c}")
        else:
            print(f"  ✅ No negative values")
    
    def check_metric_coverage(self, base_df: pd.DataFrame, derived_df: pd.DataFrame) -> None:
        """Check that all expected metrics are present in correct tables"""
        
        print("\n" + "="*80)
        print("METRIC COVERAGE (V3.5 Schema)")
        print("="*80)
        
        # Base table metrics
        base_metrics = set(base_df['metric_id'].unique())
        expected_base = set(self.expected_base_metrics)
        
        print(f"\n📊 BASE TABLE (gold.uk_macro_forecast)")
        print(f"   Expected: {len(expected_base)}, Found: {len(base_metrics)}")
        
        missing_base = expected_base - base_metrics
        if missing_base:
            self.issues.append(f"CRITICAL: Missing base metrics: {missing_base}")
            print(f"  ❌ CRITICAL: Missing {len(missing_base)} metrics:")
            for m in sorted(missing_base):
                print(f"     - {m}")
        else:
            print(f"  ✅ All expected base metrics present")
        
        # Derived table metrics
        print(f"\n📊 DERIVED TABLE (gold.uk_macro_derived)")
        if derived_df.empty:
            self.warnings.append("No derived metrics found")
            print(f"  ⚠️  Table empty or missing")
        else:
            derived_metrics = set(derived_df['metric_id'].unique())
            expected_derived = set(self.expected_derived_metrics)
            
            print(f"   Expected: {len(expected_derived)}, Found: {len(derived_metrics)}")
            
            missing_derived = expected_derived - derived_metrics
            if missing_derived:
                self.warnings.append(f"Missing derived metrics: {missing_derived}")
                print(f"  ⚠️  Missing: {missing_derived}")
            else:
                print(f"  ✅ All expected derived metrics present")
    
    def check_vintage_summary(self, base_df: pd.DataFrame) -> Dict[str, Dict]:
        """Summarize data vintages per metric (V3.5 mixed-vintage support)"""
        
        print("\n" + "="*80)
        print("DATA VINTAGE SUMMARY (Mixed-Vintage Architecture)")
        print("="*80)
        
        vintage_info = {}
        
        print(f"\n{'Metric':<35} {'Last Hist':>10} {'First Fcst':>11} {'Gap':>5}")
        print("-" * 65)
        
        for metric in sorted(base_df['metric_id'].unique()):
            metric_data = base_df[base_df['metric_id'] == metric]
            hist = metric_data[metric_data['data_type'] == 'historical']
            fcst = metric_data[metric_data['data_type'] == 'forecast']
            
            last_hist = int(hist['period'].max()) if not hist.empty else None
            first_fcst = int(fcst['period'].min()) if not fcst.empty else None
            
            if last_hist and first_fcst:
                gap = first_fcst - last_hist
                gap_str = f"{gap:+d}" if gap != 1 else "✓"
            else:
                gap_str = "?"
            
            vintage_info[metric] = {
                'last_historical': last_hist,
                'first_forecast': first_fcst
            }
            
            print(f"{metric:<35} {last_hist or 'N/A':>10} {first_fcst or 'N/A':>11} {gap_str:>5}")
        
        # Check for any gaps (should be exactly 1 year between last hist and first fcst)
        problematic = []
        for metric, info in vintage_info.items():
            if info['last_historical'] and info['first_forecast']:
                gap = info['first_forecast'] - info['last_historical']
                if gap != 1:
                    problematic.append(f"{metric}: gap={gap}")
        
        if problematic:
            print(f"\n  ⚠️  Non-standard gaps detected:")
            for p in problematic:
                print(f"     {p}")
            self.warnings.append(f"Non-standard vintage gaps: {problematic}")
        else:
            print(f"\n  ✅ All metrics have contiguous historical→forecast")
        
        return vintage_info
    
    def check_var_systems(self, base_df: pd.DataFrame) -> Dict[str, any]:
        """Validate all VAR/VECM systems"""
        
        print("\n" + "="*80)
        print("VAR/VECM SYSTEM VALIDATION")
        print("="*80)
        
        var_results = {}
        
        for system_name, system_config in self.var_systems.items():
            print(f"\n📊 {system_config['name']} System")
            print("-" * 60)
            
            metrics = system_config['metrics']
            min_corr = system_config['min_correlation']
            
            available = [m for m in metrics if m in base_df['metric_id'].values]
            print(f"  Metrics: {len(available)}/{len(metrics)}")
            
            if len(available) < 2:
                self.warnings.append(f"{system_name}: Incomplete - missing metrics")
                print(f"  ⚠️  System incomplete")
                var_results[system_name] = {'enabled': False, 'correlation': None}
                continue
            
            metric1_data = base_df[
                (base_df['metric_id'] == metrics[0]) &
                (base_df['data_type'] == 'forecast')
            ].sort_values('period')
            
            metric2_data = base_df[
                (base_df['metric_id'] == metrics[1]) &
                (base_df['data_type'] == 'forecast')
            ].sort_values('period')
            
            if metric1_data.empty or metric2_data.empty:
                self.warnings.append(f"{system_name}: Missing forecast data")
                print(f"  ⚠️  Missing forecast data")
                var_results[system_name] = {'enabled': False, 'correlation': None}
                continue
            
            merged = metric1_data.merge(
                metric2_data,
                on='period',
                suffixes=('_1', '_2')
            )
            
            if len(merged) > 5:
                correlation = merged['value_1'].corr(merged['value_2'])
                print(f"  Correlation: {correlation:.3f}")
                
                if abs(correlation) < abs(min_corr):
                    self.issues.append(
                        f"CRITICAL: {system_name} correlation = {correlation:.3f} (expected >{min_corr})"
                    )
                    print(f"  ❌ CRITICAL: Low correlation")
                elif abs(correlation) < abs(min_corr) + 0.15:
                    self.warnings.append(f"{system_name} correlation = {correlation:.3f} (borderline)")
                    print(f"  ⚠️  Borderline correlation")
                else:
                    print(f"  ✅ Strong relationship")
                
                var_results[system_name] = {'enabled': True, 'correlation': correlation}
            else:
                var_results[system_name] = {'enabled': False, 'correlation': None}
        
        return var_results
    
    def check_rate_bounds(self, base_df: pd.DataFrame) -> None:
        """Validate rate metrics are within 0-100% bounds"""
        
        print("\n" + "="*80)
        print("RATE METRIC BOUNDS VALIDATION")
        print("="*80)
        
        for rate_metric, bounds in self.rate_metrics.items():
            print(f"\n📊 {rate_metric}")
            print("-" * 60)
            
            rate_data = base_df[base_df['metric_id'] == rate_metric]
            
            if rate_data.empty:
                print(f"  ⚠️  No data found")
                continue
            
            below_min = rate_data[rate_data['value'] < bounds['min']]
            above_max = rate_data[rate_data['value'] > bounds['max']]
            
            if not below_min.empty:
                self.issues.append(
                    f"CRITICAL: {rate_metric} has {len(below_min)} values < {bounds['min']}%"
                )
                print(f"  ❌ {len(below_min)} values below {bounds['min']}%")
            
            if not above_max.empty:
                self.issues.append(
                    f"CRITICAL: {rate_metric} has {len(above_max)} values > {bounds['max']}%"
                )
                print(f"  ❌ {len(above_max)} values above {bounds['max']}%")
            
            if below_min.empty and above_max.empty:
                print(f"  ✅ All values within [{bounds['min']}, {bounds['max']}]%")
            
            forecast_data = rate_data[rate_data['data_type'] == 'forecast']
            if not forecast_data.empty:
                min_val = forecast_data['value'].min()
                max_val = forecast_data['value'].max()
                mean_val = forecast_data['value'].mean()
                
                exp_min, exp_max = bounds['expected_range']
                print(f"  Forecast range: {min_val:.1f}% - {max_val:.1f}% (mean: {mean_val:.1f}%)")
                
                if min_val < exp_min or max_val > exp_max:
                    self.warnings.append(
                        f"{rate_metric}: Outside expected range [{exp_min}, {exp_max}]%"
                    )
                    print(f"  ⚠️  Outside typical range [{exp_min}, {exp_max}]%")
                else:
                    print(f"  ✅ Within expected range")
    
    def check_derived_metrics(self, base_df: pd.DataFrame, derived_df: pd.DataFrame) -> None:
        """Validate derived metrics calculated correctly"""
        
        print("\n" + "="*80)
        print("DERIVED METRICS VALIDATION")
        print("="*80)
        
        for derived_metric, formula in self.derived_formulas.items():
            numerator = formula['numerator']
            denominator = formula['denominator']
            scale = formula['scale']
            table = formula['table']
            
            print(f"\n📐 {derived_metric}")
            print(f"   Formula: {numerator} / {denominator} × {scale:,}")
            print(f"   Table: gold.uk_macro_{table}")
            print("-" * 60)
            
            # Get source components from base table
            base_pivot = base_df[
                (base_df['metric_id'].isin([numerator, denominator])) &
                (base_df['data_type'] == 'forecast')
            ].pivot_table(
                index='period',
                columns='metric_id',
                values='value'
            )
            
            if numerator not in base_pivot.columns or denominator not in base_pivot.columns:
                self.warnings.append(f"{derived_metric}: Missing component data")
                print(f"  ⚠️  Missing components")
                continue
            
            # Calculate expected
            base_pivot['expected'] = (base_pivot[numerator] / base_pivot[denominator]) * scale
            
            # Get actual from appropriate table
            if table == 'base':
                actual_data = base_df[base_df['metric_id'] == derived_metric]
            else:
                actual_data = derived_df[derived_df['metric_id'] == derived_metric] if not derived_df.empty else pd.DataFrame()
            
            if actual_data.empty:
                self.warnings.append(f"{derived_metric}: No actual values found")
                print(f"  ⚠️  No values in {table} table")
                continue
            
            actual_pivot = actual_data[actual_data['data_type'] == 'forecast'].set_index('period')['value']
            base_pivot['actual'] = actual_pivot
            
            base_pivot['error_pct'] = (
                abs(base_pivot['actual'] - base_pivot['expected']) / base_pivot['expected'] * 100
            )
            
            for year in [2025, 2030, 2050]:
                if year not in base_pivot.index:
                    continue
                
                error = base_pivot.loc[year, 'error_pct']
                
                if pd.notna(error):
                    if error > self.tolerance * 100:
                        self.issues.append(
                            f"{derived_metric} {year}: {error:.2f}% calculation error"
                        )
                        print(f"  ❌ {year}: Error {error:.2f}% (CRITICAL)")
                    else:
                        print(f"  ✅ {year}: Error {error:.3f}%")
    
    def check_population_metrics(self, base_df: pd.DataFrame) -> None:
        """V3.5: Check working age population against total"""
        
        print("\n" + "="*80)
        print("POPULATION METRICS (V3.5)")
        print("="*80)
        
        pop_data = base_df[base_df['metric_id'].isin([
            'uk_population_total', 'uk_population_16_64'
        ])].pivot_table(
            index='period',
            columns='metric_id',
            values='value'
        )
        
        if 'uk_population_total' not in pop_data.columns:
            print("\n  ⚠️  Total population missing")
            return
        
        if 'uk_population_16_64' not in pop_data.columns:
            print("\n  ⚠️  Working age population (16-64) missing")
            self.warnings.append("Working age population metric missing")
            return
        
        # Check working age is always less than total
        pop_data['ratio'] = pop_data['uk_population_16_64'] / pop_data['uk_population_total']
        
        invalid = pop_data[pop_data['ratio'] > 1]
        if not invalid.empty:
            self.issues.append(f"CRITICAL: Working age > total population in {len(invalid)} years")
            print(f"\n  ❌ CRITICAL: Working age exceeds total in {len(invalid)} years")
        else:
            print(f"\n  ✅ Working age always < total population")
        
        # Check ratio is reasonable (typically 60-70%)
        forecast_ratios = pop_data[pop_data.index >= 2025]['ratio']
        if not forecast_ratios.empty:
            min_ratio = forecast_ratios.min()
            max_ratio = forecast_ratios.max()
            
            print(f"  Working age ratio: {min_ratio:.1%} - {max_ratio:.1%}")
            
            if min_ratio < 0.50:
                self.warnings.append(f"Working age ratio unexpectedly low: {min_ratio:.1%}")
                print(f"  ⚠️  Ratio below 50%")
            elif max_ratio > 0.75:
                self.warnings.append(f"Working age ratio unexpectedly high: {max_ratio:.1%}")
                print(f"  ⚠️  Ratio above 75%")
            else:
                print(f"  ✅ Ratio within expected range (50-75%)")
    
    def check_growth_caps(self, base_df: pd.DataFrame) -> None:
        """Validate forecasts respect growth caps"""
        
        print("\n" + "="*80)
        print("GROWTH CAP VALIDATION")
        print("="*80)
        
        violations = []
        
        for metric, caps in self.growth_caps.items():
            metric_data = base_df[
                (base_df['metric_id'] == metric) &
                (base_df['data_type'] == 'forecast')
            ].sort_values('period')
            
            if len(metric_data) < 2:
                continue
            
            values = metric_data['value'].values
            growth_rates = np.diff(values) / values[:-1]
            
            min_g, max_g = caps['min'], caps['max']
            
            breaches_min = np.sum(growth_rates < min_g - 0.001)
            breaches_max = np.sum(growth_rates > max_g + 0.001)
            
            if breaches_min > 0 or breaches_max > 0:
                violations.append(f"{metric}: {breaches_min} below min, {breaches_max} above max")
        
        print(f"\n📊 Growth cap violations: {len(violations)}")
        if violations:
            for v in violations:
                self.warnings.append(f"Growth cap breach: {v}")
                print(f"  ⚠️  {v}")
        else:
            print(f"  ✅ All forecasts within growth caps")
    
    def check_sanity_caps(self, base_df: pd.DataFrame) -> None:
        """Validate forecasts don't exceed sanity caps"""
        
        print("\n" + "="*80)
        print("SANITY CAP VALIDATION")
        print("="*80)
        
        breaches = []
        
        for metric, cap in self.sanity_caps.items():
            metric_data = base_df[base_df['metric_id'] == metric]
            exceeds = metric_data[metric_data['value'] > cap]
            
            if not exceeds.empty:
                max_val = exceeds['value'].max()
                breaches.append(f"{metric}: max {max_val:,.0f} > cap {cap:,.0f}")
        
        print(f"\n📊 Sanity cap breaches: {len(breaches)}")
        if breaches:
            for b in breaches:
                self.issues.append(f"CRITICAL: Sanity breach - {b}")
                print(f"  ❌ {b}")
        else:
            print(f"  ✅ All values within sanity caps")
    
    def check_confidence_intervals(self, ci_df: pd.DataFrame, base_df: pd.DataFrame) -> None:
        """Validate confidence interval quality"""
        
        print("\n" + "="*80)
        print("CONFIDENCE INTERVAL VALIDATION")
        print("="*80)
        
        if ci_df.empty:
            self.warnings.append("No confidence intervals found")
            print("\n  ⚠️  No confidence intervals")
            return
        
        print(f"\n📊 Coverage: {len(ci_df):,} metric-period combinations")
        
        invalid_order = ci_df[ci_df['ci_lower'] > ci_df['ci_upper']]
        if not invalid_order.empty:
            self.issues.append(f"{len(invalid_order)} CIs with invalid ordering")
            print(f"  ❌ {len(invalid_order)} CIs with lower > upper")
        else:
            print(f"  ✅ All CIs properly ordered")
        
        forecast_df = base_df[base_df['data_type'] == 'forecast'][
            ['metric_id', 'period', 'value']
        ]
        
        ci_merged = ci_df.merge(
            forecast_df,
            on=['metric_id', 'period'],
            how='left'
        )
        
        ci_merged['ci_width'] = ci_merged['ci_upper'] - ci_merged['ci_lower']
        with np.errstate(divide='ignore', invalid='ignore'):
            ci_merged['cv'] = ci_merged['ci_width'] / (2 * ci_merged['value'])
        
        print(f"\n📊 CI width statistics (coefficient of variation):")
        for metric in sorted(ci_merged['metric_id'].unique()):
            metric_cv = ci_merged[ci_merged['metric_id'] == metric]['cv']
            mean_cv = metric_cv.mean()
            
            if np.isfinite(mean_cv) and mean_cv < 1000:
                print(f"  {metric:35s}: CV mean={mean_cv:.3f}")
            else:
                print(f"  {metric:35s}: CV mean=N/A")
        
        wide_cis = ci_merged[ci_merged['cv'] > 0.5]
        if not wide_cis.empty:
            self.warnings.append(f"{len(wide_cis)} forecasts with CV >50%")
            print(f"\n  ⚠️  {len(wide_cis)} very wide CIs (CV >50%)")
    
    def check_historical_continuity(self, base_df: pd.DataFrame) -> None:
        """Check that forecasts connect smoothly with history (per-metric)"""
        
        print("\n" + "="*80)
        print("HISTORICAL CONTINUITY CHECKS (Per-Metric)")
        print("="*80)
        
        # Level metrics only (not rates or derived ratios)
        level_metrics = [
            'uk_nominal_gva_mn_gbp',
            'uk_gdhi_total_mn_gbp',
            'uk_emp_total_jobs',
            'uk_population_total',
            'uk_population_16_64'
        ]
        
        print(f"\n{'Metric':<35} {'Transition':>15} {'Jump':>10}")
        print("-" * 65)
        
        extreme_jumps = 0
        
        for metric in level_metrics:
            metric_data = base_df[base_df['metric_id'] == metric].copy()
            hist = metric_data[metric_data['data_type'] == 'historical']
            fcst = metric_data[metric_data['data_type'] == 'forecast']
            
            if hist.empty or fcst.empty:
                print(f"{metric:<35} {'N/A':>15} {'N/A':>10}")
                continue
            
            last_hist_year = int(hist['period'].max())
            first_fcst_year = int(fcst['period'].min())
            
            # Get values at boundary
            val_hist = hist[hist['period'] == last_hist_year]['value'].iloc[0]
            val_fcst = fcst[fcst['period'] == first_fcst_year]['value'].iloc[0]
            
            jump = (val_fcst - val_hist) / val_hist * 100
            
            transition_str = f"{last_hist_year} → {first_fcst_year}"
            
            if abs(jump) > 15:
                extreme_jumps += 1
                self.warnings.append(f"{metric}: {jump:+.1f}% jump ({transition_str})")
                print(f"{metric:<35} {transition_str:>15} {jump:>+9.1f}% ⚠️")
            else:
                print(f"{metric:<35} {transition_str:>15} {jump:>+9.1f}% ✅")
        
        print("-" * 65)
        if extreme_jumps == 0:
            print(f"\n  ✅ All transitions smooth (<15% jump)")
        else:
            print(f"\n  ⚠️  {extreme_jumps} metrics with large transitions")
    
    def generate_report(self, var_results: Dict, vintage_info: Dict) -> int:
        """Generate final QA report and return exit code"""
        
        print("\n" + "="*80)
        print("QA SUMMARY")
        print("="*80)
        
        print(f"\n🔍 Issues (critical): {len(self.issues)}")
        if self.issues:
            for issue in self.issues[:10]:
                print(f"  ❌ {issue}")
            if len(self.issues) > 10:
                print(f"  ... and {len(self.issues) - 10} more")
        
        print(f"\n⚠️  Warnings (non-critical): {len(self.warnings)}")
        if self.warnings:
            for warning in self.warnings[:10]:
                print(f"  ⚠️  {warning}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more")
        
        print("\n" + "="*80)
        if len(self.issues) == 0:
            print("✅ UK MACRO FORECAST QA: PASSED")
            print("="*80)
            print("\nAll critical checks passed. Ready to proceed to ITL1.")
            if self.warnings:
                print(f"\nNote: {len(self.warnings)} warnings detected but are non-critical.")
            exit_code = 0
        else:
            print("❌ UK MACRO FORECAST QA: FAILED")
            print("="*80)
            print(f"\n{len(self.issues)} critical issues must be resolved.")
            print("⛔ DO NOT proceed to ITL1 until these are fixed.")
            exit_code = 1
        
        # Export structured summary
        summary = {
            'level': 'Macro',
            'version': '3.5',
            'timestamp': datetime.now().isoformat(),
            'status': 'PASSED' if exit_code == 0 else 'FAILED',
            'exit_code': exit_code,
            'architecture': {
                'base_table': 'gold.uk_macro_forecast',
                'derived_table': 'gold.uk_macro_derived',
                'base_metrics': self.expected_base_metrics,
                'derived_metrics': self.expected_derived_metrics
            },
            'vintage_info': vintage_info,
            'stats': {
                'region': 'UK',
                'base_metrics': len(self.expected_base_metrics),
                'derived_metrics': len(self.expected_derived_metrics),
                'var_systems': len(self.var_systems),
                'forecast_years_checked': self.forecast_years,
                'critical_issues': len(self.issues),
                'warnings': len(self.warnings)
            },
            'var_systems': var_results,
            'checks_performed': {
                'table_structure': 'V3.5 dual-table architecture',
                'data_quality': 'Nulls, duplicates, negatives',
                'metric_coverage': 'Base and derived tables',
                'vintage_summary': 'Per-metric historical/forecast boundaries',
                'var_systems': 'GVA-Employment + Labour rates',
                'rate_bounds': 'Employment/unemployment 0-100%',
                'population_metrics': 'Working age vs total',
                'derived_metrics': 'Productivity, per-head, income calculations',
                'growth_caps': 'Annual growth within bounds',
                'sanity_caps': 'Absolute value limits',
                'confidence_intervals': 'CI quality and width',
                'historical_continuity': 'Per-metric smooth transitions'
            },
            'critical_issues': self.issues,
            'warnings': self.warnings,
            'for_email': {
                'status_emoji': '✅' if exit_code == 0 else '❌',
                'status_text': 'Passed' if exit_code == 0 else 'Failed',
                'anchor_quality': 'reliable' if exit_code == 0 else 'questionable',
                'warning_summary': f"{len(self.warnings)} non-blocking" if self.warnings else "none"
            }
        }
        
        output_path = Path('data/qa/macro_qa_summary.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📄 Summary exported to: {output_path}")
        print(f"📤 Exit code: {exit_code}")
        
        return exit_code
    
    def run_all_checks(self) -> int:
        """Run complete QA suite and return exit code"""
        
        print("\n" + "="*80)
        print("UK MACRO FORECAST QA V3.5")
        print("="*80)
        print(f"Database: {self.db_path}")
        print(f"Tolerance: {self.tolerance * 100}%")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Role: Anchor for ITL cascade")
        print(f"V3.5: Dual-table architecture + uk_population_16_64 + mixed-vintage")
        
        # Check table structure first
        self.check_table_structure()
        
        # Load data
        print("\n📥 Loading data...")
        base_df, derived_df, ci_df = self.load_data()
        print(f"  ✅ Base table: {len(base_df):,} rows")
        print(f"  ✅ Derived table: {len(derived_df):,} rows")
        print(f"  ✅ CI records: {len(ci_df):,} rows")
        
        # Run checks
        self.check_data_quality(base_df)
        self.check_metric_coverage(base_df, derived_df)
        vintage_info = self.check_vintage_summary(base_df)
        var_results = self.check_var_systems(base_df)
        self.check_rate_bounds(base_df)
        self.check_population_metrics(base_df)
        self.check_derived_metrics(base_df, derived_df)
        self.check_growth_caps(base_df)
        self.check_sanity_caps(base_df)
        self.check_confidence_intervals(ci_df, base_df)
        self.check_historical_continuity(base_df)
        
        # Generate report
        exit_code = self.generate_report(var_results, vintage_info)
        
        self.conn.close()
        
        return exit_code


def main():
    """Main entry point"""
    qa = MacroForecastQA(
        db_path="data/lake/warehouse.duckdb",
        tolerance=0.01
    )
    
    exit_code = qa.run_all_checks()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()