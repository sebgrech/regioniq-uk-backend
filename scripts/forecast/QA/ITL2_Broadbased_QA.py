#!/usr/bin/env python3
"""
ITL2 Forecast QA Script V4.2
============================
Aligned with ITL2 Forecast Engine (aligned with ITL1 V3.5) two-table architecture.

BASE TABLE: gold.itl2_forecast (8 metrics)
  - 5 additive: nominal_gva_mn_gbp, gdhi_total_mn_gbp, emp_total_jobs, 
                population_total, population_16_64
  - 2 rates: employment_rate_pct, unemployment_rate_pct
  - 1 calculated: gdhi_per_head_gbp

DERIVED TABLE: gold.itl2_derived (2 metrics)
  - productivity_gbp_per_job
  - income_per_worker_gbp

RECONCILIATION: ITL2 → ITL1 (not directly to UK macro)

Exit codes:
    0: All critical checks passed
    1: Fatal errors - do not proceed to ITL3

Usage:
    python3 scripts/qa/itl2_forecast_qa.py
"""

import sys
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json


class ITL2ForecastQA:
    """ITL2 Forecast QA V4.2 - Two-table architecture aligned with ITL1 V3.5"""
    
    def __init__(self, 
                 db_path: str = "data/lake/warehouse.duckdb",
                 tolerance: float = 0.01):
        self.db_path = db_path
        self.tolerance = tolerance
        self.conn = duckdb.connect(db_path, read_only=True)
        
        # ===== BASE TABLE METRICS (8 total) =====
        
        # Additive metrics - reconciled to ITL1 parent
        self.additive_metrics = [
            'nominal_gva_mn_gbp',
            'gdhi_total_mn_gbp',
            'emp_total_jobs',
            'population_total',
            'population_16_64',
        ]
        
        # Rate metrics - primary from NOMIS, not reconciled
        self.rate_metrics = {
            'employment_rate_pct': {
                'min': 0.0, 
                'max': 100.0, 
                'expected_range': (55, 85),
            },
            'unemployment_rate_pct': {
                'min': 0.0, 
                'max': 100.0, 
                'expected_range': (2, 12),
            }
        }
        
        # Calculated metric - derived from reconciled components, stored in base
        self.calculated_metrics = ['gdhi_per_head_gbp']
        
        # All expected base metrics
        self.expected_base_metrics = (
            self.additive_metrics + 
            list(self.rate_metrics.keys()) + 
            self.calculated_metrics
        )
        
        # ===== DERIVED TABLE METRICS (2 total) =====
        self.derived_metrics = [
            'productivity_gbp_per_job',
            'income_per_worker_gbp',
        ]
        
        self.forecast_years = [2024, 2025, 2030, 2040, 2050]
        self.issues = []
        self.warnings = []
        
        # Load ITL2→ITL1 mapping
        self._load_geography_mapping()
    
    def _load_geography_mapping(self) -> None:
        """Load ITL2→ITL1 parent mapping"""
        try:
            lookup = pd.read_csv('data/reference/master_2025_geography_lookup.csv')
            lookup.columns = [col.replace('\ufeff', '') for col in lookup.columns]
            
            TLC_TO_ONS = {
                'TLC': 'E12000001', 'TLD': 'E12000002', 'TLE': 'E12000003',
                'TLF': 'E12000004', 'TLG': 'E12000005', 'TLH': 'E12000006',
                'TLI': 'E12000007', 'TLJ': 'E12000008', 'TLK': 'E12000009',
                'TLL': 'W92000004', 'TLM': 'S92000003', 'TLN': 'N92000002'
            }
            
            self.itl2_to_itl1 = lookup[['ITL225CD', 'ITL125CD']].drop_duplicates()
            self.itl2_to_itl1.columns = ['itl2_code', 'itl1_code_tlc']
            self.itl2_to_itl1['itl1_code'] = self.itl2_to_itl1['itl1_code_tlc'].map(TLC_TO_ONS)
            self.itl2_to_itl1 = self.itl2_to_itl1.drop(columns=['itl1_code_tlc'])
            
        except Exception as e:
            self.issues.append(f"Failed to load geography mapping: {e}")
            self.itl2_to_itl1 = pd.DataFrame(columns=['itl2_code', 'itl1_code'])
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load base forecasts, derived forecasts, ITL1 parent, and CIs"""
        
        # Base forecasts
        base_query = """
        SELECT 
            region_code, region_name, metric_id, period,
            value, data_type, ci_lower, ci_upper
        FROM gold.itl2_forecast
        WHERE period >= 2020
        ORDER BY region_code, metric_id, period
        """
        base_df = self.conn.execute(base_query).df()
        
        # Derived forecasts (optional - may not exist)
        try:
            derived_query = """
            SELECT 
                region_code, region_name, metric_id, period,
                value, data_type, ci_lower, ci_upper
            FROM gold.itl2_derived
            WHERE period >= 2020
            ORDER BY region_code, metric_id, period
            """
            derived_df = self.conn.execute(derived_query).df()
        except Exception:
            derived_df = pd.DataFrame()
        
        # ITL1 parent anchors
        itl1_query = """
        SELECT region_code, metric_id, period, value, data_type
        FROM gold.itl1_forecast
        WHERE period >= 2020
        ORDER BY region_code, metric_id, period
        """
        itl1_df = self.conn.execute(itl1_query).df()
        
        # CI subset from base
        ci_df = base_df[
            (base_df['period'] >= 2025) & 
            (base_df['ci_lower'].notna())
        ][['region_code', 'metric_id', 'period', 'value', 'ci_lower', 'ci_upper']].copy()
        
        return base_df, derived_df, itl1_df, ci_df
    
    def check_base_metric_coverage(self, base_df: pd.DataFrame) -> None:
        """Check that all 8 expected base metrics are present"""
        
        print("\n" + "="*80)
        print("BASE TABLE METRIC COVERAGE (gold.itl2_forecast)")
        print("="*80)
        
        present = set(base_df['metric_id'].unique())
        expected = set(self.expected_base_metrics)
        
        missing = expected - present
        extra = present - expected
        
        print(f"\n📊 Expected: {len(expected)} | Present: {len(present)}")
        
        if missing:
            self.issues.append(f"CRITICAL: Missing base metrics: {missing}")
            print(f"\n  ❌ CRITICAL: Missing {len(missing)} metrics:")
            for m in sorted(missing):
                print(f"     - {m}")
        
        if extra:
            print(f"\n  ℹ️  Extra metrics (not expected):")
            for m in sorted(extra):
                print(f"     - {m}")
        
        if not missing:
            print(f"\n  ✅ All 8 expected base metrics present")
    
    def check_derived_table(self, derived_df: pd.DataFrame) -> None:
        """Check derived table exists and has expected metrics"""
        
        print("\n" + "="*80)
        print("DERIVED TABLE (gold.itl2_derived)")
        print("="*80)
        
        if derived_df.empty:
            self.warnings.append("Derived table empty or missing")
            print("\n  ⚠️  gold.itl2_derived is empty or doesn't exist")
            return
        
        present = set(derived_df['metric_id'].unique())
        expected = set(self.derived_metrics)
        
        missing = expected - present
        
        print(f"\n📊 Expected: {len(expected)} | Present: {len(present)}")
        
        if missing:
            self.warnings.append(f"Missing derived metrics: {missing}")
            print(f"\n  ⚠️  Missing derived metrics:")
            for m in sorted(missing):
                print(f"     - {m}")
        else:
            print(f"\n  ✅ All derived metrics present")
        
        # Basic validation
        for metric in self.derived_metrics:
            m_data = derived_df[derived_df['metric_id'] == metric]
            if m_data.empty:
                continue
            
            nan_count = m_data['value'].isna().sum()
            neg_count = (m_data['value'] < 0).sum()
            
            if nan_count > 0:
                self.warnings.append(f"{metric}: {nan_count} NaN values")
            if neg_count > 0:
                self.warnings.append(f"{metric}: {neg_count} negative values")
        
        print(f"  📊 Derived rows: {len(derived_df):,}")
    
    def check_reconciliation(self, base_df: pd.DataFrame, itl1_df: pd.DataFrame) -> None:
        """Check ITL2 additive metrics sum to ITL1 parent"""
        
        print("\n" + "="*80)
        print("RECONCILIATION: ITL2 → ITL1 (ADDITIVE METRICS)")
        print("="*80)
        
        if self.itl2_to_itl1.empty:
            self.issues.append("No ITL2→ITL1 mapping available")
            print("\n  ❌ Cannot check reconciliation - no mapping")
            return
        
        # Merge parent codes
        base_with_parent = base_df.merge(
            self.itl2_to_itl1,
            left_on='region_code',
            right_on='itl2_code',
            how='left'
        )
        
        for metric in self.additive_metrics:
            print(f"\n📊 {metric}")
            print("-" * 80)
            
            # ITL1 parent values
            itl1_vals = itl1_df[
                (itl1_df['metric_id'] == metric) &
                (itl1_df['data_type'] == 'forecast')
            ].set_index(['region_code', 'period'])['value']
            
            if itl1_vals.empty:
                self.warnings.append(f"{metric}: No ITL1 anchor found")
                print(f"  ⚠️  No ITL1 anchor found")
                continue
            
            # ITL2 sums by parent
            itl2_fcst = base_with_parent[
                (base_with_parent['metric_id'] == metric) &
                (base_with_parent['data_type'] == 'forecast')
            ]
            itl2_sums = itl2_fcst.groupby(['itl1_code', 'period'])['value'].sum()
            
            for year in self.forecast_years:
                year_errors = []
                
                for itl1_code in itl2_sums.index.get_level_values('itl1_code').unique():
                    if pd.isna(itl1_code):
                        continue
                    
                    try:
                        itl2_sum = itl2_sums.loc[(itl1_code, year)]
                        itl1_val = itl1_vals.loc[(itl1_code, year)]
                    except KeyError:
                        continue
                    
                    diff = itl2_sum - itl1_val
                    pct_diff = (diff / itl1_val * 100) if itl1_val != 0 else 0
                    
                    if abs(pct_diff) > self.tolerance * 100:
                        year_errors.append((itl1_code, itl2_sum, itl1_val, pct_diff))
                
                if year_errors:
                    status = "❌ FAIL"
                    self.issues.append(f"{metric} {year}: {len(year_errors)} ITL1 parents with >1% deviation")
                else:
                    status = "✅ PASS"
                
                # Show summary for key years
                total_itl2 = itl2_sums[itl2_sums.index.get_level_values('period') == year].sum()
                total_itl1 = itl1_vals[itl1_vals.index.get_level_values('period') == year].sum()
                delta = total_itl2 - total_itl1
                pct = (delta / total_itl1 * 100) if total_itl1 != 0 else 0
                
                print(f"  {year}: {status} | ITL2={total_itl2:>15,.0f} | ITL1={total_itl1:>15,.0f} | "
                      f"Δ={delta:>+12,.0f} ({pct:>+6.3f}%)")
    
    def check_gdhi_per_head(self, base_df: pd.DataFrame) -> None:
        """Validate gdhi_per_head_gbp = gdhi_total / population × 1e6"""
        
        print("\n" + "="*80)
        print("GDHI PER HEAD CALCULATION CHECK")
        print("="*80)
        
        fcst = base_df[base_df['data_type'] == 'forecast']
        
        pivot = fcst.pivot_table(
            index=['region_code', 'region_name', 'period'],
            columns='metric_id',
            values='value'
        ).reset_index()
        
        required = ['gdhi_total_mn_gbp', 'population_total', 'gdhi_per_head_gbp']
        if not all(c in pivot.columns for c in required):
            self.warnings.append("Cannot validate gdhi_per_head - missing components")
            print("\n  ⚠️  Missing required columns for validation")
            return
        
        pivot['expected'] = pivot['gdhi_total_mn_gbp'] / pivot['population_total'] * 1e6
        pivot['diff_pct'] = (pivot['gdhi_per_head_gbp'] - pivot['expected']) / pivot['expected'] * 100
        
        for year in [2025, 2030, 2050]:
            yr_data = pivot[pivot['period'] == year]
            if yr_data.empty:
                continue
            
            max_err = yr_data['diff_pct'].abs().max()
            
            if max_err > self.tolerance * 100:
                worst = yr_data.loc[yr_data['diff_pct'].abs().idxmax()]
                self.issues.append(f"gdhi_per_head {year}: {worst['region_name']} error={max_err:.2f}%")
                print(f"\n  {year}: ❌ Max error: {max_err:.3f}%")
                print(f"       Worst: {worst['region_name']} (actual={worst['gdhi_per_head_gbp']:.0f}, expected={worst['expected']:.0f})")
            else:
                print(f"\n  {year}: ✅ Max error: {max_err:.3f}%")
    
    def check_rate_bounds(self, base_df: pd.DataFrame) -> None:
        """Validate rate metrics within 0-100%"""
        
        print("\n" + "="*80)
        print("RATE METRIC BOUNDS")
        print("="*80)
        
        for rate, bounds in self.rate_metrics.items():
            data = base_df[base_df['metric_id'] == rate]
            
            if data.empty:
                self.warnings.append(f"{rate}: No data found")
                print(f"\n📊 {rate}: ⚠️  No data")
                continue
            
            print(f"\n📊 {rate}")
            print("-" * 80)
            print(f"  Regions: {data['region_code'].nunique()}")
            
            below = data[data['value'] < bounds['min']]
            above = data[data['value'] > bounds['max']]
            
            if not below.empty:
                self.issues.append(f"{rate}: {len(below)} values < {bounds['min']}%")
                print(f"  ❌ {len(below)} below {bounds['min']}%")
            
            if not above.empty:
                self.issues.append(f"{rate}: {len(above)} values > {bounds['max']}%")
                print(f"  ❌ {len(above)} above {bounds['max']}%")
            
            if below.empty and above.empty:
                print(f"  ✅ All within [{bounds['min']}, {bounds['max']}]%")
            
            # Expected range check
            fcst = data[data['data_type'] == 'forecast']
            if not fcst.empty:
                exp_min, exp_max = bounds['expected_range']
                stats = fcst.groupby('region_code')['value'].agg(['min', 'max'])
                outliers = stats[(stats['min'] < exp_min) | (stats['max'] > exp_max)]
                
                if not outliers.empty:
                    self.warnings.append(f"{rate}: {len(outliers)} regions outside typical range")
                    print(f"  ⚠️  {len(outliers)} regions outside [{exp_min}, {exp_max}]%")
    
    def check_population_16_64(self, base_df: pd.DataFrame) -> None:
        """Working-age population checks"""
        
        print("\n" + "="*80)
        print("POPULATION 16-64 CHECKS")
        print("="*80)
        
        pop_total = base_df[base_df['metric_id'] == 'population_total']
        pop_wa = base_df[base_df['metric_id'] == 'population_16_64']
        
        if pop_wa.empty:
            self.warnings.append("population_16_64 not found")
            print("\n  ⚠️  population_16_64 not found")
            return
        
        print(f"\n  Regions: {pop_wa['region_code'].nunique()}")
        print(f"  Years: {pop_wa['period'].min()}-{pop_wa['period'].max()}")
        
        # Check working-age < total
        merged = pop_total.merge(
            pop_wa,
            on=['region_code', 'period', 'data_type'],
            suffixes=('_total', '_wa')
        )
        
        if not merged.empty:
            invalid = merged[merged['value_wa'] > merged['value_total']]
            if not invalid.empty:
                self.issues.append(f"CRITICAL: {len(invalid)} cases pop_16_64 > pop_total")
                print(f"\n  ❌ {len(invalid)} cases working-age > total")
            else:
                print(f"\n  ✅ Working-age always < total")
            
            # Share check
            merged['share'] = merged['value_wa'] / merged['value_total'] * 100
            fcst = merged[merged['data_type'] == 'forecast']
            
            if not fcst.empty:
                low = fcst[fcst['share'] < 50]
                high = fcst[fcst['share'] > 75]
                
                print(f"\n  Share range: {fcst['share'].min():.1f}% - {fcst['share'].max():.1f}%")
                
                if not low.empty or not high.empty:
                    self.warnings.append(f"Working-age share outside 50-75% in {len(low) + len(high)} obs")
                else:
                    print(f"  ✅ Shares within expected bounds")
    
    def check_historical_continuity(self, base_df: pd.DataFrame) -> None:
        """Check smooth transition at forecast horizon"""
        
        print("\n" + "="*80)
        print("HISTORICAL CONTINUITY")
        print("="*80)
        
        hist = base_df[base_df['data_type'] == 'historical']
        fcst = base_df[base_df['data_type'] == 'forecast']
        
        if hist.empty or fcst.empty:
            print("\n  ⚠️  Cannot check continuity")
            return
        
        last_hist = hist['period'].max()
        first_fcst = fcst['period'].min()
        
        print(f"\n🔗 Transition: {int(last_hist)} → {int(first_fcst)}")
        print("-" * 80)
        
        for metric in self.expected_base_metrics:
            trans = base_df[
                (base_df['metric_id'] == metric) &
                (base_df['period'].isin([last_hist, first_fcst]))
            ].pivot_table(
                index='region_code',
                columns='period',
                values='value'
            )
            
            if trans.empty or last_hist not in trans.columns or first_fcst not in trans.columns:
                continue
            
            if metric in self.rate_metrics:
                trans['change'] = trans[first_fcst] - trans[last_hist]
                extreme = trans[trans['change'].abs() > 5]
                threshold = ">5pp"
            else:
                trans['change'] = (trans[first_fcst] / trans[last_hist] - 1) * 100
                extreme = trans[(trans['change'] > 20) | (trans['change'] < -10)]
                threshold = ">20% or <-10%"
            
            if not extreme.empty:
                self.warnings.append(f"{metric}: {len(extreme)} extreme jumps")
                print(f"  ⚠️  {metric}: {len(extreme)} jumps ({threshold})")
            else:
                print(f"  ✅ {metric}")
    
    def check_confidence_intervals(self, ci_df: pd.DataFrame) -> None:
        """CI validation"""
        
        print("\n" + "="*80)
        print("CONFIDENCE INTERVALS")
        print("="*80)
        
        if ci_df.empty:
            self.warnings.append("No CIs found")
            print("\n  ⚠️  No confidence intervals")
            return
        
        print(f"\n📊 {len(ci_df):,} CI records")
        
        # Ordering
        invalid = ci_df[ci_df['ci_lower'] > ci_df['ci_upper']]
        if not invalid.empty:
            self.issues.append(f"{len(invalid)} CIs with lower > upper")
            print(f"  ❌ {len(invalid)} invalid ordering")
        else:
            print(f"  ✅ All CIs properly ordered")
        
        # Point within CI
        outside = ci_df[
            (ci_df['value'] < ci_df['ci_lower']) |
            (ci_df['value'] > ci_df['ci_upper'])
        ]
        if not outside.empty:
            self.warnings.append(f"{len(outside)} points outside CIs")
            print(f"  ⚠️  {len(outside)} points outside CIs")
        else:
            print(f"  ✅ All points within CIs")
    
    def check_ci_diagnostics_per_metric(self, base_df: pd.DataFrame) -> None:
        """Detailed CI width analysis per metric"""
        
        print("\n" + "="*80)
        print("CI WIDTH DIAGNOSTICS BY METRIC")
        print("="*80)
        
        ci_data = base_df[
            (base_df['data_type'] == 'forecast') &
            (base_df['ci_lower'].notna()) &
            (base_df['ci_upper'].notna())
        ].copy()
        
        if ci_data.empty:
            print("\n  ⚠️  No CI data for diagnostics")
            return
        
        ci_data['ci_width'] = ci_data['ci_upper'] - ci_data['ci_lower']
        ci_data['ci_width_pct'] = (ci_data['ci_width'] / ci_data['value'].abs()) * 100
        
        for metric in self.expected_base_metrics:
            m_data = ci_data[ci_data['metric_id'] == metric]
            if m_data.empty:
                continue
            
            print(f"\n📊 {metric}")
            print("-" * 80)
            
            # By forecast horizon
            for year in [2025, 2030, 2040, 2050]:
                yr = m_data[m_data['period'] == year]
                if yr.empty:
                    continue
                
                width_pct = yr['ci_width_pct']
                print(f"  {year}: median={width_pct.median():>6.1f}% | "
                      f"p90={width_pct.quantile(0.9):>6.1f}% | "
                      f"max={width_pct.max():>6.1f}%")
                
                # Flag extremely wide CIs
                extreme = yr[yr['ci_width_pct'] > 50]
                if not extreme.empty:
                    self.warnings.append(f"{metric} {year}: {len(extreme)} CIs >50% width")
            
            # Check CI growth over horizon
            if 2025 in m_data['period'].values and 2050 in m_data['period'].values:
                early = m_data[m_data['period'] == 2025]['ci_width_pct'].median()
                late = m_data[m_data['period'] == 2050]['ci_width_pct'].median()
                if late > early * 5:
                    self.warnings.append(f"{metric}: CI width expands >5x by 2050")
                    print(f"  ⚠️  CI expands {late/early:.1f}x from 2025→2050")
    
    def check_growth_rate_bounds(self, base_df: pd.DataFrame) -> None:
        """YoY growth rate validation per metric"""
        
        print("\n" + "="*80)
        print("YEAR-ON-YEAR GROWTH RATE VALIDATION")
        print("="*80)
        
        # Bounds: (min_growth%, max_growth%)
        growth_bounds = {
            'nominal_gva_mn_gbp': (-8, 12),
            'gdhi_total_mn_gbp': (-8, 12),
            'gdhi_per_head_gbp': (-6, 10),
            'emp_total_jobs': (-5, 5),
            'population_total': (-2, 3),
            'population_16_64': (-3, 3),
            'employment_rate_pct': (-5, 5),  # pp change
            'unemployment_rate_pct': (-3, 5),  # pp change
        }
        
        fcst = base_df[base_df['data_type'] == 'forecast'].copy()
        
        for metric, (min_g, max_g) in growth_bounds.items():
            m_data = fcst[fcst['metric_id'] == metric]
            if m_data.empty:
                continue
            
            print(f"\n📊 {metric} (bounds: {min_g}% to {max_g}%)")
            print("-" * 80)
            
            # Calculate YoY growth per region
            violations = []
            for region in m_data['region_code'].unique():
                r_data = m_data[m_data['region_code'] == region].sort_values('period')
                
                if metric in self.rate_metrics:
                    # Absolute change for rates
                    r_data = r_data.copy()
                    r_data['growth'] = r_data['value'].diff()
                else:
                    # Percentage change for levels
                    r_data = r_data.copy()
                    r_data['growth'] = r_data['value'].pct_change() * 100
                
                extreme = r_data[(r_data['growth'] < min_g) | (r_data['growth'] > max_g)]
                if not extreme.empty:
                    region_name = r_data['region_name'].iloc[0] if 'region_name' in r_data.columns else region
                    for _, row in extreme.iterrows():
                        violations.append({
                            'region': region_name,
                            'year': int(row['period']),
                            'growth': row['growth']
                        })
            
            if violations:
                self.warnings.append(f"{metric}: {len(violations)} YoY growth violations")
                print(f"  ⚠️  {len(violations)} violations")
                # Show worst 3
                sorted_v = sorted(violations, key=lambda x: abs(x['growth']), reverse=True)[:3]
                for v in sorted_v:
                    print(f"     {v['region']} {v['year']}: {v['growth']:+.1f}%")
            else:
                print(f"  ✅ All regions within bounds")
    
    def check_cross_metric_coherence(self, base_df: pd.DataFrame, derived_df: pd.DataFrame) -> None:
        """Cross-metric ratio consistency checks"""
        
        print("\n" + "="*80)
        print("CROSS-METRIC COHERENCE")
        print("="*80)
        
        fcst = base_df[base_df['data_type'] == 'forecast']
        
        pivot = fcst.pivot_table(
            index=['region_code', 'region_name', 'period'],
            columns='metric_id',
            values='value'
        ).reset_index()
        
        # 1. Employment rate vs jobs/population ratio
        if all(c in pivot.columns for c in ['emp_total_jobs', 'population_16_64', 'employment_rate_pct']):
            print("\n🔗 Employment rate consistency")
            print("-" * 80)
            
            pivot['jobs_per_wa'] = pivot['emp_total_jobs'] / pivot['population_16_64'] * 100
            pivot['rate_diff'] = pivot['employment_rate_pct'] - pivot['jobs_per_wa']
            
            # Note: These won't match exactly because emp_rate is surveyed, not derived
            extreme = pivot[pivot['rate_diff'].abs() > 30]
            if not extreme.empty:
                self.warnings.append(f"Employment rate diverges >30pp from jobs/pop in {len(extreme)} obs")
                print(f"  ⚠️  {len(extreme)} obs with >30pp divergence")
            else:
                print(f"  ✅ Employment rate broadly consistent with jobs/working-age")
        
        # 2. Productivity from derived table vs calculated
        if not derived_df.empty:
            print("\n🔗 Productivity consistency")
            print("-" * 80)
            
            prod = derived_df[
                (derived_df['metric_id'] == 'productivity_gbp_per_job') &
                (derived_df['data_type'] == 'forecast')
            ]
            
            if not prod.empty and all(c in pivot.columns for c in ['nominal_gva_mn_gbp', 'emp_total_jobs']):
                merged = pivot.merge(
                    prod[['region_code', 'period', 'value']].rename(columns={'value': 'prod_stored'}),
                    on=['region_code', 'period'],
                    how='inner'
                )
                
                merged['prod_calc'] = merged['nominal_gva_mn_gbp'] / merged['emp_total_jobs'] * 1e6
                merged['prod_diff_pct'] = (merged['prod_stored'] - merged['prod_calc']) / merged['prod_calc'] * 100
                
                max_diff = merged['prod_diff_pct'].abs().max()
                if max_diff > 1:
                    self.warnings.append(f"Productivity deviation up to {max_diff:.1f}%")
                    print(f"  ⚠️  Max deviation: {max_diff:.2f}%")
                else:
                    print(f"  ✅ Stored vs calculated within 1%")
        
        # 3. Income per worker consistency
        if not derived_df.empty:
            print("\n🔗 Income per worker consistency")
            print("-" * 80)
            
            income = derived_df[
                (derived_df['metric_id'] == 'income_per_worker_gbp') &
                (derived_df['data_type'] == 'forecast')
            ]
            
            if not income.empty and all(c in pivot.columns for c in ['gdhi_total_mn_gbp', 'emp_total_jobs']):
                merged = pivot.merge(
                    income[['region_code', 'period', 'value']].rename(columns={'value': 'inc_stored'}),
                    on=['region_code', 'period'],
                    how='inner'
                )
                
                merged['inc_calc'] = merged['gdhi_total_mn_gbp'] / merged['emp_total_jobs'] * 1e6
                merged['inc_diff_pct'] = (merged['inc_stored'] - merged['inc_calc']) / merged['inc_calc'] * 100
                
                max_diff = merged['inc_diff_pct'].abs().max()
                if max_diff > 1:
                    self.warnings.append(f"Income/worker deviation up to {max_diff:.1f}%")
                    print(f"  ⚠️  Max deviation: {max_diff:.2f}%")
                else:
                    print(f"  ✅ Stored vs calculated within 1%")
        
        # 4. Population 16-64 share stability
        if all(c in pivot.columns for c in ['population_16_64', 'population_total']):
            print("\n🔗 Working-age share trajectory")
            print("-" * 80)
            
            pivot['wa_share'] = pivot['population_16_64'] / pivot['population_total'] * 100
            
            # Check for implausible share changes over time
            for region in pivot['region_code'].unique():
                r_data = pivot[pivot['region_code'] == region].sort_values('period')
                if len(r_data) < 2:
                    continue
                
                share_change = r_data['wa_share'].iloc[-1] - r_data['wa_share'].iloc[0]
                if abs(share_change) > 15:
                    region_name = r_data['region_name'].iloc[0]
                    self.warnings.append(f"{region_name}: WA share changes {share_change:+.1f}pp")
            
            # Summary
            share_2025 = pivot[pivot['period'] == 2025]['wa_share'].mean()
            share_2050 = pivot[pivot['period'] == 2050]['wa_share'].mean()
            if not pd.isna(share_2025) and not pd.isna(share_2050):
                print(f"  Mean share: {share_2025:.1f}% (2025) → {share_2050:.1f}% (2050)")
            print(f"  ✅ Working-age trajectories validated")
    
    def check_terminal_year_diagnostics(self, base_df: pd.DataFrame, derived_df: pd.DataFrame) -> None:
        """2050 snapshot summary"""
        
        print("\n" + "="*80)
        print("TERMINAL YEAR DIAGNOSTICS (2050)")
        print("="*80)
        
        fcst_2050 = base_df[
            (base_df['data_type'] == 'forecast') &
            (base_df['period'] == 2050)
        ]
        
        if fcst_2050.empty:
            print("\n  ⚠️  No 2050 forecasts found")
            return
        
        pivot = fcst_2050.pivot_table(
            index=['region_code', 'region_name'],
            columns='metric_id',
            values='value'
        ).reset_index()
        
        print(f"\n📊 2050 Snapshot ({len(pivot)} regions)")
        print("-" * 80)
        
        # UK totals for additive metrics (sum of ITL2 = UK)
        print("\n  UK Totals (sum of ITL2):")
        for metric in self.additive_metrics:
            if metric in pivot.columns:
                total = pivot[metric].sum()
                if 'gva' in metric or 'gdhi' in metric:
                    print(f"    {metric}: £{total:,.0f}m")
                elif 'population' in metric or 'emp' in metric:
                    print(f"    {metric}: {total:,.0f}")
        
        # Regional rankings
        print("\n  📍 Top 5 by GVA (if available):")
        if 'nominal_gva_mn_gbp' in pivot.columns:
            top_gva = pivot.nlargest(5, 'nominal_gva_mn_gbp')
            for _, row in top_gva.iterrows():
                print(f"    {row['region_name']}: £{row['nominal_gva_mn_gbp']:,.0f}m")
        
        print("\n  📍 Top 5 by GDHI/head (if available):")
        if 'gdhi_per_head_gbp' in pivot.columns:
            top_gdhi = pivot.nlargest(5, 'gdhi_per_head_gbp')
            for _, row in top_gdhi.iterrows():
                print(f"    {row['region_name']}: £{row['gdhi_per_head_gbp']:,.0f}")
        
        # London sub-regions spotlight
        london_regions = pivot[pivot['region_code'].str.startswith('TLI', na=False)]
        if not london_regions.empty:
            print("\n  🏙️  London sub-regions 2050 (sample):")
            for _, row in london_regions.head(3).iterrows():
                if 'gdhi_per_head_gbp' in pivot.columns:
                    print(f"    {row['region_name']}: GDHI/head £{row['gdhi_per_head_gbp']:,.0f}")
        
        # Derived metrics 2050
        if not derived_df.empty:
            derived_2050 = derived_df[
                (derived_df['data_type'] == 'forecast') &
                (derived_df['period'] == 2050)
            ]
            
            if not derived_2050.empty:
                print("\n  📈 Derived Metrics 2050:")
                for metric in self.derived_metrics:
                    m_data = derived_2050[derived_2050['metric_id'] == metric]
                    if not m_data.empty:
                        mean_val = m_data['value'].mean()
                        print(f"    {metric}: Mean £{mean_val:,.0f}")
        
        # Sanity flags
        print("\n  🚨 Sanity Checks:")
        issues_2050 = []
        
        if 'nominal_gva_mn_gbp' in pivot.columns:
            max_gva = pivot['nominal_gva_mn_gbp'].max()
            if max_gva > 500_000:  # £500bn for a single ITL2 would be extreme
                issues_2050.append(f"Max regional GVA >£500bn: £{max_gva:,.0f}m")
        
        if 'gdhi_per_head_gbp' in pivot.columns:
            max_gdhi = pivot['gdhi_per_head_gbp'].max()
            if max_gdhi > 150_000:
                issues_2050.append(f"Max GDHI/head >£150k: £{max_gdhi:,.0f}")
            min_gdhi = pivot['gdhi_per_head_gbp'].min()
            if min_gdhi < 10_000:
                issues_2050.append(f"Min GDHI/head <£10k: £{min_gdhi:,.0f}")
        
        if 'population_total' in pivot.columns:
            max_pop = pivot['population_total'].max()
            if max_pop > 5_000_000:  # 5m for single ITL2 would be extreme
                issues_2050.append(f"Max regional pop >5m: {max_pop:,.0f}")
        
        if issues_2050:
            for issue in issues_2050:
                self.warnings.append(f"2050: {issue}")
                print(f"    ⚠️  {issue}")
        else:
            print(f"    ✅ All 2050 values within sanity bounds")
    
    def check_var_system_outputs(self, base_df: pd.DataFrame) -> None:
        """Validate GVA-employment productivity relationship (INFORMATIONAL)"""
        
        print("\n" + "="*80)
        print("GVA-EMPLOYMENT RELATIONSHIP (INFORMATIONAL)")
        print("="*80)
        print("ℹ️  Note: Post-processing (reconciliation, dampening) can break raw correlation.")
        print("   Low output correlation ≠ VAR system failure.")
        
        regions_with_both = base_df[
            (base_df['metric_id'].isin(['nominal_gva_mn_gbp', 'emp_total_jobs'])) &
            (base_df['data_type'] == 'forecast')
        ].groupby('region_code').filter(
            lambda x: len(x['metric_id'].unique()) == 2
        )['region_code'].unique()
        
        print(f"\n🔗 Regions with GVA+employment forecasts: {len(regions_with_both)}")
        
        if len(regions_with_both) == 0:
            self.warnings.append("No GVA-employment pairs found")
            print("  ⚠️  No GVA-employment forecast pairs found")
            return
        
        extreme_count = 0
        for region_code in regions_with_both:
            region_data = base_df[
                (base_df['region_code'] == region_code) &
                (base_df['metric_id'].isin(['nominal_gva_mn_gbp', 'emp_total_jobs']))
            ]
            
            forecast_data = region_data[region_data['data_type'] == 'forecast'].pivot_table(
                index='period',
                columns='metric_id',
                values='value'
            )
            
            if forecast_data.empty or 'emp_total_jobs' not in forecast_data.columns:
                continue
            
            forecast_data['productivity'] = (
                forecast_data['nominal_gva_mn_gbp'] / forecast_data['emp_total_jobs'] * 1_000_000
            )
            
            prod_growth = forecast_data['productivity'].pct_change() * 100
            extreme = prod_growth[(prod_growth > 10) | (prod_growth < -5)]
            
            if not extreme.empty:
                extreme_count += 1
        
        if extreme_count > len(regions_with_both) * 0.3:
            self.warnings.append(f"{extreme_count}/{len(regions_with_both)} regions with extreme productivity growth")
            print(f"  ⚠️  {extreme_count} regions with extreme productivity growth (expected after dampening)")
        else:
            print(f"  ✅ Most regions have reasonable productivity growth")
    
    def check_derived_metrics_full(self, derived_df: pd.DataFrame) -> None:
        """Full validation of derived metrics table"""
        
        print("\n" + "="*80)
        print("DERIVED METRICS VALIDATION (gold.itl2_derived)")
        print("="*80)
        
        if derived_df.empty:
            self.warnings.append("Derived table empty or missing")
            print("\n  ⚠️  No derived metrics to validate")
            return
        
        for metric in self.derived_metrics:
            m_data = derived_df[
                (derived_df['metric_id'] == metric) &
                (derived_df['data_type'] == 'forecast')
            ]
            
            if m_data.empty:
                self.warnings.append(f"Derived metric {metric} not found")
                print(f"\n⚠️  {metric}: Not found")
                continue
            
            print(f"\n📈 {metric}")
            print("-" * 80)
            
            # NaN check
            nan_count = m_data['value'].isna().sum()
            if nan_count > 0:
                self.warnings.append(f"{metric}: {nan_count} NaN values")
                print(f"  ⚠️  {nan_count} NaN values")
            else:
                print(f"  ✅ No NaN values")
            
            # Negative check
            neg_count = (m_data['value'] < 0).sum()
            if neg_count > 0:
                self.warnings.append(f"{metric}: {neg_count} negative values")
                print(f"  ⚠️  {neg_count} negative values")
            else:
                print(f"  ✅ No negative values")
            
            # Coverage
            print(f"  Regions covered: {m_data['region_code'].nunique()}")
            
            # CI check if available
            if 'ci_lower' in m_data.columns and 'ci_upper' in m_data.columns:
                ci_data = m_data.dropna(subset=['ci_lower', 'ci_upper'])
                if not ci_data.empty:
                    outside = ci_data[
                        (ci_data['value'] < ci_data['ci_lower']) |
                        (ci_data['value'] > ci_data['ci_upper'])
                    ]
                    if len(outside) > 0:
                        self.warnings.append(f"{metric}: {len(outside)} points outside CIs")
                        print(f"  ⚠️  {len(outside)} points outside CIs")
                    else:
                        print(f"  ✅ All points within CIs")
            
            # Plausibility bounds
            if metric == 'productivity_gbp_per_job':
                low = m_data[m_data['value'] < 20_000]
                high = m_data[m_data['value'] > 500_000]
                if not low.empty or not high.empty:
                    self.warnings.append(f"{metric}: {len(low) + len(high)} implausible values")
                    print(f"  ⚠️  {len(low) + len(high)} values outside £20k-£500k range")
            
            if metric == 'income_per_worker_gbp':
                low = m_data[m_data['value'] < 15_000]
                high = m_data[m_data['value'] > 300_000]
                if not low.empty or not high.empty:
                    self.warnings.append(f"{metric}: {len(low) + len(high)} implausible values")
                    print(f"  ⚠️  {len(low) + len(high)} values outside £15k-£300k range")
    
    def check_regional_plausibility(self, base_df: pd.DataFrame) -> None:
        """Regional sanity checks - London sub-regions focus"""
        
        print("\n" + "="*80)
        print("REGIONAL PLAUSIBILITY")
        print("="*80)
        
        # London sub-regions GDHI per head
        london_regions = base_df[
            base_df['region_code'].str.startswith('TLI', na=False)
        ]['region_code'].unique()
        
        if len(london_regions) > 0:
            print(f"\n🏙️  London GDHI per head ({len(london_regions)} sub-regions)")
            
            for region in london_regions[:3]:  # Sample
                gdhi_data = base_df[
                    (base_df['region_code'] == region) &
                    (base_df['metric_id'] == 'gdhi_per_head_gbp')
                ]
                
                region_name = gdhi_data['region_name'].iloc[0] if 'region_name' in gdhi_data.columns and not gdhi_data.empty else region
                
                h_2023 = gdhi_data[(gdhi_data['data_type'] == 'historical') & (gdhi_data['period'] == 2023)]
                f_2025 = gdhi_data[(gdhi_data['data_type'] == 'forecast') & (gdhi_data['period'] == 2025)]
                
                if not h_2023.empty:
                    val = h_2023['value'].iloc[0]
                    if val < 20_000:
                        self.warnings.append(f"{region_name} 2023 GDHI/head low: £{val:,.0f}")
                        print(f"  ⚠️  {region_name} 2023: £{val:,.0f} (expected >£20k)")
                    else:
                        print(f"  ✅ {region_name} 2023: £{val:,.0f}")
                
                if not h_2023.empty and not f_2025.empty:
                    h = h_2023['value'].iloc[0]
                    f = f_2025['value'].iloc[0]
                    g = (f / h - 1) * 100
                    print(f"     {region_name} 2025: £{f:,.0f} ({g:+.1f}%)")
        
        # CAGRs for key metrics
        print(f"\n📍 Regional CAGRs (2025-2030) - extreme check")
        
        for metric in ['nominal_gva_mn_gbp', 'emp_total_jobs', 'population_16_64']:
            m_data = base_df[
                (base_df['metric_id'] == metric) &
                (base_df['data_type'] == 'forecast') &
                (base_df['period'].isin([2025, 2030]))
            ].pivot_table(
                index='region_code',
                columns='period',
                values='value'
            )
            
            if m_data.empty or 2025 not in m_data.columns or 2030 not in m_data.columns:
                continue
            
            m_data['cagr'] = ((m_data[2030] / m_data[2025]) ** 0.2 - 1) * 100
            extreme = m_data[(m_data['cagr'] > 8) | (m_data['cagr'] < -3)]
            
            if not extreme.empty:
                self.warnings.append(f"{metric}: {len(extreme)} extreme CAGRs")
                print(f"  ⚠️  {metric}: {len(extreme)} extreme")
            else:
                print(f"  ✅ {metric}")
    
    def generate_report(self) -> int:
        """Generate summary and return exit code"""
        
        print("\n" + "="*80)
        print("QA SUMMARY")
        print("="*80)
        
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
        
        print("\n" + "="*80)
        if len(self.issues) == 0:
            print("✅ ITL2 FORECAST QA V4.2: PASSED")
            print("="*80)
            print("\nReady to proceed to ITL3.")
            exit_code = 0
        else:
            print("❌ ITL2 FORECAST QA V4.2: FAILED")
            print("="*80)
            print(f"\n{len(self.issues)} critical issues. Fix before ITL3.")
            exit_code = 1
        
        # JSON export
        summary = {
            'level': 'ITL2',
            'version': '4.2',
            'timestamp': datetime.now().isoformat(),
            'status': 'PASSED' if exit_code == 0 else 'FAILED',
            'exit_code': exit_code,
            'architecture': {
                'base_table': 'gold.itl2_forecast',
                'base_metrics': self.expected_base_metrics,
                'derived_table': 'gold.itl2_derived',
                'derived_metrics': self.derived_metrics,
                'reconciles_to': 'gold.itl1_forecast',
            },
            'stats': {
                'base_metrics_expected': len(self.expected_base_metrics),
                'derived_metrics_expected': len(self.derived_metrics),
                'critical_issues': len(self.issues),
                'warnings': len(self.warnings),
            },
            'critical_issues': self.issues,
            'warnings': self.warnings,
        }
        
        output_path = Path('data/qa/itl2_qa_summary.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📄 Summary: {output_path}")
        print(f"📤 Exit code: {exit_code}")
        
        return exit_code
    
    def run_all_checks(self) -> int:
        """Run full QA suite"""
        
        print("\n" + "="*80)
        print("ITL2 FORECAST QA V4.2")
        print("="*80)
        print(f"Database: {self.db_path}")
        print(f"Tolerance: {self.tolerance * 100}%")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nArchitecture:")
        print(f"  Base table: gold.itl2_forecast (8 metrics)")
        print(f"  Derived table: gold.itl2_derived (2 metrics)")
        print(f"  Reconciles to: gold.itl1_forecast")
        
        print("\n📥 Loading data...")
        base_df, derived_df, itl1_df, ci_df = self.load_data()
        print(f"  ✅ Base: {len(base_df):,} rows")
        print(f"  ✅ Derived: {len(derived_df):,} rows")
        print(f"  ✅ ITL1 parent: {len(itl1_df):,} rows")
        print(f"  ✅ CIs: {len(ci_df):,} rows")
        
        # Core validation
        self.check_base_metric_coverage(base_df)
        self.check_derived_table(derived_df)
        self.check_reconciliation(base_df, itl1_df)
        self.check_gdhi_per_head(base_df)
        self.check_rate_bounds(base_df)
        self.check_population_16_64(base_df)
        self.check_historical_continuity(base_df)
        self.check_confidence_intervals(ci_df)
        
        # Extended diagnostics
        self.check_ci_diagnostics_per_metric(base_df)
        self.check_growth_rate_bounds(base_df)
        self.check_cross_metric_coherence(base_df, derived_df)
        self.check_terminal_year_diagnostics(base_df, derived_df)
        
        # VAR and derived full validation
        self.check_var_system_outputs(base_df)
        self.check_derived_metrics_full(derived_df)
        
        # Regional plausibility
        self.check_regional_plausibility(base_df)
        
        exit_code = self.generate_report()
        self.conn.close()
        
        return exit_code


def main():
    qa = ITL2ForecastQA(
        db_path="data/lake/warehouse.duckdb",
        tolerance=0.01
    )
    exit_code = qa.run_all_checks()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()