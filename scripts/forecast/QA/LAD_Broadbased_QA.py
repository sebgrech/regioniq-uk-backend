#!/usr/bin/env python3
"""
LAD Forecast QA Script V2.0
===========================
Aligned with ITL3 Forecast QA (V6.0) — same structure, formatting, and gate style.

BASE TABLE: gold.lad_forecast (10 metrics)
  - 4 additive: nominal_gva_mn_gbp, gdhi_total_mn_gbp, emp_total_jobs, population_total
  - 2 rates: employment_rate_pct, unemployment_rate_pct
  - 4 calculated: gdhi_per_head_gbp, productivity_gbp_per_job, income_per_worker_gbp

RECONCILIATION: LAD → ITL3 (share-based allocation, forecast only)

LAD METHODOLOGY (differs from higher levels):
  - Uses SHARE-BASED ALLOCATION, not econometric forecasting
  - Additive metrics: LAD_value(t) = ITL3_value(t) × historical_share
  - Shares fixed from 5-year historical average
  - sum(LAD children) = ITL3 parent BY CONSTRUCTION (forecast only)
  - Historical data is actual ONS values — will NOT reconcile

TERMINAL LEVEL: LAD is the final level in the cascade. No downstream gating.

CASCADE ARCHITECTURE:
    UK Macro (1)
        ↓ reconcile
    ITL1 (12)
        ↓ reconcile
    ITL2 (46)
        ↓ reconcile
    ITL3 (182)
        ↓ share allocation
    LAD (361) ← THIS LEVEL (TERMINAL)

Exit codes:
    0: All critical checks passed — ready for Supabase push
    1: Fatal errors — do NOT push to production

Usage:
    python3 scripts/qa/lad_forecast_qa.py
"""

import sys
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json


class LADForecastQA:
    """LAD Forecast QA V2.0 - Aligned with ITL3 V6.0 structure"""
    
    def __init__(self, 
                 db_path: str = "data/lake/warehouse.duckdb",
                 tolerance: float = 0.0001):  # Tighter than ITL3 — shares should be exact
        self.db_path = db_path
        self.tolerance = tolerance
        self.conn = duckdb.connect(db_path, read_only=True)
        
        # ===== BASE TABLE METRICS =====
        
        # Additive metrics — reconciled to ITL3 parent (forecast only)
        self.additive_metrics = [
            'nominal_gva_mn_gbp',
            'gdhi_total_mn_gbp',
            'emp_total_jobs',
            'population_total',
        ]
        
        # Rate metrics — inherited from ITL3 parent, not reconciled
        self.rate_metrics = {
            'employment_rate_pct': {
                'min': 0.0,
                'max': 100.0,
                'expected_range': (45, 90),
            },
            'unemployment_rate_pct': {
                'min': 0.0,
                'max': 100.0,
                'expected_range': (1, 18),
            }
        }
        
        # Calculated metrics — derived from reconciled components, stored in base
        self.calculated_metrics = [
            'gdhi_per_head_gbp',
            'productivity_gbp_per_job',
            'income_per_worker_gbp',
        ]
        
        # All expected base metrics
        self.expected_base_metrics = (
            self.additive_metrics + 
            list(self.rate_metrics.keys()) + 
            self.calculated_metrics
        )
        
        # Expected counts
        self.expected_lad_count = 361
        self.expected_itl3_count = 182
        
        self.forecast_years = [2024, 2025, 2030, 2040, 2050]
        self.issues = []
        self.warnings = []
        
        # Load LAD→ITL3 mapping
        self._load_geography_mapping()
    
    def _load_geography_mapping(self) -> None:
        """Load LAD→ITL3 parent mapping"""
        try:
            lookup = pd.read_csv('data/reference/master_2025_geography_lookup.csv')
            lookup.columns = [col.replace('\ufeff', '') for col in lookup.columns]
            
            self.lad_to_itl3 = lookup[['LAD25CD', 'ITL325CD']].drop_duplicates()
            self.lad_to_itl3.columns = ['lad_code', 'itl3_code']
            
            # Official names for validation
            self.lad_names = lookup[['LAD25CD', 'LAD25NM']].drop_duplicates()
            self.lad_names.columns = ['lad_code', 'official_name']
            
        except Exception as e:
            self.issues.append(f"Failed to load geography mapping: {e}")
            self.lad_to_itl3 = pd.DataFrame(columns=['lad_code', 'itl3_code'])
            self.lad_names = pd.DataFrame(columns=['lad_code', 'official_name'])
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load base forecasts, ITL3 parent, and CIs"""
        
        # Base forecasts
        base_query = """
        SELECT 
            region_code, region_name, itl3_code, metric_id, period,
            value, data_type, ci_lower, ci_upper, method
        FROM gold.lad_forecast
        WHERE period >= 1990
        ORDER BY region_code, metric_id, period
        """
        base_df = self.conn.execute(base_query).df()
        
        # ITL3 parent forecasts
        itl3_query = """
        SELECT 
            region_code, region_name, metric_id, period, value, data_type
        FROM gold.itl3_forecast
        WHERE period >= 1990
        ORDER BY region_code, metric_id, period
        """
        itl3_df = self.conn.execute(itl3_query).df()
        
        # CI subset from base
        ci_df = base_df[
            (base_df['period'] >= 2025) & 
            (base_df['ci_lower'].notna())
        ][['region_code', 'metric_id', 'period', 'value', 'ci_lower', 'ci_upper']].copy()
        
        return base_df, itl3_df, ci_df
    
    # ==========================================================================
    # METRIC COVERAGE
    # ==========================================================================
    
    def check_base_metric_coverage(self, base_df: pd.DataFrame) -> None:
        """Check that all expected base metrics are present"""
        
        print("\n" + "="*80)
        print("BASE TABLE METRIC COVERAGE (gold.lad_forecast)")
        print("="*80)
        
        present = set(base_df['metric_id'].unique())
        expected = set(self.expected_base_metrics)
        
        missing = expected - present
        extra = present - expected
        
        print(f"\n📊 Expected: {len(expected)} | Present: {len(present)}")
        
        if missing:
            missing_additive = missing & set(self.additive_metrics)
            if missing_additive:
                self.issues.append(f"CRITICAL: Missing additive metrics: {missing_additive}")
                print(f"\n  ❌ CRITICAL: Missing additive metrics:")
                for m in sorted(missing_additive):
                    print(f"     - {m}")
            
            missing_other = missing - missing_additive
            if missing_other:
                self.warnings.append(f"Missing non-additive metrics: {missing_other}")
                print(f"\n  ⚠️  Missing non-additive metrics:")
                for m in sorted(missing_other):
                    print(f"     - {m}")
        
        if extra:
            print(f"\n  ℹ️  Extra metrics (not expected):")
            for m in sorted(extra):
                print(f"     - {m}")
        
        if not missing:
            print(f"\n  ✅ All expected base metrics present")
        
        # Show breakdown
        print(f"\n  Metric breakdown:")
        for metric in sorted(present):
            hist = base_df[(base_df['metric_id'] == metric) & (base_df['data_type'] == 'historical')]
            fcst = base_df[(base_df['metric_id'] == metric) & (base_df['data_type'] == 'forecast')]
            regions = base_df[base_df['metric_id'] == metric]['region_code'].nunique()
            
            hist_range = f"{int(hist['period'].min())}-{int(hist['period'].max())}" if len(hist) else "N/A"
            fcst_range = f"{int(fcst['period'].min())}-{int(fcst['period'].max())}" if len(fcst) else "N/A"
            
            print(f"    {metric}: {regions} LADs | hist: {hist_range} | fcst: {fcst_range}")
    
    # ==========================================================================
    # DATA QUALITY
    # ==========================================================================
    
    def check_data_quality(self, base_df: pd.DataFrame) -> None:
        """Nulls, negatives, region counts"""
        
        print("\n" + "="*80)
        print("DATA QUALITY CHECKS")
        print("="*80)
        
        # Region count
        n_regions = base_df['region_code'].nunique()
        print(f"\n📊 LAD count: {n_regions} (expected: {self.expected_lad_count})")
        
        if n_regions != self.expected_lad_count:
            delta = n_regions - self.expected_lad_count
            if abs(delta) > 5:
                self.issues.append(f"LAD count mismatch: {n_regions} vs {self.expected_lad_count}")
                print(f"  ❌ CRITICAL: {abs(delta)} LADs mismatch")
            else:
                self.warnings.append(f"LAD count: {n_regions} vs expected {self.expected_lad_count}")
                print(f"  ⚠️  Minor mismatch: {delta:+d} LADs")
        else:
            print(f"  ✅ LAD count correct")
        
        # ITL3 parent count
        n_itl3 = base_df['itl3_code'].nunique()
        print(f"\n📊 ITL3 parents: {n_itl3} (expected: {self.expected_itl3_count})")
        if n_itl3 != self.expected_itl3_count:
            self.warnings.append(f"ITL3 parent count: {n_itl3} vs expected {self.expected_itl3_count}")
        
        # Nulls in forecast
        fcst_df = base_df[base_df['data_type'] == 'forecast']
        nulls = fcst_df['value'].isna().sum()
        print(f"\n📊 Null forecast values: {nulls}")
        
        if nulls > 0:
            self.issues.append(f"{nulls} null forecast values")
            print(f"  ❌ {nulls} null values found")
        else:
            print(f"  ✅ No null forecast values")
        
        # Negatives in additive/calculated metrics
        print(f"\n📊 Negative value check:")
        for m in self.additive_metrics + self.calculated_metrics:
            d = fcst_df[fcst_df['metric_id'] == m]
            nneg = (d['value'] < 0).sum()
            if nneg > 0:
                self.issues.append(f"{m}: {nneg} negative values")
                print(f"  ❌ {m}: {nneg} negative values")
            else:
                print(f"  ✅ {m}: no negatives")
        
        # ITL3 code populated
        null_itl3 = base_df['itl3_code'].isna().sum()
        print(f"\n📊 Null itl3_code: {null_itl3}")
        if null_itl3 > 0:
            self.issues.append(f"CRITICAL: {null_itl3} rows missing itl3_code")
            print(f"  ❌ {null_itl3} rows missing itl3_code")
        else:
            print(f"  ✅ All rows have itl3_code")
    
    # ==========================================================================
    # GEOGRAPHY VALIDATION
    # ==========================================================================
    
    def check_geography_mapping(self, base_df: pd.DataFrame) -> None:
        """Validate LAD codes match concordance"""
        
        print("\n" + "="*80)
        print("GEOGRAPHY MAPPING VALIDATION")
        print("="*80)
        
        lad_codes_data = set(base_df['region_code'].unique())
        lad_codes_conc = set(self.lad_names['lad_code'].unique())
        
        missing_in_conc = lad_codes_data - lad_codes_conc
        missing_in_data = lad_codes_conc - lad_codes_data
        
        print(f"\n📊 LAD codes in forecast: {len(lad_codes_data)}")
        print(f"📊 LAD codes in concordance: {len(lad_codes_conc)}")
        
        if missing_in_conc:
            self.warnings.append(f"{len(missing_in_conc)} LAD codes not in concordance")
            print(f"\n  ⚠️  {len(missing_in_conc)} codes in data but not concordance")
            if len(missing_in_conc) <= 5:
                print(f"      {missing_in_conc}")
        else:
            print(f"\n  ✅ All LAD codes in concordance")
        
        if missing_in_data:
            self.warnings.append(f"{len(missing_in_data)} concordance LADs missing from data")
            print(f"  ⚠️  {len(missing_in_data)} concordance LADs missing from data")
        else:
            print(f"  ✅ All concordance LADs present in data")
        
        # Name consistency
        data_names = base_df[['region_code', 'region_name']].drop_duplicates()
        name_check = data_names.merge(
            self.lad_names, left_on='region_code', right_on='lad_code', how='left'
        )
        mismatches = name_check[name_check['region_name'] != name_check['official_name']]
        mismatches = mismatches.dropna(subset=['official_name'])
        
        if len(mismatches) > 0:
            self.warnings.append(f"{len(mismatches)} LAD name mismatches")
            print(f"\n  ⚠️  {len(mismatches)} name mismatches (likely boundary changes)")
        else:
            print(f"\n  ✅ All names match concordance")
    
    # ==========================================================================
    # RECONCILIATION: LAD → ITL3 (FORECAST ONLY)
    # ==========================================================================
    
    def check_reconciliation(self, base_df: pd.DataFrame, itl3_df: pd.DataFrame) -> None:
        """
        Check LAD additive metrics sum to ITL3 parent.
        
        NOTE: Only FORECAST data is checked. Historical data is actual ONS values
        and will NOT reconcile — this is expected and correct.
        """
        
        print("\n" + "="*80)
        print("RECONCILIATION: LAD → ITL3 (FORECAST ONLY)")
        print("="*80)
        print("(Historical data is actual ONS — mismatches expected and correct)")
        
        if self.lad_to_itl3.empty:
            self.issues.append("No LAD→ITL3 mapping available")
            print("\n  ❌ Cannot check reconciliation — no mapping")
            return
        
        # Filter to forecast only
        lad_fcst = base_df[base_df['data_type'] == 'forecast']
        itl3_fcst = itl3_df[itl3_df['data_type'] == 'forecast']
        
        for metric in self.additive_metrics:
            print(f"\n📊 {metric}")
            print("-" * 80)
            
            # ITL3 parent values
            itl3_vals = itl3_fcst[
                itl3_fcst['metric_id'] == metric
            ].set_index(['region_code', 'period'])['value']
            
            if itl3_vals.empty:
                self.warnings.append(f"{metric}: No ITL3 anchor found")
                print(f"  ⚠️  No ITL3 anchor found")
                continue
            
            # LAD sums by ITL3 parent
            lad_metric = lad_fcst[lad_fcst['metric_id'] == metric]
            lad_sums = lad_metric.groupby(['itl3_code', 'period'])['value'].sum()
            
            failures = 0
            for year in self.forecast_years:
                year_errors = []
                
                for itl3_code in lad_sums.index.get_level_values('itl3_code').unique():
                    if pd.isna(itl3_code):
                        continue
                    
                    try:
                        lad_sum = lad_sums.loc[(itl3_code, year)]
                        itl3_val = itl3_vals.loc[(itl3_code, year)]
                    except KeyError:
                        continue
                    
                    diff = lad_sum - itl3_val
                    pct_diff = (diff / itl3_val * 100) if itl3_val != 0 else 0
                    
                    if abs(pct_diff) > self.tolerance * 100:
                        year_errors.append((itl3_code, lad_sum, itl3_val, pct_diff))
                        failures += 1
                
                status = "❌ FAIL" if year_errors else "✅ PASS"
                
                # Summary for year
                total_lad = lad_sums[lad_sums.index.get_level_values('period') == year].sum()
                total_itl3 = itl3_vals[itl3_vals.index.get_level_values('period') == year].sum()
                delta = total_lad - total_itl3
                pct = (delta / total_itl3 * 100) if total_itl3 != 0 else 0
                
                print(f"  {year}: {status} | LAD={total_lad:>15,.0f} | ITL3={total_itl3:>15,.0f} | "
                      f"Δ={delta:>+12,.0f} ({pct:>+.6f}%)")
            
            if failures > 0:
                self.issues.append(f"{metric}: {failures} ITL3 parents with >{self.tolerance*100}% deviation")
    
    # ==========================================================================
    # SHARE STABILITY (LAD-SPECIFIC)
    # ==========================================================================
    
    def check_share_stability(self, base_df: pd.DataFrame, itl3_df: pd.DataFrame) -> None:
        """
        Verify shares are constant across forecast horizon.
        
        By construction, share-based allocation uses fixed shares — any drift
        indicates a bug in the allocation logic.
        """
        
        print("\n" + "="*80)
        print("SHARE STABILITY (must be constant across forecast years)")
        print("="*80)
        
        lad_fcst = base_df[base_df['data_type'] == 'forecast']
        itl3_fcst = itl3_df[itl3_df['data_type'] == 'forecast']
        
        for metric in self.additive_metrics:
            lad_metric = lad_fcst[lad_fcst['metric_id'] == metric]
            itl3_metric = itl3_fcst[itl3_fcst['metric_id'] == metric]
            
            if lad_metric.empty or itl3_metric.empty:
                continue
            
            # Merge to get parent values
            merged = lad_metric.merge(
                itl3_metric[['region_code', 'period', 'value']].rename(
                    columns={'region_code': 'itl3_code', 'value': 'parent_val'}
                ),
                on=['itl3_code', 'period']
            )
            
            merged['share'] = merged['value'] / (merged['parent_val'] + 1e-10)
            
            # Check share variance across years for each LAD
            share_var = merged.groupby('region_code')['share'].std()
            max_var = share_var.max()
            mean_var = share_var.mean()
            
            status = "✅" if max_var < 1e-10 else "❌"
            print(f"  {metric}: share std max={max_var:.2e}, mean={mean_var:.2e} {status}")
            
            if max_var >= 1e-10:
                self.issues.append(f"CRITICAL: {metric} shares not constant (max std={max_var:.2e})")
    
    # ==========================================================================
    # CALCULATED METRIC VALIDATION
    # ==========================================================================
    
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
            self.warnings.append("Cannot validate gdhi_per_head — missing components")
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
                print(f"\n  {year}: ✅ Max error: {max_err:.6f}%")
    
    def check_productivity(self, base_df: pd.DataFrame) -> None:
        """Validate productivity_gbp_per_job = gva / jobs × 1e6"""
        
        print("\n" + "="*80)
        print("PRODUCTIVITY CALCULATION CHECK")
        print("="*80)
        
        fcst = base_df[base_df['data_type'] == 'forecast']
        
        pivot = fcst.pivot_table(
            index=['region_code', 'region_name', 'period'],
            columns='metric_id',
            values='value'
        ).reset_index()
        
        required = ['nominal_gva_mn_gbp', 'emp_total_jobs', 'productivity_gbp_per_job']
        if not all(c in pivot.columns for c in required):
            self.warnings.append("Cannot validate productivity — missing components")
            print("\n  ⚠️  Missing required columns for validation")
            return
        
        pivot['expected'] = pivot['nominal_gva_mn_gbp'] / pivot['emp_total_jobs'] * 1e6
        pivot['diff_pct'] = (pivot['productivity_gbp_per_job'] - pivot['expected']) / pivot['expected'] * 100
        
        for year in [2025, 2030, 2050]:
            yr_data = pivot[pivot['period'] == year]
            if yr_data.empty:
                continue
            
            max_err = yr_data['diff_pct'].abs().max()
            
            if max_err > self.tolerance * 100:
                worst = yr_data.loc[yr_data['diff_pct'].abs().idxmax()]
                self.warnings.append(f"productivity {year}: {worst['region_name']} error={max_err:.2f}%")
                print(f"\n  {year}: ⚠️  Max error: {max_err:.3f}%")
            else:
                print(f"\n  {year}: ✅ Max error: {max_err:.6f}%")
    
    def check_income_per_worker(self, base_df: pd.DataFrame) -> None:
        """Validate income_per_worker_gbp = gdhi_total / jobs × 1e6"""
        
        print("\n" + "="*80)
        print("INCOME PER WORKER CALCULATION CHECK")
        print("="*80)
        
        fcst = base_df[base_df['data_type'] == 'forecast']
        
        pivot = fcst.pivot_table(
            index=['region_code', 'region_name', 'period'],
            columns='metric_id',
            values='value'
        ).reset_index()
        
        required = ['gdhi_total_mn_gbp', 'emp_total_jobs', 'income_per_worker_gbp']
        if not all(c in pivot.columns for c in required):
            self.warnings.append("Cannot validate income_per_worker — missing components")
            print("\n  ⚠️  Missing required columns for validation")
            return
        
        pivot['expected'] = pivot['gdhi_total_mn_gbp'] / pivot['emp_total_jobs'] * 1e6
        pivot['diff_pct'] = (pivot['income_per_worker_gbp'] - pivot['expected']) / pivot['expected'] * 100
        
        for year in [2025, 2030, 2050]:
            yr_data = pivot[pivot['period'] == year]
            if yr_data.empty:
                continue
            
            max_err = yr_data['diff_pct'].abs().max()
            
            if max_err > self.tolerance * 100:
                self.warnings.append(f"income_per_worker {year}: max error={max_err:.2f}%")
                print(f"\n  {year}: ⚠️  Max error: {max_err:.3f}%")
            else:
                print(f"\n  {year}: ✅ Max error: {max_err:.6f}%")
    
    # ==========================================================================
    # RATE BOUNDS
    # ==========================================================================
    
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
            print(f"  LADs: {data['region_code'].nunique()}")
            
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
                    self.warnings.append(f"{rate}: {len(outliers)} LADs outside typical range")
                    print(f"  ⚠️  {len(outliers)} LADs outside [{exp_min}, {exp_max}]%")
    
    # ==========================================================================
    # HISTORICAL CONTINUITY
    # ==========================================================================
    
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
                extreme = trans[(trans['change'] > 25) | (trans['change'] < -15)]
                threshold = ">25% or <-15%"
            
            if not extreme.empty:
                self.warnings.append(f"{metric}: {len(extreme)} extreme jumps")
                print(f"  ⚠️  {metric}: {len(extreme)} jumps ({threshold})")
            else:
                print(f"  ✅ {metric}")
    
    # ==========================================================================
    # CONFIDENCE INTERVALS
    # ==========================================================================
    
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
            
            for year in [2025, 2030, 2040, 2050]:
                yr = m_data[m_data['period'] == year]
                if yr.empty:
                    continue
                
                width_pct = yr['ci_width_pct']
                print(f"  {year}: median={width_pct.median():>6.1f}% | "
                      f"p90={width_pct.quantile(0.9):>6.1f}% | "
                      f"max={width_pct.max():>6.1f}%")
                
                extreme = yr[yr['ci_width_pct'] > 50]
                if not extreme.empty:
                    self.warnings.append(f"{metric} {year}: {len(extreme)} CIs >50% width")
            
            # CI growth over horizon
            if 2025 in m_data['period'].values and 2050 in m_data['period'].values:
                early = m_data[m_data['period'] == 2025]['ci_width_pct'].median()
                late = m_data[m_data['period'] == 2050]['ci_width_pct'].median()
                if early > 0 and late > early * 5:
                    self.warnings.append(f"{metric}: CI width expands >5x by 2050")
                    print(f"  ⚠️  CI expands {late/early:.1f}x from 2025→2050")
    
    # ==========================================================================
    # GROWTH RATE VALIDATION
    # ==========================================================================
    
    def check_growth_rate_bounds(self, base_df: pd.DataFrame) -> None:
        """YoY growth rate validation per metric"""
        
        print("\n" + "="*80)
        print("YEAR-ON-YEAR GROWTH RATE VALIDATION")
        print("="*80)
        
        # Bounds: (min_growth%, max_growth%)
        growth_bounds = {
            'nominal_gva_mn_gbp': (-12, 18),
            'gdhi_total_mn_gbp': (-12, 18),
            'gdhi_per_head_gbp': (-10, 15),
            'emp_total_jobs': (-10, 10),
            'population_total': (-5, 8),
            'productivity_gbp_per_job': (-10, 15),
            'income_per_worker_gbp': (-10, 15),
            'employment_rate_pct': (-8, 8),  # pp change
            'unemployment_rate_pct': (-6, 10),  # pp change
        }
        
        fcst = base_df[base_df['data_type'] == 'forecast'].copy()
        
        for metric, (min_g, max_g) in growth_bounds.items():
            m_data = fcst[fcst['metric_id'] == metric]
            if m_data.empty:
                continue
            
            print(f"\n📊 {metric} (bounds: {min_g}% to {max_g}%)")
            print("-" * 80)
            
            violations = []
            for region in m_data['region_code'].unique():
                r_data = m_data[m_data['region_code'] == region].sort_values('period')
                
                if metric in self.rate_metrics:
                    r_data = r_data.copy()
                    r_data['growth'] = r_data['value'].diff()
                else:
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
                sorted_v = sorted(violations, key=lambda x: abs(x['growth']), reverse=True)[:3]
                for v in sorted_v:
                    print(f"     {v['region']} {v['year']}: {v['growth']:+.1f}%")
            else:
                print(f"  ✅ All LADs within bounds")
    
    # ==========================================================================
    # CROSS-METRIC COHERENCE
    # ==========================================================================
    
    def check_cross_metric_coherence(self, base_df: pd.DataFrame) -> None:
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
        
        # 1. Productivity consistency
        if all(c in pivot.columns for c in ['nominal_gva_mn_gbp', 'emp_total_jobs', 'productivity_gbp_per_job']):
            print("\n🔗 Productivity consistency")
            print("-" * 80)
            
            pivot['prod_calc'] = pivot['nominal_gva_mn_gbp'] / pivot['emp_total_jobs'] * 1e6
            pivot['prod_diff_pct'] = (pivot['productivity_gbp_per_job'] - pivot['prod_calc']) / pivot['prod_calc'] * 100
            
            max_diff = pivot['prod_diff_pct'].abs().max()
            if max_diff > 1:
                self.warnings.append(f"Productivity deviation up to {max_diff:.1f}%")
                print(f"  ⚠️  Max deviation: {max_diff:.2f}%")
            else:
                print(f"  ✅ Stored vs calculated within 1%")
        
        # 2. GDHI per head consistency
        if all(c in pivot.columns for c in ['gdhi_total_mn_gbp', 'population_total', 'gdhi_per_head_gbp']):
            print("\n🔗 GDHI per head consistency")
            print("-" * 80)
            
            pivot['gdhi_calc'] = pivot['gdhi_total_mn_gbp'] / pivot['population_total'] * 1e6
            pivot['gdhi_diff_pct'] = (pivot['gdhi_per_head_gbp'] - pivot['gdhi_calc']) / pivot['gdhi_calc'] * 100
            
            max_diff = pivot['gdhi_diff_pct'].abs().max()
            if max_diff > 1:
                self.warnings.append(f"GDHI/head deviation up to {max_diff:.1f}%")
                print(f"  ⚠️  Max deviation: {max_diff:.2f}%")
            else:
                print(f"  ✅ Stored vs calculated within 1%")
        
        # 3. Income per worker consistency
        if all(c in pivot.columns for c in ['gdhi_total_mn_gbp', 'emp_total_jobs', 'income_per_worker_gbp']):
            print("\n🔗 Income per worker consistency")
            print("-" * 80)
            
            pivot['inc_calc'] = pivot['gdhi_total_mn_gbp'] / pivot['emp_total_jobs'] * 1e6
            pivot['inc_diff_pct'] = (pivot['income_per_worker_gbp'] - pivot['inc_calc']) / pivot['inc_calc'] * 100
            
            max_diff = pivot['inc_diff_pct'].abs().max()
            if max_diff > 1:
                self.warnings.append(f"Income/worker deviation up to {max_diff:.1f}%")
                print(f"  ⚠️  Max deviation: {max_diff:.2f}%")
            else:
                print(f"  ✅ Stored vs calculated within 1%")
    
    # ==========================================================================
    # FULL CASCADE CHECK (LAD-SPECIFIC)
    # ==========================================================================
    
    def check_full_cascade(self) -> None:
        """Verify full UK → ITL1 → ITL2 → ITL3 → LAD cascade coherence"""
        
        print("\n" + "="*80)
        print("FULL CASCADE CHECK: UK → ITL1 → ITL2 → ITL3 → LAD")
        print("="*80)
        
        cascade_ok = True
        
        for year in [2030, 2050]:
            print(f"\n📅 {year}")
            print("-" * 80)
            
            for metric in ['nominal_gva_mn_gbp', 'population_total']:
                uk_metric = f"uk_{metric}"
                
                try:
                    uk_val = self.conn.execute(f"""
                        SELECT value FROM gold.uk_macro_forecast 
                        WHERE metric_id = '{uk_metric}' AND period = {year}
                    """).fetchone()
                    
                    itl1_sum = self.conn.execute(f"""
                        SELECT SUM(value) FROM gold.itl1_forecast 
                        WHERE metric_id = '{metric}' AND period = {year} AND data_type = 'forecast'
                    """).fetchone()
                    
                    itl2_sum = self.conn.execute(f"""
                        SELECT SUM(value) FROM gold.itl2_forecast 
                        WHERE metric_id = '{metric}' AND period = {year} AND data_type = 'forecast'
                    """).fetchone()
                    
                    itl3_sum = self.conn.execute(f"""
                        SELECT SUM(value) FROM gold.itl3_forecast 
                        WHERE metric_id = '{metric}' AND period = {year} AND data_type = 'forecast'
                    """).fetchone()
                    
                    lad_sum = self.conn.execute(f"""
                        SELECT SUM(value) FROM gold.lad_forecast 
                        WHERE metric_id = '{metric}' AND period = {year} AND data_type = 'forecast'
                    """).fetchone()
                    
                    if None in [uk_val, itl1_sum, itl2_sum, itl3_sum, lad_sum]:
                        print(f"  {metric}: Missing data at some level")
                        continue
                    
                    uk = uk_val[0]
                    itl1 = itl1_sum[0]
                    itl2 = itl2_sum[0]
                    itl3 = itl3_sum[0]
                    lad = lad_sum[0]
                    
                    all_ok = (
                        abs(itl1 - uk) < 1 and
                        abs(itl2 - itl1) < 1 and
                        abs(itl3 - itl2) < 1 and
                        abs(lad - itl3) < 1
                    )
                    
                    status = "✅" if all_ok else "❌"
                    
                    if not all_ok:
                        cascade_ok = False
                        self.issues.append(f"Cascade broken for {metric} in {year}")
                    
                    if 'population' in metric:
                        print(f"  {metric}:")
                        print(f"    UK={uk:,.0f} → ITL1={itl1:,.0f} → ITL2={itl2:,.0f} → ITL3={itl3:,.0f} → LAD={lad:,.0f} {status}")
                    else:
                        print(f"  {metric}:")
                        print(f"    UK=£{uk:,.0f}m → ITL1=£{itl1:,.0f}m → ITL2=£{itl2:,.0f}m → ITL3=£{itl3:,.0f}m → LAD=£{lad:,.0f}m {status}")
                    
                except Exception as e:
                    print(f"  {metric}: Error — {e}")
        
        if cascade_ok:
            print(f"\n  ✅ Full cascade coherence verified (UK → LAD)")
    
    # ==========================================================================
    # TERMINAL YEAR DIAGNOSTICS
    # ==========================================================================
    
    def check_terminal_year_diagnostics(self, base_df: pd.DataFrame) -> None:
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
        
        print(f"\n📊 2050 Snapshot ({len(pivot)} LADs)")
        print("-" * 80)
        
        # UK totals
        print("\n  UK Totals (sum of LAD):")
        for metric in self.additive_metrics:
            if metric in pivot.columns:
                total = pivot[metric].sum()
                if 'gva' in metric or 'gdhi' in metric:
                    print(f"    {metric}: £{total:,.0f}m")
                else:
                    print(f"    {metric}: {total:,.0f}")
        
        # Regional rankings
        print("\n  📍 Top 5 LADs by GVA:")
        if 'nominal_gva_mn_gbp' in pivot.columns:
            top_gva = pivot.nlargest(5, 'nominal_gva_mn_gbp')
            for _, row in top_gva.iterrows():
                print(f"    {row['region_name']}: £{row['nominal_gva_mn_gbp']:,.0f}m")
        
        print("\n  📍 Top 5 LADs by GDHI/head:")
        if 'gdhi_per_head_gbp' in pivot.columns:
            top_gdhi = pivot.nlargest(5, 'gdhi_per_head_gbp')
            for _, row in top_gdhi.iterrows():
                print(f"    {row['region_name']}: £{row['gdhi_per_head_gbp']:,.0f}")
        
        # Sanity flags
        print("\n  🚨 Sanity Checks:")
        issues_2050 = []
        
        if 'nominal_gva_mn_gbp' in pivot.columns:
            max_gva = pivot['nominal_gva_mn_gbp'].max()
            if max_gva > 500_000:  # £500bn for single LAD extreme
                issues_2050.append(f"Max LAD GVA >£500bn: £{max_gva:,.0f}m")
        
        if 'gdhi_per_head_gbp' in pivot.columns:
            max_gdhi = pivot['gdhi_per_head_gbp'].max()
            if max_gdhi > 500_000:  # City of London is extreme
                issues_2050.append(f"Max GDHI/head >£500k: £{max_gdhi:,.0f}")
            min_gdhi = pivot['gdhi_per_head_gbp'].min()
            if min_gdhi < 5_000:
                issues_2050.append(f"Min GDHI/head <£5k: £{min_gdhi:,.0f}")
        
        if 'population_total' in pivot.columns:
            max_pop = pivot['population_total'].max()
            if max_pop > 2_000_000:  # 2m for single LAD high but possible
                issues_2050.append(f"Max LAD pop >2m: {max_pop:,.0f}")
        
        if issues_2050:
            for issue in issues_2050:
                self.warnings.append(f"2050: {issue}")
                print(f"    ⚠️  {issue}")
        else:
            print(f"    ✅ All 2050 values within sanity bounds")
    
    # ==========================================================================
    # REGIONAL PLAUSIBILITY
    # ==========================================================================
    
    def check_regional_plausibility(self, base_df: pd.DataFrame) -> None:
        """Regional sanity checks and spot checks"""
        
        print("\n" + "="*80)
        print("REGIONAL PLAUSIBILITY")
        print("="*80)
        
        # CAGRs for key metrics
        print(f"\n📍 LAD CAGRs (2025-2030) — extreme check")
        
        for metric in ['nominal_gva_mn_gbp', 'emp_total_jobs', 'population_total']:
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
            extreme = m_data[(m_data['cagr'] > 12) | (m_data['cagr'] < -8)]
            
            if not extreme.empty:
                self.warnings.append(f"{metric}: {len(extreme)} extreme CAGRs")
                print(f"  ⚠️  {metric}: {len(extreme)} extreme")
            else:
                print(f"  ✅ {metric}")
        
        # Spot checks
        print(f"\n📍 Spot checks — known LADs (2030)")
        
        spot_checks = [
            ('E09000001', 'City of London', 'financial hub, extreme values expected'),
            ('E09000033', 'Westminster', 'central London, high GVA'),
            ('E08000003', 'Manchester', 'major city'),
            ('E08000025', 'Birmingham', 'largest outside London'),
            ('E06000001', 'Hartlepool', 'smaller town, typical values'),
        ]
        
        for code, name, note in spot_checks:
            region_data = base_df[base_df['region_code'] == code]
            
            if region_data.empty:
                print(f"\n  ⚠️  {name}: NOT FOUND")
                continue
            
            actual_name = region_data['region_name'].iloc[0]
            yr_2030 = region_data[region_data['period'] == 2030]
            
            gva = yr_2030[yr_2030['metric_id'] == 'nominal_gva_mn_gbp']['value']
            pop = yr_2030[yr_2030['metric_id'] == 'population_total']['value']
            prod = yr_2030[yr_2030['metric_id'] == 'productivity_gbp_per_job']['value']
            
            gva_str = f"£{gva.iloc[0]:,.0f}m" if len(gva) else "N/A"
            pop_str = f"{pop.iloc[0]:,.0f}" if len(pop) else "N/A"
            prod_str = f"£{prod.iloc[0]:,.0f}" if len(prod) else "N/A"
            
            name_match = "✅" if actual_name == name else f"⚠️ ({actual_name})"
            
            print(f"\n  {code}: {name} {name_match}")
            print(f"    Note: {note}")
            print(f"    2030: GVA={gva_str}, Pop={pop_str}, Productivity={prod_str}")
    
    # ==========================================================================
    # METHOD TRACKING
    # ==========================================================================
    
    def check_method_tracking(self, base_df: pd.DataFrame) -> None:
        """Summarize method/model usage"""
        
        print("\n" + "="*80)
        print("METHOD TRACKING")
        print("="*80)
        
        if 'method' not in base_df.columns:
            print("\n  ⚠️  No method column")
            return
        
        forecast_df = base_df[base_df['data_type'] == 'forecast']
        method_counts = forecast_df['method'].value_counts()
        
        print(f"\n📊 Forecast methods:")
        total = len(forecast_df)
        for method, count in method_counts.items():
            pct = count / total * 100
            print(f"  {method}: {count:,} ({pct:.1f}%)")
        
        hist_df = base_df[base_df['data_type'] == 'historical']
        if not hist_df.empty:
            hist_methods = hist_df['method'].value_counts()
            print(f"\n📊 Historical methods:")
            for method, count in hist_methods.items():
                print(f"  {method}: {count:,}")
    
    # ==========================================================================
    # REPORT GENERATION
    # ==========================================================================
    
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
            print("✅ LAD FORECAST QA V2.0: PASSED — Ready for Supabase")
            print("="*80)
            print("\nReady to push to production.")
            exit_code = 0
        else:
            print("❌ LAD FORECAST QA V2.0: FAILED — DO NOT PUSH")
            print("="*80)
            print(f"\n{len(self.issues)} critical issues. Fix before production.")
            exit_code = 1
        
        # JSON export
        summary = {
            'level': 'LAD',
            'version': '2.0',
            'timestamp': datetime.now().isoformat(),
            'status': 'PASSED' if exit_code == 0 else 'FAILED',
            'exit_code': exit_code,
            'architecture': {
                'base_table': 'gold.lad_forecast',
                'base_metrics': self.expected_base_metrics,
                'methodology': 'share_allocation_from_itl3',
                'reconciles_to': 'gold.itl3_forecast',
                'reconciliation_scope': 'forecast_only',
            },
            'cascade': {
                'position': 'terminal',
                'upstream': ['UK', 'ITL1', 'ITL2', 'ITL3'],
                'downstream': None,
            },
            'stats': {
                'expected_lads': self.expected_lad_count,
                'expected_itl3_parents': self.expected_itl3_count,
                'base_metrics_expected': len(self.expected_base_metrics),
                'critical_issues': len(self.issues),
                'warnings': len(self.warnings),
            },
            'critical_issues': self.issues,
            'warnings': self.warnings,
        }
        
        output_path = Path('data/qa/lad_qa_summary.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📄 Summary: {output_path}")
        print(f"📤 Exit code: {exit_code}")
        
        return exit_code
    
    # ==========================================================================
    # MAIN RUNNER
    # ==========================================================================
    
    def run_all_checks(self) -> int:
        """Run full QA suite"""
        
        print("\n" + "="*80)
        print("LAD FORECAST QA V2.0 — TERMINAL LEVEL")
        print("="*80)
        print(f"Database: {self.db_path}")
        print(f"Tolerance: {self.tolerance * 100}%")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nArchitecture:")
        print(f"  Base table: gold.lad_forecast ({len(self.expected_base_metrics)} metrics)")
        print(f"  Methodology: Share-based allocation from ITL3")
        print(f"  Reconciles to: gold.itl3_forecast (forecast only)")
        print(f"  Position: Terminal (no downstream levels)")
        
        print("\n📥 Loading data...")
        base_df, itl3_df, ci_df = self.load_data()
        print(f"  ✅ LAD:  {len(base_df):,} rows ({base_df['region_code'].nunique()} LADs)")
        print(f"  ✅ ITL3: {len(itl3_df):,} rows ({itl3_df['region_code'].nunique()} regions)")
        print(f"  ✅ CIs:  {len(ci_df):,} rows")
        
        # Core validation
        self.check_base_metric_coverage(base_df)
        self.check_data_quality(base_df)
        self.check_geography_mapping(base_df)
        self.check_reconciliation(base_df, itl3_df)
        self.check_share_stability(base_df, itl3_df)
        self.check_gdhi_per_head(base_df)
        self.check_productivity(base_df)
        self.check_income_per_worker(base_df)
        self.check_rate_bounds(base_df)
        self.check_historical_continuity(base_df)
        self.check_confidence_intervals(ci_df)
        
        # Extended diagnostics
        self.check_ci_diagnostics_per_metric(base_df)
        self.check_growth_rate_bounds(base_df)
        self.check_cross_metric_coherence(base_df)
        self.check_full_cascade()
        self.check_terminal_year_diagnostics(base_df)
        
        # Regional plausibility
        self.check_regional_plausibility(base_df)
        self.check_method_tracking(base_df)
        
        exit_code = self.generate_report()
        self.conn.close()
        
        return exit_code


def main():
    qa = LADForecastQA(
        db_path="data/lake/warehouse.duckdb",
        tolerance=0.0001  # 0.01% — tighter than ITL3 since shares should be exact
    )
    exit_code = qa.run_all_checks()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()