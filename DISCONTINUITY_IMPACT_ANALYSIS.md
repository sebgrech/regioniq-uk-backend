# Discontinuity Impact Analysis

## Scope of the Problem

The discontinuity issue affects **the majority of LADs**, not just Aberdeenshire:

### Population Total
- **53.5%** of LADs (193 out of 361) have jumps >0.5%
- **28.0%** of LADs (101 out of 361) have jumps >1.0%
- **8.9%** of LADs (32 out of 361) have jumps >2.0%
- **Mean absolute jump**: 0.81%
- **Worst case**: +4.16% (E09000027 - Richmond upon Thames)

### Population 16-64
- **67.0%** of LADs (242 out of 361) have jumps >0.5%
- **41.3%** of LADs (149 out of 361) have jumps >1.0%
- **13.3%** of LADs (48 out of 361) have jumps >2.0%
- **Mean absolute jump**: 1.00%
- **Worst case**: +4.70% (E09000020 - Kingston upon Thames)

## Why This Happens

The discontinuity occurs because:

1. **Fixed 5-year average share** is used for all forecast years
2. **Actual share in last historical year** (2024) often differs from this average
3. **Share drift**: LADs' shares within their ITL3 parents can change over time
4. **No continuity check**: The original method doesn't ensure the first forecast year matches the last historical year

### Example (Aberdeenshire)
- Fixed share (5-year avg): 0.5384
- Actual 2024 share: 0.5335
- Difference: 0.49 percentage points → causes 1.48% jump

## Impact of the Fix

**Yes, the continuity adjustment will change EVERY LAD that has this issue** - which is the majority of them.

### What Changes
1. **All LADs with historical data** get a continuity check
2. **If there's a mismatch** (>0.1% threshold), an adjustment factor is calculated
3. **All forecast years** for that LAD/metric are scaled by this factor
4. **Result**: Smooth transition from historical to forecast

### Why This is Good
- ✅ Eliminates artificial jumps in forecasts
- ✅ More realistic and credible forecasts
- ✅ Better user experience (no visual discontinuities in charts)
- ✅ Preserves growth trajectory (still follows ITL3 growth)

### Trade-offs
- ⚠️ **Reconciliation**: Sum of LADs may not exactly equal ITL3 (but within 0.1%)
- ⚠️ **Share drift**: The adjustment effectively accounts for share changes that the fixed share method ignores

## Which LADs Are Most Affected?

### Highest Positive Jumps (will see biggest reductions)
- E09000027 (Richmond upon Thames): +4.16% → will be smoothed
- E09000010 (Hackney): +4.03% → will be smoothed
- E09000020 (Kingston upon Thames): +3.94% → will be smoothed

### Highest Negative Jumps (will see biggest increases)
- E08000021 (Newcastle upon Tyne): -3.79% → will be smoothed
- E08000024 (North Tyneside): -2.87% → will be smoothed
- E06000001 (Hartlepool): -2.75% → will be smoothed

## Recommendation

**This fix should be applied** because:
1. It affects the majority of LADs (50-67% depending on metric)
2. The jumps are significant (mean 0.8-1.0%, some >4%)
3. The fix is mathematically sound (preserves growth trajectory)
4. The reconciliation impact is minimal (<0.1% deviation)

The continuity adjustment ensures forecasts start from the actual last observed value, which is the correct approach for time series forecasting.

