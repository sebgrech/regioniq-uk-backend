# Northern Ireland LGD Labour Metrics (NISRA PxStat)

## Sources
- Rates & people: `LMSLGD` (Labour Market Status), NISRA PxStat  
  - `EMPR` → `employment_rate_pct` (16–64, %)
  - `UNEMPR` → `unemployment_rate_pct` (16+, %)
- Jobs: `BRESHEADLGD` (Employee Jobs), NISRA PxStat  
  - `EJOBS` → `emp_total_jobs_ni` (employee jobs, NI only)

## Geography
- `LGD2014` codes `N09000001` … `N09000011` (NI Local Government Districts)  
- NI total `N92000002` is **excluded** from LAD-level outputs.
- Treated as `region_level = "LAD"`; country flag = NI.

## Semantics
- **Do NOT merge** NI jobs into `emp_total_jobs` (GB BRES “total jobs”).  
  - Keep NI jobs as `emp_total_jobs_ni` (employee jobs only, NISRA).
- Rates are people-based (LFS), consistent with GB APS rates conceptually.

## Export / UI notes
- “Employment (jobs)” should display NI rows with an info note:  
  “NI: employee jobs (NISRA, BRESHEADLGD); GB: total employee jobs (ONS/NOMIS BRES).”
- Rollups that mix incompatible job definitions should exclude NI from UK/GB totals.

## Forecast policy (V1)
- NI rates/people: flat carry (last observed → 2025–2050) via `ni_flat_carry`.
- NI jobs: `emp_total_jobs_ni` is NI-only; forecasting/reconciliation is handled inside the standard ITL/LAD pipeline without mixing into GB/UK totals.

