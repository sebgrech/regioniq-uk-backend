#!/usr/bin/env python3
"""
RegionIQ Weekly Pipeline Email Report (v3 - Dynamic)

Fully dynamic report - queries DuckDB for indicator names and geography counts.
No hardcoded mappings.

Usage:
    python3 scripts/notify/weekly_email.py --run-id 20251130_224151
    python3 scripts/notify/weekly_email.py --run-id 20251130_224151 --dry-run
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, List

try:
    import requests  # type: ignore
except ImportError:
    # Optional: only required for actually sending email (not for --dry-run preview)
    requests = None  # type: ignore

try:
    import duckdb  # type: ignore
except ImportError:
    # Optional: used for nicer dynamic metadata; report still works with fallbacks
    duckdb = None  # type: ignore

try:
    import psycopg2  # type: ignore
except ImportError:
    psycopg2 = None  # type: ignore

try:
    # Load dotenv from common repo layouts.
    # We support both:
    #   - repo_root/.env
    #   - repo_root/backend/.env
    from dotenv import load_dotenv  # type: ignore

    backend_root = Path(__file__).resolve().parents[2]  # .../backend
    candidates = [
        backend_root / ".env",
        backend_root.parent / ".env",
    ]
    _DOTENV_LOADED_FROM = None
    for dotenv_path in candidates:
        if dotenv_path.exists():
            # Do not override already-exported env vars (prod should win)
            load_dotenv(dotenv_path=dotenv_path, override=False)
            _DOTENV_LOADED_FROM = str(dotenv_path)
            break
except Exception:
    # If python-dotenv isn't installed or filesystem access is restricted, just continue.
    pass

import base64

# -----------------------------
# Configuration
# -----------------------------
PIPELINE_DIR = Path("data/pipeline")
LOG_DIR = Path("data/logs")
DUCK_PATH = Path("data/lake/warehouse.duckdb")

RESEND_API_URL = "https://api.resend.com/emails"
FROM_EMAIL = "RegionIQ <onboarding@resend.dev>"

# Supabase Postgres connection string (used to fetch email recipients).
# In production this should come from systemd/cron env (not .env).
SUPABASE_URI = os.environ.get("SUPABASE_URI", "")

# Pipeline mode (local/bootstrap/prod). In prod, missing Supabase/Resend config should fail loudly.
PIPELINE_MODE = os.environ.get("PIPELINE_MODE", "prod").lower()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("weekly_email")


# -----------------------------
# Logo Handling
# -----------------------------

def get_logo_base64() -> str:
    """Load logo and return base64 data URI."""
    # Support both repo_root/assets and backend/assets layouts.
    backend_root = Path(__file__).resolve().parents[2]
    candidates = [
        Path("assets/logo.png"),
        backend_root / "assets" / "logo.png",
        backend_root.parent / "assets" / "logo.png",
    ]
    logo_path = next((p for p in candidates if p.exists()), None)
    if not logo_path:
        log.warning("Logo not found (searched: %s)", ", ".join(str(p) for p in candidates))
        return ""
    try:
        with open(logo_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{data}"
    except Exception as e:
        log.warning(f"Failed to load logo: {e}")
        return ""


# -----------------------------
# Dynamic Metadata from DuckDB
# -----------------------------

def get_indicator_names() -> Dict[str, str]:
    """
    Query metadata.indicators for NOMIS dataset → display name mapping.
    Returns dict like {'NM_185_1': 'Household Income (GDHI)', ...}
    """
    if duckdb is None or not DUCK_PATH.exists():
        log.warning("DuckDB not found, using fallback names")
        return {}
    
    try:
        con = duckdb.connect(str(DUCK_PATH), read_only=True)
        df = con.execute("""
            SELECT DISTINCT nomis_datasets, display_name 
            FROM metadata.indicators 
            WHERE nomis_datasets != 'derived'
        """).fetchdf()
        con.close()
        
        # Build mapping (handle comma-separated datasets)
        mapping = {}
        for _, row in df.iterrows():
            datasets = row['nomis_datasets'].split(',')
            for ds in datasets:
                ds = ds.strip()
                if ds and ds not in mapping:
                    mapping[ds] = row['display_name']
                    # Add variants for rate datasets
                    if ds == 'NM_17_5':
                        mapping['NM_17_5_emp_rate'] = 'Employment Rate'
                        mapping['NM_17_5_unemp_rate'] = 'Unemployment Rate'
        
        log.info(f"  Loaded {len(mapping)} indicator names from DuckDB")
        return mapping
        
    except Exception as e:
        log.warning(f"Failed to load indicator names: {e}")
        return {}


def get_metric_display_names() -> Dict[str, str]:
    """
    Query metadata.indicators for metric_id → display_name mapping.
    Used for baseline "latest year" matrix labels.
    """
    if duckdb is None or not DUCK_PATH.exists():
        return {}

    try:
        con = duckdb.connect(str(DUCK_PATH), read_only=True)
        df = con.execute("""
            SELECT metric_id, display_name
            FROM metadata.indicators
        """).fetchdf()
        con.close()
        mapping: Dict[str, str] = {}
        for _, row in df.iterrows():
            mid = str(row.get("metric_id") or "").strip()
            dn = str(row.get("display_name") or "").strip()
            if mid and dn and mid not in mapping:
                mapping[mid] = dn
        return mapping
    except Exception as e:
        log.warning(f"Failed to load metric display names: {e}")
        return {}


def get_table_labels() -> Dict[str, str]:
    """
    Generate human-readable table labels.
    """
    return {
        'silver.uk_macro_history': 'UK Macro Data',
        'silver.itl1_history': 'Regional Data (ITL1)',
        'silver.itl2_history': 'Sub-Regional Data (ITL2)',
        'silver.itl3_history': 'Local Area Data (ITL3)',
        'silver.lad_history': 'District Data (LAD)',
        'gold.uk_macro_forecast': 'UK Forecasts',
        'gold.itl1_forecast': 'Regional Forecasts (ITL1)',
        'gold.itl2_forecast': 'Sub-Regional Forecasts (ITL2)',
        'gold.itl3_forecast': 'Local Area Forecasts (ITL3)',
        'gold.lad_forecast': 'District Forecasts (LAD)',
    }


def get_geography_counts() -> Dict[str, Dict]:
    """
    Query gold tables for actual geography counts.
    Returns dict like {'UK': {'count': 1, 'label': 'UK Macro'}, ...}
    """
    if duckdb is None or not DUCK_PATH.exists():
        log.warning("DuckDB not found, using fallback counts")
        return _fallback_geo_counts()
    
    try:
        con = duckdb.connect(str(DUCK_PATH), read_only=True)
        
        counts = {}
        
        # UK Macro
        counts['UK'] = {'count': 1, 'label': 'UK Macro'}
        
        # ITL1
        try:
            n = con.execute("SELECT COUNT(DISTINCT region_code) FROM gold.itl1_forecast").fetchone()[0]
            counts['ITL1'] = {'count': n, 'label': 'Regions'}
        except:
            counts['ITL1'] = {'count': 12, 'label': 'Regions'}
        
        # ITL2
        try:
            n = con.execute("SELECT COUNT(DISTINCT region_code) FROM gold.itl2_forecast").fetchone()[0]
            counts['ITL2'] = {'count': n, 'label': 'Sub-Regions'}
        except:
            counts['ITL2'] = {'count': 46, 'label': 'Sub-Regions'}
        
        # ITL3
        try:
            n = con.execute("SELECT COUNT(DISTINCT region_code) FROM gold.itl3_forecast").fetchone()[0]
            counts['ITL3'] = {'count': n, 'label': 'Local Areas'}
        except:
            counts['ITL3'] = {'count': 182, 'label': 'Local Areas'}
        
        # LAD
        try:
            n = con.execute("SELECT COUNT(DISTINCT region_code) FROM gold.lad_forecast").fetchone()[0]
            counts['LAD'] = {'count': n, 'label': 'Districts'}
        except:
            counts['LAD'] = {'count': 361, 'label': 'Districts'}
        
        con.close()
        counts_str = ", ".join(f"{k}={v['count']}" for k, v in counts.items())
        log.info(f"  Loaded geography counts: {counts_str}")
        return counts
        
    except Exception as e:
        log.warning(f"Failed to load geography counts: {e}")
        return _fallback_geo_counts()


def _fallback_geo_counts() -> Dict[str, Dict]:
    """Fallback geography counts if DuckDB query fails."""
    return {
        'UK': {'count': 1, 'label': 'UK Macro'},
        'ITL1': {'count': 12, 'label': 'Regions'},
        'ITL2': {'count': 46, 'label': 'Sub-Regions'},
        'ITL3': {'count': 182, 'label': 'Local Areas'},
        'LAD': {'count': 361, 'label': 'Districts'},
    }


def get_indicator_coverage() -> Dict[str, int]:
    """
    Query gold tables for indicator counts per geography level.
    """
    if duckdb is None or not DUCK_PATH.exists():
        return {}
    
    try:
        con = duckdb.connect(str(DUCK_PATH), read_only=True)
        
        coverage = {}
        for level, table in [('UK', 'uk_macro_forecast'), ('ITL1', 'itl1_forecast'), 
                              ('ITL2', 'itl2_forecast'), ('ITL3', 'itl3_forecast'), 
                              ('LAD', 'lad_forecast')]:
            try:
                n = con.execute(f"SELECT COUNT(DISTINCT metric_id) FROM gold.{table}").fetchone()[0]
                coverage[level] = n
            except:
                pass
        
        con.close()
        return coverage
        
    except Exception as e:
        log.warning(f"Failed to load indicator coverage: {e}")
        return {}


# -----------------------------
# Data Loading
# -----------------------------

def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        log.error(f"Failed to load {path}: {e}")
        return None


def load_vintage_sources() -> Dict[str, Dict]:
    sources = {}
    for name in ['macro', 'itl1', 'lad']:
        path = PIPELINE_DIR / f"vintage_{name}.json"
        data = load_json(path)
        if data:
            sources[name] = data
    return sources


def load_run_data(run_id: str) -> Dict:
    run_dir = PIPELINE_DIR / run_id
    log_dir = LOG_DIR / f"pipeline_{run_id}"
    
    return {
        'pre_vintage': load_json(run_dir / "pre_vintage.json"),
        'vintage_diff': load_json(run_dir / "vintage_diff.json"),
        'pipeline_summary': load_json(log_dir / "pipeline_summary.json"),
    }


# -----------------------------
# Recipient loading (Supabase)
# -----------------------------

def _is_lenient_mode() -> bool:
    return PIPELINE_MODE in ("local", "bootstrap")


def fetch_recipients_from_supabase(pipeline_success: bool) -> List[str]:
    """
    Fetch email recipients from Supabase Postgres table: public.notification_recipients.

    Option A behaviour:
      - enabled=true users receive weekly reports
      - failures_only=true users receive emails only when pipeline_success is False
    """
    if not SUPABASE_URI:
        raise ValueError("SUPABASE_URI not set")
    if psycopg2 is None:
        raise ImportError("psycopg2 is required to fetch recipients from Supabase")

    # If pipeline succeeded, exclude failures_only recipients.
    # If pipeline failed, include both weekly_report recipients and failures_only recipients.
    query = """
        SELECT email
        FROM public.notification_recipients
        WHERE enabled = TRUE
          AND (
                weekly_report = TRUE
                OR (failures_only = TRUE AND %s = FALSE)
              )
        ORDER BY email;
    """

    with psycopg2.connect(SUPABASE_URI) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (pipeline_success,))
            rows = cur.fetchall()

    # Basic cleanup + dedupe
    emails = []
    seen = set()
    for r in rows:
        if not r:
            continue
        email = (r[0] or "").strip()
        if not email:
            continue
        if email in seen:
            continue
        seen.add(email)
        emails.append(email)
    return emails


# -----------------------------
# Report Generation
# -----------------------------

def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f} seconds"
    minutes = seconds / 60
    return f"{minutes:.1f} minutes"


def dedupe_and_humanize_changes(sources: Dict, indicator_names: Dict[str, str]) -> List[Dict]:
    """
    Dedupe upstream changes by HUMAN NAME (not dataset ID).
    Combines year ranges for same indicator across datasets.
    """
    seen = {}
    
    for source_name, source_data in sources.items():
        for change in source_data.get('changes', []):
            dataset = change.get('dataset', 'unknown')
            period_range = change.get('period_range', '')
            is_new = change.get('is_new', False)
            
            # Parse years from period range
            min_year = max_year = None
            if '-' in str(period_range):
                try:
                    parts = str(period_range).split('-')
                    min_year = int(parts[0])
                    max_year = int(parts[1])
                except:
                    pass
            
            # Get human name — this is the dedupe key
            human_name = indicator_names.get(dataset, dataset)
            
            # Dedupe by human name, keep widest year range
            if human_name not in seen:
                seen[human_name] = {
                    'dataset': dataset,
                    'name': human_name,
                    'min_year': min_year,
                    'max_year': max_year,
                    'is_new': is_new,
                }
            else:
                # Extend year range if wider
                if min_year and (seen[human_name]['min_year'] is None or min_year < seen[human_name]['min_year']):
                    seen[human_name]['min_year'] = min_year
                if max_year and (seen[human_name]['max_year'] is None or max_year > seen[human_name]['max_year']):
                    seen[human_name]['max_year'] = max_year
                # If any source is not new, mark as update
                if not is_new:
                    seen[human_name]['is_new'] = False
    
    return list(seen.values())


def build_indicator_matrix(sources: Dict, indicator_names: Dict[str, str]) -> Dict[str, Dict]:
    """
    Build indicator × geography matrix from vintage sources.
    Returns: {'Employment': {'UK': {'changed': True, 'max_year': 2024}, 'Regions': {...}, ...}, ...}
    """
    # Map source names to display geography
    source_to_geo = {
        'macro': 'UK',
        'itl1': 'Regions',
        'itl2': 'Regions',
        'itl3': 'Regions',
        'lad': 'Districts',
    }
    
    # Rename mappings (consolidate variants to single display name)
    rename_map = {
        'Working Age Population (16-64)': 'Population',
        'Population (16-64)': 'Population',
    }

    def normalize_name(human_name: str) -> str:
        """Normalize indicator naming so baseline + vintage overlays line up."""
        human_name = rename_map.get(human_name, human_name)
        return human_name
    
    matrix: Dict[str, Dict] = {}
    
    for source_name, source_data in sources.items():
        geo = source_to_geo.get(source_name, source_name)
        
        for change in source_data.get('changes', []):
            dataset = change.get('dataset', 'unknown')
            period_range = change.get('period_range', '')
            
            # Parse max year
            max_year = None
            if '-' in str(period_range):
                try:
                    max_year = int(str(period_range).split('-')[1])
                except:
                    pass
            
            # Get human name and apply renames
            human_name = indicator_names.get(dataset, dataset)
            human_name = normalize_name(human_name)
            
            # Initialize indicator row if needed
            if human_name not in matrix:
                matrix[human_name] = {}
            
            # Update geography cell (keep latest year)
            if geo not in matrix[human_name] or (max_year and max_year > matrix[human_name][geo].get('max_year', 0)):
                matrix[human_name][geo] = {
                    'changed': True,
                    'max_year': max_year,
                }
    
    return matrix


def get_latest_years_matrix(metric_display_names: Dict[str, str]) -> Dict[str, Dict[str, Dict]]:
    """
    Build a baseline matrix showing latest available year by core indicator and geography.
    This uses DuckDB silver history tables so the matrix is populated even when there were
    no upstream changes detected this run.

    Returns matrix in the same shape as build_indicator_matrix:
      {indicator_name: {geo: {"changed": False, "max_year": 2024}}}
    """
    if duckdb is None or not DUCK_PATH.exists():
        return {}

    # Core metrics we want to surface in the weekly email.
    # These align with the email's "Core Indicators" framing.
    core_metric_ids = [
        "nominal_gva_mn_gbp",
        "emp_total_jobs",
        "employment_rate_pct",
        "gdhi_total_mn_gbp",
        "population_total",
        "unemployment_rate_pct",
    ]

    # UK macro tables often use uk_* metric ids. Map those back onto the common ids used
    # across Regions/Districts so the matrix can show UK years too.
    uk_metric_map = {
        "nominal_gva_mn_gbp": "uk_nominal_gva_mn_gbp",
        "emp_total_jobs": "uk_emp_total_jobs",
        "employment_rate_pct": "uk_employment_rate_pct",
        "gdhi_total_mn_gbp": "uk_gdhi_total_mn_gbp",
        "population_total": "uk_population_total",
        "unemployment_rate_pct": "uk_unemployment_rate_pct",
    }

    # Fallback labels (used if metadata.indicators doesn't have a display_name yet)
    fallback_labels = {
        "nominal_gva_mn_gbp": "Economic Output (GVA)",
        "emp_total_jobs": "Employment",
        "employment_rate_pct": "Employment Rate",
        "gdhi_total_mn_gbp": "Household Income (GDHI)",
        "population_total": "Population",
        "unemployment_rate_pct": "Unemployment Rate",
    }

    def label_for(metric_id: str) -> str:
        return metric_display_names.get(metric_id) or fallback_labels.get(metric_id) or metric_id

    def max_year_by_metric(con: duckdb.DuckDBPyConnection, table: str, metric_ids: List[str]) -> Dict[str, int]:
        vals = ",".join([f"'{m}'" for m in metric_ids])
        rows = con.execute(f"""
            SELECT metric_id, MAX(CAST(period AS INT)) AS max_year
            FROM {table}
            WHERE metric_id IN ({vals})
              AND period IS NOT NULL
            GROUP BY 1
        """).fetchall()
        out: Dict[str, int] = {}
        for metric_id, max_year in rows:
            if metric_id and max_year is not None:
                out[str(metric_id)] = int(max_year)
        return out

    try:
        con = duckdb.connect(str(DUCK_PATH), read_only=True)

        # Baseline years from history tables
        # UK: query both common ids and uk_* ids, then map into the common id space.
        uk_query_ids = list(dict.fromkeys(core_metric_ids + [uk_metric_map[m] for m in core_metric_ids]))
        uk_raw = max_year_by_metric(con, "silver.uk_macro_history", uk_query_ids)
        uk: Dict[str, int] = {}
        for common_id in core_metric_ids:
            uk_year = uk_raw.get(common_id) or uk_raw.get(uk_metric_map.get(common_id, ""))
            if uk_year:
                uk[common_id] = int(uk_year)

        regions = max_year_by_metric(con, "silver.itl1_history", core_metric_ids)
        districts = max_year_by_metric(con, "silver.lad_history", core_metric_ids)
        con.close()

        matrix: Dict[str, Dict[str, Dict]] = {}
        for metric_id in core_metric_ids:
            human = label_for(metric_id)
            matrix[human] = {
                "UK": {"changed": False, "max_year": uk.get(metric_id)},
                "Regions": {"changed": False, "max_year": regions.get(metric_id)},
                "Districts": {"changed": False, "max_year": districts.get(metric_id)},
            }
        return matrix
    except Exception as e:
        log.warning(f"Failed to build baseline latest-years matrix: {e}")
        return {}


def generate_outlook_section() -> str:
    """
    Generate a forward-looking Outlook section (next year only) for UK (gold.uk_macro_forecast).

    - Additive metrics: YoY % growth
    - Rate metrics: pp change

    Returns HTML string (may be empty if prerequisites are missing).
    """
    if duckdb is None or not DUCK_PATH.exists():
        return ""

    # Core indicators shown in the email (keep aligned with matrix)
    metrics = [
        ("nominal_gva_mn_gbp", "Economic Output (GVA)", "pct"),
        ("emp_total_jobs", "Employment", "pct"),
        ("employment_rate_pct", "Employment Rate", "pp"),
        ("gdhi_total_mn_gbp", "Household Income (GDHI)", "pct"),
        ("population_total", "Population", "pct"),
        ("unemployment_rate_pct", "Unemployment Rate", "pp"),
    ]

    # UK macro commonly prefixes metric ids with uk_
    uk_alias = {
        "nominal_gva_mn_gbp": "uk_nominal_gva_mn_gbp",
        "emp_total_jobs": "uk_emp_total_jobs",
        "employment_rate_pct": "uk_employment_rate_pct",
        "gdhi_total_mn_gbp": "uk_gdhi_total_mn_gbp",
        "population_total": "uk_population_total",
        "unemployment_rate_pct": "uk_unemployment_rate_pct",
    }

    def detect_metric_col(con: "duckdb.DuckDBPyConnection", schema: str, table: str) -> str:
        row = con.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = ?
              AND table_name = ?
              AND column_name IN ('metric_id', 'metric')
            ORDER BY column_name
            LIMIT 1
            """,
            [schema, table],
        ).fetchone()
        return row[0] if row else "metric_id"

    try:
        con = duckdb.connect(str(DUCK_PATH), read_only=True)

        uk_metric_col = detect_metric_col(con, "gold", "uk_macro_forecast")

        # Determine forecast start year from UK macro (prefer last historical year if present)
        last_hist_row = con.execute(f"""
            SELECT MAX(CAST(period AS INT))
            FROM gold.uk_macro_forecast
            WHERE COALESCE(data_type, 'forecast') = 'historical'
        """).fetchone()
        last_hist = last_hist_row[0] if last_hist_row and last_hist_row[0] is not None else None
        if last_hist is None:
            # Fallback: use max(period)-3 as a "current" anchor
            max_row = con.execute("SELECT MAX(CAST(period AS INT)) FROM gold.uk_macro_forecast").fetchone()
            max_period = max_row[0] if max_row and max_row[0] is not None else None
            if max_period is None:
                con.close()
                return ""
            last_hist = int(max_period) - 3

        y1 = int(last_hist) + 1

        # Build UK outlook (YoY % or pp changes per metric)
        uk_ids = []
        for mid, _, _ in metrics:
            uk_ids.append(mid)
            uk_ids.append(uk_alias.get(mid, mid))
        uk_ids = list(dict.fromkeys(uk_ids))
        uk_vals = ",".join([f"'{m}'" for m in uk_ids])
        years_vals = ",".join([str(y) for y in [last_hist, y1]])

        con.execute(f"""
            CREATE OR REPLACE TEMP VIEW uk_series AS
            SELECT
                CASE
                    WHEN {uk_metric_col} LIKE 'uk_%' THEN SUBSTR({uk_metric_col}, 4)
                    ELSE {uk_metric_col}
                END AS metric_id_norm,
                CAST(period AS INT) AS period,
                CAST(value AS DOUBLE) AS value
            FROM gold.uk_macro_forecast
            WHERE CAST(period AS INT) IN ({years_vals})
              AND {uk_metric_col} IN ({uk_vals});
        """)

        con.execute("""
            CREATE OR REPLACE TEMP VIEW uk_outlook AS
            SELECT
                metric_id_norm AS metric_id,
                period,
                value,
                LAG(value) OVER (PARTITION BY metric_id_norm ORDER BY period) AS prev_value
            FROM uk_series;
        """)

        uk_rows = con.execute(f"""
            SELECT metric_id, period,
                   CASE
                     WHEN prev_value IS NULL THEN NULL
                     WHEN metric_id IN ('employment_rate_pct','unemployment_rate_pct')
                       THEN (value - prev_value)
                     ELSE (value / NULLIF(prev_value, 0) - 1.0) * 100.0
                   END AS delta
            FROM uk_outlook
            WHERE period = {y1}
            ORDER BY metric_id;
        """).fetchall()

        uk_delta: Dict[str, float | None] = {}
        for mid, _period, delta in uk_rows:
            uk_delta[str(mid)] = float(delta) if delta is not None else None

        con.close()

        def fmt(metric_id: str, delta: Optional[float]) -> str:
            if delta is None:
                return "—"
            if metric_id in ("employment_rate_pct", "unemployment_rate_pct"):
                return f"{delta:+.1f}pp"
            return f"{delta:+.1f}%"

        # Build compact HTML (no wide tables)
        html = f"""
        <h2>📈 Outlook (Next Year)</h2>
        <div style="margin: -4px 0 12px 0; font-size: 13px; color: #64748b; font-weight: 600;">
            UK {y1} vs {int(last_hist)}. YoY % for additive metrics; pp change for rates.
        </div>
        <div class="data-card">
        """

        for mid, label, _kind in metrics:
            html += f"""
            <div class="data-item">
                <span class="data-label">{label}</span>
                <span class="data-value">{fmt(mid, uk_delta.get(mid))}</span>
            </div>
            """

        html += "</div>"
        return html
    except Exception as e:
        log.warning(f"Outlook section unavailable: {e}")
        try:
            con.close()  # type: ignore[name-defined]
        except Exception:
            pass
        return ""


def generate_data_updates_table(matrix: Dict, geo_order: List[str] = None) -> str:
    """Generate HTML table for indicator × geography matrix."""
    if geo_order is None:
        geo_order = ['UK', 'Regions', 'Districts']
    
    if not matrix:
        return """
            <div style="color: #64748b; text-align: center; padding: 12px;">
                No upstream data available
            </div>
        """
    
    html = """
        <div style="margin-bottom: 12px; font-size: 12px; color: #64748b;">
            <span style="background: #dcfce7; color: #16a34a; padding: 3px 8px; border-radius: 4px; margin-right: 8px;">✓ 2024</span> Updated this run
            <span style="margin-left: 16px; background: #f1f5f9; color: #64748b; padding: 3px 8px; border-radius: 4px; margin-right: 8px;">2024</span> Latest available (unchanged)
        </div>
        <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
            <thead>
                <tr style="border-bottom: 2px solid #e2e8f0;">
                    <th style="text-align: left; padding: 12px 8px; color: #64748b; font-weight: 600;">Indicator</th>
    """
    
    for geo in geo_order:
        html += f'<th style="text-align: center; padding: 12px 8px; color: #64748b; font-weight: 600;">{geo}</th>'
    
    html += """
                </tr>
            </thead>
            <tbody>
    """
    
    for indicator in sorted(matrix.keys()):
        html += f"""
                <tr style="border-bottom: 1px solid #e2e8f0;">
                    <td style="padding: 12px 8px; color: #1a1a1a;">{indicator}</td>
        """
        
        for geo in geo_order:
            status = matrix[indicator].get(geo, {})
            year = status.get('max_year', '')
            if status.get('changed') and year:
                # Green - new data received
                html += f"""
                    <td style="text-align: center; padding: 12px 8px;">
                        <span style="background: #dcfce7; color: #16a34a; padding: 6px 12px; border-radius: 6px; font-weight: 600; font-size: 13px;">✓ {year}</span>
                    </td>
                """
            else:
                # Grey - unchanged (still show latest year if known)
                if year:
                    html += f"""
                        <td style="text-align: center; padding: 12px 8px;">
                            <span style="background: #f1f5f9; color: #64748b; padding: 6px 12px; border-radius: 6px; font-size: 13px;">{year}</span>
                        </td>
                    """
                else:
                    html += """
                        <td style="text-align: center; padding: 12px 8px;">
                            <span style="background: #f1f5f9; color: #64748b; padding: 6px 12px; border-radius: 6px; font-size: 13px;">—</span>
                        </td>
                    """
        
        html += "</tr>"
    
    html += """
            </tbody>
        </table>
    """
    
    return html


def matrix_has_any_updates(matrix: Dict, geo_order: Optional[List[str]] = None) -> bool:
    """Return True if any cell in the matrix is marked changed=True."""
    if not matrix:
        return False
    if geo_order is None:
        geo_order = ['UK', 'Regions', 'Districts']
    try:
        for _, geo_map in matrix.items():
            if not isinstance(geo_map, dict):
                continue
            for geo in geo_order:
                cell = geo_map.get(geo) or {}
                if isinstance(cell, dict) and cell.get("changed"):
                    return True
    except Exception:
        return False
    return False


def generate_report(
    run_id: str,
    sources: Dict,
    run_data: Dict,
    indicator_names: Dict,
    table_labels: Dict,
    geo_counts: Dict,
    indicator_coverage: Dict,
    matrix_override: Optional[Dict[str, Dict]] = None,
) -> Dict[str, str]:
    """Generate business-friendly email with dynamic data."""
    
    now = datetime.now(timezone.utc).strftime("%d %B %Y")
    
    # Pipeline status
    # NOTE: When this script is invoked as the final pipeline stage (email_report),
    # the orchestrator may not have written pipeline_summary.json yet. Be robust.
    pipeline = run_data.get('pipeline_summary') or {}
    pipeline_success = bool(pipeline.get('success', True))
    pipeline_duration = sum(s.get('duration_seconds', 0) for s in (pipeline.get('stages') or []))
    status_emoji = "✅" if pipeline_success else "❌"
    status_text = "All systems operational" if pipeline_success else "Pipeline encountered issues"
    
    # Build indicator × geography matrix
    matrix = matrix_override if matrix_override is not None else build_indicator_matrix(sources, indicator_names)
    
    # Table diff (humanized)
    # NOTE: vintage_diff.json may be missing for some run_ids (e.g. partial runs).
    diff = run_data.get('vintage_diff') or {}
    tables_modified = diff.get('tables_modified', 0)
    tables_unchanged = diff.get('tables_unchanged', 0)
    
    modified_tables = []
    for d in (diff.get('diffs') or []):
        if d.get('status') != 'UNCHANGED':
            table = d.get('table', 'unknown')
            human_name = table_labels.get(table, table)
            rows_delta = d.get('rows_delta', 0)
            period_ext = d.get('period_extended', False)
            max_period = d.get('max_period_after')
            
            detail = human_name
            if period_ext and max_period:
                detail += f" — extended to {max_period}"
            elif rows_delta != 0:
                detail += f" — {rows_delta:+d} observations"
            
            modified_tables.append(detail)
    
    # Data updates summary (changes vs steady-state)
    any_updates = matrix_has_any_updates(matrix)

    # Subject line
    if any_updates:
        subject = f"RegionIQ Weekly Update — New data received"
    elif modified_tables:
        subject = f"RegionIQ Weekly Update — Forecasts refreshed"
    else:
        subject = f"RegionIQ Weekly Update — No changes this week"
    
    # Calculate totals
    total_regions = sum(g['count'] for g in geo_counts.values())
    total_indicators = max(indicator_coverage.values()) if indicator_coverage else 0
    
    # Load logo
    logo_src = get_logo_base64()
    logo_html = f'<img src="{logo_src}" alt="RegionIQ" style="height: 64px;">' if logo_src else ""
    
    # Data Updates subtitle (legible at a glance)
    updates_banner = (
        '<div style="margin: -4px 0 12px 0; font-size: 16px; color: #16a34a; font-weight: 700;">'
        '✓ Changes this week'
        '</div>'
        if any_updates
        else
        '<div style="margin: -4px 0 12px 0; font-size: 16px; color: #64748b; font-weight: 700;">'
        'No changes this week'
        '</div>'
    )
    
    # HTML Body
    body = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ 
            font-family: 'Plus Jakarta Sans', -apple-system, sans-serif; 
            line-height: 1.6; 
            color: #1a1a1a; 
            max-width: 600px; 
            margin: 0 auto; 
            padding: 24px;
            background: #f8fafc;
        }}
        .container {{
            background: white;
            border-radius: 12px;
            padding: 32px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .header-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}
        h1 {{ 
            color: #0f172a; 
            font-size: 24px;
            font-weight: 700;
            margin: 0;
        }}
        .subtitle {{
            color: #64748b;
            font-size: 14px;
            margin-bottom: 24px;
        }}
        .status-bar {{
            background: {'#f0fdf4' if pipeline_success else '#fef2f2'};
            border-left: 4px solid {'#16a34a' if pipeline_success else '#dc2626'};
            padding: 12px 16px;
            border-radius: 0 8px 8px 0;
            margin-bottom: 24px;
        }}
        .status-text {{
            color: {'#166534' if pipeline_success else '#991b1b'};
            font-weight: 600;
        }}
        h2 {{ 
            color: #334155; 
            font-size: 16px;
            font-weight: 600;
            margin: 24px 0 12px 0;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .data-card {{
            background: #f8fafc;
            padding: 16px;
            border-radius: 8px;
            margin: 8px 0;
        }}
        .data-item {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #e2e8f0;
        }}
        .data-item:last-child {{
            border-bottom: none;
        }}
        .data-label {{
            color: #475569;
        }}
        .data-value {{
            color: #0f172a;
            font-weight: 600;
        }}
        .highlight {{
            color: #2563eb;
        }}
        .tag {{
            display: inline-block;
            background: #dbeafe;
            color: #1e40af;
            font-size: 12px;
            padding: 2px 8px;
            border-radius: 4px;
            margin-left: 8px;
        }}
        .tag-new {{
            background: #dcfce7;
            color: #166534;
        }}
        .coverage {{
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
        }}
        .coverage-item {{
            background: #f1f5f9;
            padding: 8px 16px;
            border-radius: 6px;
            text-align: center;
            flex: 1;
            min-width: 80px;
        }}
        .coverage-count {{
            font-size: 20px;
            font-weight: 700;
            color: #0f172a;
        }}
        .coverage-label {{
            font-size: 11px;
            color: #64748b;
            text-transform: uppercase;
        }}
        .summary-row {{
            display: flex;
            justify-content: space-between;
            padding: 12px 0;
            border-bottom: 1px solid #e2e8f0;
        }}
        .summary-row:last-child {{
            border-bottom: none;
        }}
        .footer {{
            margin-top: 32px;
            padding-top: 16px;
            border-top: 1px solid #e2e8f0;
            color: #94a3b8;
            font-size: 12px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header-row">
            <h1>Weekly Pipeline Report</h1>
            {logo_html}
        </div>
        <div class="subtitle" style="margin-top: -4px; margin-bottom: 4px; font-weight: 500; color: #334155;">Core Indicators</div>
        <div class="subtitle">{now}</div>
        
        <div class="status-bar">
            <span class="status-text">{status_emoji} {status_text}</span>
            <span style="color: #64748b; float: right;">Completed in {format_duration(pipeline_duration)}</span>
        </div>
        
        <h2>📊 Data Updates</h2>
        {updates_banner}
        <div class="data-card">
"""
    
    body += generate_data_updates_table(matrix)
    body += "</div>"
    
    # Outlook section (forward-looking deltas)
    outlook_html = generate_outlook_section()
    if outlook_html:
        body += outlook_html

    body += """
        
        <h2>🔄 Forecast Changes</h2>
        <div class="data-card">
"""
    
    if modified_tables:
        for table in modified_tables:
            body += f"""
            <div class="data-item">
                <span class="data-label">{table}</span>
                <span class="data-value highlight">Changed</span>
            </div>
"""
    else:
        body += """
            <div style="color: #64748b; text-align: center; padding: 12px;">
                Forecasts recomputed — outputs unchanged
            </div>
"""
    
    body += """
        </div>
        
        <h2>🌍 Coverage</h2>
        <div class="coverage">
"""
    
    # Dynamic geography coverage
    for level in ['UK', 'ITL1', 'ITL2', 'ITL3', 'LAD']:
        if level in geo_counts:
            count = geo_counts[level]['count']
            label = geo_counts[level]['label']
            body += f"""
            <div class="coverage-item">
                <div class="coverage-count">{count:,}</div>
                <div class="coverage-label">{label}</div>
            </div>
"""
    
    body += """
        </div>
        
        <h2>📈 Summary</h2>
        <div class="data-card">
            <div class="summary-row">
                <span class="data-label">Total Geographies</span>
                <span class="data-value">""" + f"{total_regions:,}" + """</span>
            </div>
            <div class="summary-row">
                <span class="data-label">Indicators per Region</span>
                <span class="data-value">""" + f"{total_indicators}" + """</span>
            </div>
            <div class="summary-row">
                <span class="data-label">Data Points</span>
                <span class="data-value">""" + f"{total_regions * total_indicators:,}+" + """</span>
            </div>
        </div>
"""
    
    body += f"""
        <div class="footer">
            <p><strong>RegionIQ</strong> — Economic data. Live. Programmable. Instant.</p>
            <p>Run ID: {run_id}</p>
        </div>
    </div>
</body>
</html>
"""
    
    return {
        'subject': subject,
        'body': body
    }


# -----------------------------
# Email Sending
# -----------------------------

def send_email(subject: str, body: str, recipients: List[str], dry_run: bool = False) -> bool:
    if dry_run:
        log.info("DRY RUN — would send email:")
        log.info(f"  To: {', '.join(recipients) if recipients else '(none)'}")
        log.info(f"  Subject: {subject}")
        log.info(f"  Body length: {len(body)} chars")
        
        # Save HTML for preview
        preview_path = Path("data/logs/email_preview.html")
        preview_path.write_text(body)
        log.info(f"  Preview saved: {preview_path}")
        return True

    if requests is None:
        log.error("requests not installed. Install it or use --dry-run.")
        return False

    api_key = os.environ.get('RESEND_API_KEY')
    if not api_key:
        log.error("RESEND_API_KEY not found in environment")
        return False

    if not recipients:
        log.warning("No recipients resolved; skipping email send")
        return True
    
    payload = {
        "from": FROM_EMAIL,
        "to": recipients,
        "subject": subject,
        "html": body
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(RESEND_API_URL, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            log.info(f"✅ Email sent to {len(recipients)} recipient(s)")
            return True
        else:
            log.error(f"Resend API error: {response.status_code} — {response.text}")
            return False
            
    except Exception as e:
        log.error(f"Failed to send email: {e}")
        return False


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    log.info("=" * 60)
    log.info("REGIONIQ WEEKLY EMAIL REPORT (v3 - Dynamic)")
    log.info("=" * 60)
    log.info(f"Run ID: {args.run_id}")
    try:
        if "_DOTENV_LOADED_FROM" in globals() and globals().get("_DOTENV_LOADED_FROM"):
            log.info(f"Dotenv loaded from: {globals().get('_DOTENV_LOADED_FROM')}")
        else:
            log.info("Dotenv not loaded (relying on process environment)")
    except Exception:
        pass
    
    # Load dynamic metadata from DuckDB
    log.info("\nLoading metadata from DuckDB...")
    indicator_names = get_indicator_names()
    metric_display_names = get_metric_display_names()
    table_labels = get_table_labels()
    geo_counts = get_geography_counts()
    indicator_coverage = get_indicator_coverage()
    
    # Load vintage data
    log.info("\nLoading vintage data...")
    sources = load_vintage_sources()
    log.info(f"  Loaded {len(sources)} upstream source files")
    
    run_data = load_run_data(args.run_id)
    loaded = sum(1 for v in run_data.values() if v is not None)
    log.info(f"  Loaded {loaded}/3 run data files")

    pipeline = run_data.get("pipeline_summary") or {}
    pipeline_success = bool(pipeline.get("success", False))
    
    # Build baseline latest-years matrix (so matrix still shows years even if no upstream changes)
    baseline_matrix = get_latest_years_matrix(metric_display_names)

    # Overlay upstream changes onto baseline matrix (green ticks only where changed)
    upstream_matrix = build_indicator_matrix(sources, indicator_names)
    for indicator_name, geo_map in upstream_matrix.items():
        if indicator_name not in baseline_matrix:
            baseline_matrix[indicator_name] = {}
        for geo, st in geo_map.items():
            existing = baseline_matrix[indicator_name].get(geo, {"changed": False, "max_year": None})
            # Mark changed, and update max_year if present
            existing["changed"] = True
            if st.get("max_year"):
                existing["max_year"] = st.get("max_year")
            baseline_matrix[indicator_name][geo] = existing

    # Generate report
    log.info("\nGenerating report...")
    report = generate_report(
        args.run_id, sources, run_data,
        indicator_names, table_labels, geo_counts, indicator_coverage,
        matrix_override=baseline_matrix
    )
    log.info(f"  Subject: {report['subject']}")

    # Resolve recipients from Supabase
    recipients: List[str] = []
    try:
        recipients = fetch_recipients_from_supabase(pipeline_success=pipeline_success)
        log.info(f"  Resolved {len(recipients)} recipient(s) from Supabase")
    except Exception as e:
        if _is_lenient_mode():
            log.warning(f"Recipient resolution failed (allowed in {PIPELINE_MODE}): {e}")
        else:
            log.error(f"Recipient resolution failed: {e}")
            sys.exit(1)
    
    # Send email
    log.info("\nSending email...")
    success = send_email(report['subject'], report['body'], recipients=recipients, dry_run=args.dry_run)
    
    if success:
        log.info("\n✅ Report complete")
    else:
        log.error("\n❌ Report failed")
        sys.exit(1)


if __name__ == "__main__":
    main()