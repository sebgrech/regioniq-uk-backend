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
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests")
    sys.exit(1)

try:
    import duckdb
except ImportError:
    print("ERROR: duckdb not installed. Run: pip install duckdb")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
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
TO_EMAIL = "slrgrech@hotmail.com"

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
    logo_path = Path("assets/logo.png")
    if not logo_path.exists():
        log.warning(f"Logo not found at {logo_path}")
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
    if not DUCK_PATH.exists():
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
    if not DUCK_PATH.exists():
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
    if not DUCK_PATH.exists():
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
    }
    
    matrix = {}
    
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
            human_name = rename_map.get(human_name, human_name)
            
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


def generate_data_updates_table(matrix: Dict, geo_order: List[str] = None) -> str:
    """Generate HTML table for indicator × geography matrix."""
    if geo_order is None:
        geo_order = ['UK', 'Regions', 'Districts']
    
    if not matrix:
        return """
            <div style="color: #64748b; text-align: center; padding: 12px;">
                No upstream data changes detected
            </div>
        """
    
    html = """
        <div style="margin-bottom: 12px; font-size: 12px; color: #64748b;">
            <span style="background: #dcfce7; color: #16a34a; padding: 3px 8px; border-radius: 4px; margin-right: 8px;">✓ 2024</span> Updated
            <span style="margin-left: 16px; background: #f1f5f9; color: #64748b; padding: 3px 8px; border-radius: 4px; margin-right: 8px;">No change</span> Not updated this run
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
            if status.get('changed'):
                year = status.get('max_year', '')
                # Green - new data received
                html += f"""
                    <td style="text-align: center; padding: 12px 8px;">
                        <span style="background: #dcfce7; color: #16a34a; padding: 6px 12px; border-radius: 6px; font-weight: 600; font-size: 13px;">✓ {year}</span>
                    </td>
                """
            else:
                # Grey - no change
                html += """
                    <td style="text-align: center; padding: 12px 8px;">
                        <span style="background: #f1f5f9; color: #64748b; padding: 6px 12px; border-radius: 6px; font-size: 13px;">No change</span>
                    </td>
                """
        
        html += "</tr>"
    
    html += """
            </tbody>
        </table>
    """
    
    return html


def generate_report(run_id: str, sources: Dict, run_data: Dict, 
                    indicator_names: Dict, table_labels: Dict, 
                    geo_counts: Dict, indicator_coverage: Dict) -> Dict[str, str]:
    """Generate business-friendly email with dynamic data."""
    
    now = datetime.now(timezone.utc).strftime("%d %B %Y")
    
    # Pipeline status
    pipeline = run_data.get('pipeline_summary', {})
    pipeline_success = pipeline.get('success', False)
    pipeline_duration = sum(s.get('duration_seconds', 0) for s in pipeline.get('stages', []))
    status_emoji = "✅" if pipeline_success else "❌"
    status_text = "All systems operational" if pipeline_success else "Pipeline encountered issues"
    
    # Build indicator × geography matrix
    matrix = build_indicator_matrix(sources, indicator_names)
    
    # Table diff (humanized)
    diff = run_data.get('vintage_diff', {})
    tables_modified = diff.get('tables_modified', 0)
    tables_unchanged = diff.get('tables_unchanged', 0)
    
    modified_tables = []
    for d in diff.get('diffs', []):
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
    
    # Subject line
    if matrix:
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
        <div class="data-card">
"""
    
    body += generate_data_updates_table(matrix)
    body += "</div>"
    
    body += """
        
        <h2>🔄 Forecast Changes</h2>
        <div class="data-card">
"""
    
    if modified_tables:
        for table in modified_tables:
            body += f"""
            <div class="data-item">
                <span class="data-label">{table}</span>
                <span class="data-value highlight">Refreshed</span>
            </div>
"""
    else:
        body += """
            <div style="color: #64748b; text-align: center; padding: 12px;">
                All forecasts unchanged
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

def send_email(subject: str, body: str, dry_run: bool = False) -> bool:
    api_key = os.environ.get('RESEND_API_KEY')
    
    if not api_key:
        log.error("RESEND_API_KEY not found in environment")
        return False
    
    if dry_run:
        log.info("DRY RUN — would send email:")
        log.info(f"  To: {TO_EMAIL}")
        log.info(f"  Subject: {subject}")
        log.info(f"  Body length: {len(body)} chars")
        
        # Save HTML for preview
        preview_path = Path("data/logs/email_preview.html")
        preview_path.write_text(body)
        log.info(f"  Preview saved: {preview_path}")
        return True
    
    payload = {
        "from": FROM_EMAIL,
        "to": [TO_EMAIL],
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
            log.info(f"✅ Email sent to {TO_EMAIL}")
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
    
    # Load dynamic metadata from DuckDB
    log.info("\nLoading metadata from DuckDB...")
    indicator_names = get_indicator_names()
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
    
    # Generate report
    log.info("\nGenerating report...")
    report = generate_report(
        args.run_id, sources, run_data,
        indicator_names, table_labels, geo_counts, indicator_coverage
    )
    log.info(f"  Subject: {report['subject']}")
    
    # Send email
    log.info("\nSending email...")
    success = send_email(report['subject'], report['body'], dry_run=args.dry_run)
    
    if success:
        log.info("\n✅ Report complete")
    else:
        log.error("\n❌ Report failed")
        sys.exit(1)


if __name__ == "__main__":
    main()