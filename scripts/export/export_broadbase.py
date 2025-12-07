#!/usr/bin/env python3
"""
RegionIQ → Supabase SYNC V5.0
=============================
Syncs full forecast cascade (Macro + ITL1/2/3 + LAD) to Supabase.

V5 Changes:
- Added LAD (Local Authority District) support (361 regions, ~135k rows)
- New parent_code column for hierarchical relationships (LAD → ITL3)
- LAD reads from gold.lad_forecast (different naming convention)
- Simplified LAD export (data_type already correct from forecast scripts)

V4 Changes:
- Aligned with V4 forecast architecture (forecast + reconcile cascade)
- employment_rate_pct and unemployment_rate_pct are PRIMARY metrics (from NOMIS APS)
- Only productivity, gdhi_per_head, income_per_worker are DERIVED (Monte Carlo)
- Proper deduplication: historical takes priority over forecast for same period
- Trust data_type from forecast scripts (no global override)

Usage:
    python3 scripts/sync/supabase_sync_v5.py

Requires:
    SUPABASE_URI in .env file
"""

from dotenv import load_dotenv
load_dotenv()

import os
import sys
import duckdb
import psycopg2
from psycopg2 import sql
from pathlib import Path
from datetime import datetime
import logging

# ==============================================================================
# Configuration
# ==============================================================================

SUPABASE_URI = os.getenv("SUPABASE_URI", "")

if not SUPABASE_URI:
    print("ERROR: SUPABASE_URI not set in .env")
    sys.exit(1)

DB_PATH = Path("data/lake/warehouse.duckdb")
EXPORT_DIR = Path("data/export")
LOG_DIR = Path("logs")

EXPORT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# Logging
# ==============================================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = LOG_DIR / f"supabase_sync_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)

if not DB_PATH.exists():
    logger.error(f"DuckDB not found: {DB_PATH}")
    sys.exit(1)

logger.info("=" * 70)
logger.info("REGIONIQ → SUPABASE SYNC V5.0")
logger.info("=" * 70)
logger.info(f"DuckDB: {DB_PATH}")
logger.info(f"Log file: {log_path}")

# ==============================================================================
# Schema Definition
# ==============================================================================

DDL = """
region_code TEXT,
region_name TEXT,
region_level TEXT,
parent_code TEXT,
metric_id TEXT,
period INT,
value DOUBLE PRECISION,
unit TEXT,
freq TEXT,
data_type TEXT,
ci_lower DOUBLE PRECISION,
ci_upper DOUBLE PRECISION,
vintage TIMESTAMP NULL,
forecast_run_date DATE NULL,
forecast_version TEXT,
is_calculated BOOLEAN NOT NULL
"""

COLUMNS = [
    "region_code", "region_name", "region_level", "parent_code", "metric_id",
    "period", "value", "unit", "freq", "data_type",
    "ci_lower", "ci_upper", "vintage", "forecast_run_date",
    "forecast_version", "is_calculated"
]

# Derived-only metrics (calculated via Monte Carlo from reconciled components)
# NOTE: employment_rate_pct and unemployment_rate_pct are PRIMARY metrics
#       (forecast directly from NOMIS APS data, not derived)
DERIVED_METRICS = [
    'productivity_gbp_per_job',
    'gdhi_per_head_gbp',
    'income_per_worker_gbp'
]

# ==============================================================================
# Export Functions
# ==============================================================================

def export_macro(con) -> Path:
    """Export macro-level forecasts"""
    logger.info("\n[1/5] Exporting macro_latest_all...")
    
    csv_path = EXPORT_DIR / "macro_latest_all.csv"
    
    con.execute("""
    CREATE OR REPLACE TEMP VIEW macro_export AS
    SELECT
        region_code,
        region_name,
        region_level,
        NULL::TEXT AS parent_code,
        metric_id,
        CAST(period AS INT) AS period,
        CAST(value AS DOUBLE) AS value,
        unit,
        freq,
        data_type,
        CAST(ci_lower AS DOUBLE) AS ci_lower,
        CAST(ci_upper AS DOUBLE) AS ci_upper,
        NULL::TIMESTAMP AS vintage,
        CAST(forecast_run_date AS DATE) AS forecast_run_date,
        forecast_version,
        FALSE AS is_calculated
    FROM gold.uk_macro_forecast
    WHERE period IS NOT NULL 
      AND metric_id IS NOT NULL;
    """)
    
    con.execute(f"COPY macro_export TO '{csv_path}' (HEADER, DELIMITER ',');")
    
    row_count = con.execute("SELECT COUNT(*) FROM macro_export").fetchone()[0]
    logger.info(f"  ✓ Exported {row_count:,} rows → {csv_path}")
    
    return csv_path


def export_regional(con, level: str) -> Path:
    """Export regional forecasts (ITL1/ITL2/ITL3)"""
    logger.info(f"  • {level.upper()}...")
    
    # Build derived metrics SQL list
    calc_vals = ",".join([f"('{m}')" for m in DERIVED_METRICS])
    
    # Create dimension views
    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW dim_region_{level} AS
    SELECT 
        region_code, 
        any_value(region_name) AS region_name, 
        any_value(region_level) AS region_level
    FROM silver.{level}_history 
    GROUP BY 1;
    """)

    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW dim_metric_{level} AS
    SELECT 
        metric_id, 
        any_value(unit) AS unit, 
        any_value(freq) AS freq
    FROM silver.{level}_history 
    GROUP BY 1;
    """)

    # Derived metrics lookup
    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW calc_metrics_{level} AS
    SELECT * FROM (VALUES {calc_vals}) AS t(metric_id);
    """)

    # Last historical year per region/metric (for dedup filtering)
    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW last_hist_{level} AS
    SELECT 
        region_code, 
        metric_id, 
        MAX(CAST(period AS INT)) AS last_hist_year
    FROM silver.{level}_history 
    GROUP BY 1, 2;
    """)

    # Silver (historical data)
    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW silver_norm_{level} AS
    SELECT
        s.region_code,
        s.region_name,
        s.region_level,
        NULL::TEXT AS parent_code,
        s.metric_id,
        CAST(s.period AS INT) AS period,
        CAST(s.value AS DOUBLE) AS value,
        s.unit,
        s.freq,
        'historical' AS data_type,
        NULL::DOUBLE AS ci_lower,
        NULL::DOUBLE AS ci_upper,
        CAST(s.vintage AS TIMESTAMP) AS vintage,
        NULL::DATE AS forecast_run_date,
        NULL::VARCHAR AS forecast_version,
        (s.metric_id IN (SELECT metric_id FROM calc_metrics_{level})) AS is_calculated
    FROM silver.{level}_history s 
    WHERE s.period IS NOT NULL;
    """)

    # Gold (forecast data) - trust data_type from forecast scripts
    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW gold_filtered_{level} AS
    SELECT
        gl.region_code,
        COALESCE(gl.region_name, r.region_name) AS region_name,
        COALESCE(gl.region_level, r.region_level) AS region_level,
        NULL::TEXT AS parent_code,
        gl.metric_id,
        CAST(gl.period AS INT) AS period,
        CAST(gl.value AS DOUBLE) AS value,
        COALESCE(gl.unit, m.unit) AS unit,
        COALESCE(gl.freq, m.freq) AS freq,
        COALESCE(gl.data_type, 'forecast') AS data_type,
        CAST(gl.ci_lower AS DOUBLE) AS ci_lower,
        CAST(gl.ci_upper AS DOUBLE) AS ci_upper,
        NULL::TIMESTAMP AS vintage,
        CAST(gl.forecast_run_date AS DATE) AS forecast_run_date,
        gl.forecast_version,
        (gl.metric_id IN (SELECT metric_id FROM calc_metrics_{level})) AS is_calculated
    FROM gold.{level}_latest gl
    LEFT JOIN dim_region_{level} r USING (region_code)
    LEFT JOIN dim_metric_{level} m USING (metric_id)
    LEFT JOIN last_hist_{level} lh USING (region_code, metric_id)
    WHERE gl.period IS NOT NULL 
      AND gl.metric_id IS NOT NULL
      AND (
          -- Non-derived metrics: only include forecast periods (after last historical)
          (gl.metric_id NOT IN (SELECT metric_id FROM calc_metrics_{level})
           AND (lh.last_hist_year IS NULL OR CAST(gl.period AS INT) > lh.last_hist_year))
          -- Derived metrics: include all periods (will be deduped below)
          OR (gl.metric_id IN (SELECT metric_id FROM calc_metrics_{level}))
      );
    """)

    # Union + deduplicate (historical takes priority for same region/metric/period)
    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW {level}_export AS
    WITH ranked AS (
        SELECT 
            u.*,
            ROW_NUMBER() OVER (
                PARTITION BY region_code, metric_id, period
                ORDER BY CASE WHEN u.data_type = 'historical' THEN 1 ELSE 2 END
            ) AS rn
        FROM (
            SELECT * FROM silver_norm_{level}
            UNION ALL
            SELECT * FROM gold_filtered_{level}
        ) u
    )
    SELECT
        region_code, region_name, region_level, parent_code, metric_id,
        period, value, unit, freq, data_type,
        ci_lower, ci_upper, vintage, forecast_run_date,
        forecast_version, is_calculated
    FROM ranked 
    WHERE rn = 1
    ORDER BY region_code, metric_id, period;
    """)

    csv_path = EXPORT_DIR / f"{level}_latest_all.csv"
    con.execute(f"COPY {level}_export TO '{csv_path}' (HEADER, DELIMITER ',');")
    
    row_count = con.execute(f"SELECT COUNT(*) FROM {level}_export").fetchone()[0]
    logger.info(f"    ✓ Exported {row_count:,} rows → {csv_path}")
    
    return csv_path


def export_lad(con) -> Path:
    """Export LAD (Local Authority District) forecasts
    
    LAD uses gold.lad_forecast (different naming from ITL levels).
    data_type is already correctly set by the forecast script (V1.3+).
    Maps itl3_code → parent_code for hierarchical reference.
    """
    logger.info("\n[3/5] Exporting lad_latest_all...")
    
    csv_path = EXPORT_DIR / "lad_latest_all.csv"
    
    # Build derived metrics SQL list
    calc_vals = ",".join([f"('{m}')" for m in DERIVED_METRICS])
    
    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW calc_metrics_lad AS
    SELECT * FROM (VALUES {calc_vals}) AS t(metric_id);
    """)
    
    # LAD export - simpler since data_type is already correct
    # Deduplicate in case of any overlap (historical takes priority)
    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW lad_export AS
    WITH ranked AS (
        SELECT 
            gl.region_code,
            gl.region_name,
            gl.region_level,
            gl.itl3_code AS parent_code,
            gl.metric_id,
            CAST(gl.period AS INT) AS period,
            CAST(gl.value AS DOUBLE) AS value,
            gl.unit,
            gl.freq,
            COALESCE(gl.data_type, 'forecast') AS data_type,
            CAST(gl.ci_lower AS DOUBLE) AS ci_lower,
            CAST(gl.ci_upper AS DOUBLE) AS ci_upper,
            NULL::TIMESTAMP AS vintage,
            CAST(gl.forecast_run_date AS DATE) AS forecast_run_date,
            gl.forecast_version,
            (gl.metric_id IN (SELECT metric_id FROM calc_metrics_lad)) AS is_calculated,
            ROW_NUMBER() OVER (
                PARTITION BY gl.region_code, gl.metric_id, gl.period
                ORDER BY CASE WHEN gl.data_type = 'historical' THEN 1 ELSE 2 END
            ) AS rn
        FROM gold.lad_forecast gl
        WHERE gl.period IS NOT NULL 
          AND gl.metric_id IS NOT NULL
    )
    SELECT
        region_code, region_name, region_level, parent_code, metric_id,
        period, value, unit, freq, data_type,
        ci_lower, ci_upper, vintage, forecast_run_date,
        forecast_version, is_calculated
    FROM ranked 
    WHERE rn = 1
    ORDER BY region_code, metric_id, period;
    """)
    
    con.execute(f"COPY lad_export TO '{csv_path}' (HEADER, DELIMITER ',');")
    
    row_count = con.execute("SELECT COUNT(*) FROM lad_export").fetchone()[0]
    region_count = con.execute("SELECT COUNT(DISTINCT region_code) FROM lad_export").fetchone()[0]
    metric_count = con.execute("SELECT COUNT(DISTINCT metric_id) FROM lad_export").fetchone()[0]
    
    logger.info(f"  ✓ Exported {row_count:,} rows → {csv_path}")
    logger.info(f"    ({region_count} LADs × {metric_count} metrics)")
    
    return csv_path


# ==============================================================================
# Supabase Sync
# ==============================================================================

def sync_table(conn, table: str, csv_path: Path):
    """Sync a CSV file to a Supabase table (TRUNCATE + COPY)"""
    logger.info(f"  • {table}...")
    
    cols = sql.SQL(",").join(map(sql.Identifier, COLUMNS))

    with conn.cursor() as cur:
        # Create table if not exists
        cur.execute(
            sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})")
            .format(sql.Identifier(table), sql.SQL(DDL))
        )
        
        # V5 migration: add parent_code column if missing
        cur.execute(sql.SQL("""
            ALTER TABLE {} ADD COLUMN IF NOT EXISTS parent_code TEXT;
        """).format(sql.Identifier(table)))

        # Truncate existing data
        cur.execute(sql.SQL("TRUNCATE {};").format(sql.Identifier(table)))

        # COPY from CSV
        copy_sql = sql.SQL(
            "COPY {} ({}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE)"
        ).format(sql.Identifier(table), cols)

        with open(csv_path, "r", encoding="utf-8") as f:
            cur.copy_expert(copy_sql.as_string(conn), f)

        # Log row count
        cur.execute(sql.SQL("SELECT COUNT(*) FROM {};").format(sql.Identifier(table)))
        row_count = cur.fetchone()[0]
        logger.info(f"    ✓ {table}: {row_count:,} rows")


# ==============================================================================
# Main
# ==============================================================================

def main():
    """Run full sync pipeline"""
    
    # Connect to DuckDB
    con = duckdb.connect(str(DB_PATH))
    
    # Export all levels
    csv_files = {}
    
    csv_files['macro'] = export_macro(con)
    
    logger.info("\n[2/5] Exporting ITL1/ITL2/ITL3...")
    for level in ['itl1', 'itl2', 'itl3']:
        csv_files[level] = export_regional(con, level)
    
    csv_files['lad'] = export_lad(con)
    
    con.close()
    
    # Sync to Supabase
    logger.info("\n[4/5] Syncing to Supabase...")
    
    with psycopg2.connect(SUPABASE_URI) as pg:
        sync_table(pg, "macro_latest_all", csv_files['macro'])
        sync_table(pg, "itl1_latest_all", csv_files['itl1'])
        sync_table(pg, "itl2_latest_all", csv_files['itl2'])
        sync_table(pg, "itl3_latest_all", csv_files['itl3'])
        sync_table(pg, "lad_latest_all", csv_files['lad'])
    
    # Summary
    logger.info("\n[5/5] ✅ SYNC COMPLETE")
    logger.info("=" * 70)
    logger.info("Tables synced:")
    logger.info("  • macro_latest_all")
    logger.info("  • itl1_latest_all")
    logger.info("  • itl2_latest_all")
    logger.info("  • itl3_latest_all")
    logger.info("  • lad_latest_all")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()