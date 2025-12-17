#!/usr/bin/env python3
"""
RegionIQ → Supabase SYNC V5.1
=============================
Syncs full forecast cascade (Macro + ITL1/2/3 + LAD) to Supabase.

V5.1 Changes:
- Added LAD support (same pattern as ITL1/2/3)

V5 Changes:
- Aligns with new gold schema:
    - Base metrics in gold.<level>_forecast
    - Monte Carlo ratios in gold.<level>_derived
- Only two derived metrics:
    - productivity_gbp_per_job
    - income_per_worker_gbp
- Macro export now unions uk_macro_forecast + uk_macro_derived
- Regional export now unions:
    - silver.<level>_history (historical)
    - gold.<level>_forecast (base metrics)
    - gold.<level>_derived (Monte Carlo ratios)
  with proper deduplication (historical > forecast)

Outputs:
    - macro_latest_all (CSV + Supabase table)
    - itl1_latest_all
    - itl2_latest_all
    - itl3_latest_all
    - lad_latest_all

Usage:
    python3 scripts/sync/supabase_sync_v5.py

Requires:
    SUPABASE_URI in .env file
"""

from dotenv import load_dotenv

# In some sandboxed / CI environments, reading a repo-root .env may be blocked.
# Treat that as "no .env", and rely on already-provided process env vars instead.
try:
    load_dotenv()
except PermissionError:
    pass

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

SKIP_SUPABASE_SYNC = not bool(SUPABASE_URI)
if SKIP_SUPABASE_SYNC:
    # This script is used both locally and in sandboxed/CI environments.
    # Missing SUPABASE_URI should not block CSV export (frontend can consume CSVs),
    # so we skip the sync step when credentials aren't provided.
    print("WARNING: SUPABASE_URI not set. Supabase sync will be skipped (CSV export will still run).")

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
logger.info("REGIONIQ → SUPABASE SYNC V5.1")
logger.info("=" * 70)
logger.info(f"DuckDB: {DB_PATH}")
logger.info(f"Log file: {log_path}")

# ==============================================================================
# Schema Definition (unchanged)
# ==============================================================================

DDL = """
region_code TEXT,
region_name TEXT,
region_level TEXT,
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
is_calculated BOOLEAN NOT NULL,
data_quality TEXT NULL
"""

COLUMNS = [
    "region_code", "region_name", "region_level", "metric_id",
    "period", "value", "unit", "freq", "data_type",
    "ci_lower", "ci_upper", "vintage", "forecast_run_date",
    "forecast_version", "is_calculated", "data_quality"
]

# Only Monte Carlo ratios are treated as "derived"
DERIVED_METRICS = [
    "productivity_gbp_per_job",
    "income_per_worker_gbp",
]

# ==============================================================================
# Export: MACRO
# ==============================================================================

def export_macro(con) -> Path:
    """
    Export macro-level data (UK aggregate) to CSV.

    Pulls from:
      - gold.uk_macro_forecast  (base metrics, including gdhi_per_head_gbp)
      - gold.uk_macro_derived   (Monte Carlo ratios)

    No silver macro usage here (consistent with previous behaviour).
    """

    logger.info("\n[1/5] Exporting macro_latest_all...")

    csv_path = EXPORT_DIR / "macro_latest_all.csv"

    # Detect correct metric column name (metric vs metric_id) for robustness
    metric_col_row = con.execute("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'gold'
          AND table_name = 'uk_macro_forecast'
          AND column_name IN ('metric_id', 'metric')
        ORDER BY column_name
        LIMIT 1
    """).fetchone()

    if not metric_col_row:
        logger.error("Could not detect metric column for gold.uk_macro_forecast")
        sys.exit(1)

    macro_metric_col = metric_col_row[0]

    # Base macro: from gold.uk_macro_forecast (all metrics in that table)
    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW macro_base AS
    SELECT
        region_code,
        region_name,
        region_level,
        {macro_metric_col} AS metric_id,
        CAST(period AS INT) AS period,
        CAST(value AS DOUBLE) AS value,
        unit,
        freq,
        COALESCE(data_type, 'forecast') AS data_type,
        CAST(ci_lower AS DOUBLE) AS ci_lower,
        CAST(ci_upper AS DOUBLE) AS ci_upper,
        NULL::TIMESTAMP AS vintage,
        CAST(forecast_run_date AS DATE) AS forecast_run_date,
        forecast_version,
        FALSE AS is_calculated,
        NULL::TEXT AS data_quality
    FROM gold.uk_macro_forecast
    WHERE period IS NOT NULL
      AND {macro_metric_col} IS NOT NULL;
    """)

    # Derived macro metrics: from gold.uk_macro_derived, always is_calculated = TRUE
    # Use the same metric column name detection
    metric_col_row_derived = con.execute("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'gold'
          AND table_name = 'uk_macro_derived'
          AND column_name IN ('metric_id', 'metric')
        ORDER BY column_name
        LIMIT 1
    """).fetchone()

    if not metric_col_row_derived:
        logger.error("Could not detect metric column for gold.uk_macro_derived")
        sys.exit(1)

    macro_derived_metric_col = metric_col_row_derived[0]

    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW macro_derived AS
    SELECT
        region_code,
        region_name,
        region_level,
        {macro_derived_metric_col} AS metric_id,
        CAST(period AS INT) AS period,
        CAST(value AS DOUBLE) AS value,
        unit,
        freq,
        COALESCE(data_type, 'forecast') AS data_type,
        CAST(ci_lower AS DOUBLE) AS ci_lower,
        CAST(ci_upper AS DOUBLE) AS ci_upper,
        NULL::TIMESTAMP AS vintage,
        CAST(forecast_run_date AS DATE) AS forecast_run_date,
        forecast_version,
        TRUE AS is_calculated,
        NULL::TEXT AS data_quality
    FROM gold.uk_macro_derived
    WHERE period IS NOT NULL
      AND {macro_derived_metric_col} IS NOT NULL;
    """)

    # Union + deduplicate (historical rows win over forecast if any overlap)
    con.execute("""
    CREATE OR REPLACE TEMP VIEW macro_export AS
    WITH unioned AS (
        SELECT * FROM macro_base
        UNION ALL
        SELECT * FROM macro_derived
    ),
    ranked AS (
        SELECT
            u.*,
            ROW_NUMBER() OVER (
                PARTITION BY region_code, metric_id, period
                ORDER BY CASE WHEN u.data_type = 'historical' THEN 1 ELSE 2 END
            ) AS rn
        FROM unioned u
    )
    SELECT
        region_code, region_name, region_level, metric_id,
        period, value, unit, freq, data_type,
        ci_lower, ci_upper, vintage, forecast_run_date,
        forecast_version, is_calculated, data_quality
    FROM ranked
    WHERE rn = 1
    ORDER BY region_code, metric_id, period;
    """)

    con.execute(f"COPY macro_export TO '{csv_path}' (HEADER, DELIMITER ',');")

    row_count = con.execute("SELECT COUNT(*) FROM macro_export").fetchone()[0]
    logger.info(f"  ✓ Exported {row_count:,} rows → {csv_path}")

    return csv_path

# ==============================================================================
# Export: Regional (ITL1 / ITL2 / ITL3 / LAD)
# ==============================================================================

def export_regional(con, level: str) -> Path:
    """
    Export ITL1 / ITL2 / ITL3 / LAD to CSV.

    For each level:
      - silver.<level>_history        → historical
      - gold.<level>_forecast         → base metrics (nominal_gva, gdhi_total, etc.)
      - gold.<level>_derived          → Monte Carlo ratios (productivity, income_per_worker)
    """

    logger.info(f"  • {level.upper()}...")

    # Build derived metrics SQL list for VALUES clause
    calc_vals = ",".join([f"('{m}')" for m in DERIVED_METRICS])

    # Region dimension (name + level)
    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW dim_region_{level} AS
    SELECT 
        region_code, 
        any_value(region_name) AS region_name, 
        any_value(region_level) AS region_level
    FROM silver.{level}_history 
    GROUP BY 1;
    """)

    # Metric dimension (unit + freq)
    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW dim_metric_{level} AS
    SELECT 
        metric_id, 
        any_value(unit) AS unit, 
        any_value(freq) AS freq
    FROM silver.{level}_history 
    GROUP BY 1;
    """)

    # Derived metrics lookup (for is_calculated flag)
    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW calc_metrics_{level} AS
    SELECT * FROM (VALUES {calc_vals}) AS t(metric_id);
    """)

    # Last historical year per region/metric (for base-forecast filtering)
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
    # Check if data_quality column exists (only LAD has it)
    has_data_quality_col = False
    try:
        col_check = con.execute(f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = 'silver' 
              AND table_name = '{level}_history'
              AND column_name = 'data_quality'
        """).fetchone()
        has_data_quality_col = col_check is not None
    except:
        pass
    
    if has_data_quality_col:
        # LAD has data_quality column
        con.execute(f"""
        CREATE OR REPLACE TEMP VIEW silver_norm_{level} AS
        SELECT
            s.region_code,
            s.region_name,
            s.region_level,
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
            (s.metric_id IN (SELECT metric_id FROM calc_metrics_{level})) AS is_calculated,
            COALESCE(s.data_quality, 'ONS') AS data_quality
        FROM silver.{level}_history s
        WHERE s.period IS NOT NULL;
        """)
    else:
        # ITL3/ITL2/ITL1 don't have data_quality, default to 'observed'
        con.execute(f"""
        CREATE OR REPLACE TEMP VIEW silver_norm_{level} AS
        SELECT
            s.region_code,
            s.region_name,
            s.region_level,
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
            (s.metric_id IN (SELECT metric_id FROM calc_metrics_{level})) AS is_calculated,
            'ONS' AS data_quality
        FROM silver.{level}_history s
        WHERE s.period IS NOT NULL;
        """)

    # Gold base metrics (forecast) - NO derived metrics expected here post-patch
    # We still handle the case robustly, but architecture is:
    #   gold.<level>_forecast    => primary/base metrics (+ gdhi_per_head_gbp)
    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW gold_base_{level} AS
    SELECT
        gl.region_code,
        COALESCE(gl.region_name, r.region_name) AS region_name,
        COALESCE(gl.region_level, r.region_level) AS region_level,
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
        (gl.metric_id IN (SELECT metric_id FROM calc_metrics_{level})) AS is_calculated,
        NULL::TEXT AS data_quality
    FROM gold.{level}_forecast gl
    LEFT JOIN dim_region_{level} r USING (region_code)
    LEFT JOIN dim_metric_{level} m USING (metric_id)
    LEFT JOIN last_hist_{level} lh USING (region_code, metric_id)
    WHERE gl.period IS NOT NULL 
      AND gl.metric_id IS NOT NULL
      AND (
          -- Non-derived metrics: only include forecast periods (after last historical)
          (gl.metric_id NOT IN (SELECT metric_id FROM calc_metrics_{level})
           AND (lh.last_hist_year IS NULL OR CAST(gl.period AS INT) > lh.last_hist_year))
          -- Derived metrics in base (defensive, should not occur post-patch)
          OR (gl.metric_id IN (SELECT metric_id FROM calc_metrics_{level}))
      );
    """)

    # Gold derived metrics (Monte Carlo ratios) from gold.<level>_derived
    # Always marked is_calculated = TRUE
    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW gold_derived_{level} AS
    SELECT
        gl.region_code,
        COALESCE(gl.region_name, r.region_name) AS region_name,
        COALESCE(gl.region_level, r.region_level) AS region_level,
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
        TRUE AS is_calculated,
        NULL::TEXT AS data_quality
    FROM gold.{level}_derived gl
    LEFT JOIN dim_region_{level} r USING (region_code)
    LEFT JOIN dim_metric_{level} m USING (metric_id)
    WHERE gl.period IS NOT NULL 
      AND gl.metric_id IS NOT NULL;
    """)

    # Union + deduplicate:
    #   - silver_norm_{level}   (historical)
    #   - gold_base_{level}     (base forecast metrics)
    #   - gold_derived_{level}  (Monte Carlo ratios)
    # Historical rows always win for overlapping period/metric/region
    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW {level}_export AS
    WITH unioned AS (
        SELECT * FROM silver_norm_{level}
        UNION ALL
        SELECT * FROM gold_base_{level}
        UNION ALL
        SELECT * FROM gold_derived_{level}
    ),
    ranked AS (
        SELECT 
            u.*,
            ROW_NUMBER() OVER (
                PARTITION BY region_code, metric_id, period
                ORDER BY CASE WHEN u.data_type = 'historical' THEN 1 ELSE 2 END
            ) AS rn
        FROM unioned u
    )
    SELECT
        region_code, region_name, region_level, metric_id,
        period, value, unit, freq, data_type,
        ci_lower, ci_upper, vintage, forecast_run_date,
        forecast_version, is_calculated,
        CASE
            -- NI employment metrics: relabel historical source as NISRA (not ONS).
            -- This is a display/metadata fix for the export layer only, and does not affect
            -- non-employment metrics or GB geographies.
            WHEN metric_id IN ('employment_rate_pct','unemployment_rate_pct','emp_total_jobs_ni')
             AND (region_code LIKE 'N09%' OR region_code LIKE 'TLN%' OR region_code = 'N92000002')
             AND data_type = 'historical'
            THEN 'NISRA'
            ELSE data_quality
        END AS data_quality
    FROM ranked 
    WHERE rn = 1
    ORDER BY region_code, metric_id, period;
    """)

    csv_path = EXPORT_DIR / f"{level}_latest_all.csv"
    con.execute(f"COPY {level}_export TO '{csv_path}' (HEADER, DELIMITER ',');")

    row_count = con.execute(f"SELECT COUNT(*) FROM {level}_export").fetchone()[0]
    logger.info(f"    ✓ Exported {row_count:,} rows → {csv_path}")

    return csv_path

# ==============================================================================
# Supabase Sync (unchanged interface)
# ==============================================================================

def sync_table(conn, table: str, csv_path: Path):
    """Sync a CSV file to a Supabase table (TRUNCATE + COPY)."""

    logger.info(f"  • {table}...")

    cols = sql.SQL(",").join(map(sql.Identifier, COLUMNS))

    with conn.cursor() as cur:
        # Create table if not exists
        cur.execute(
            sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})")
            .format(sql.Identifier(table), sql.SQL(DDL))
        )
        
        # Add data_quality column if it doesn't exist (for existing tables)
        try:
            cur.execute(
                sql.SQL("ALTER TABLE {} ADD COLUMN IF NOT EXISTS data_quality TEXT NULL")
                .format(sql.Identifier(table))
            )
        except Exception as e:
            # Column might already exist or table might not exist yet
            logger.debug(f"  Note: {e}")

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
    """Run full sync pipeline."""

    # Connect to DuckDB
    con = duckdb.connect(str(DB_PATH))

    # Export all levels
    csv_files = {}

    csv_files["macro"] = export_macro(con)

    logger.info("\n[2/5] Exporting ITL1/ITL2/ITL3/LAD...")
    for level in ["itl1", "itl2", "itl3", "lad"]:
        csv_files[level] = export_regional(con, level)

    con.close()

    # Sync to Supabase
    if SKIP_SUPABASE_SYNC:
        logger.warning("\n[3/5] Skipping Supabase sync (SUPABASE_URI not set).")
        logger.info("\n[4/5] ✅ EXPORT COMPLETE (CSV only)")
        logger.info("=" * 70)
        logger.info("CSVs written:")
        logger.info(f"  • {csv_files['macro']}")
        logger.info(f"  • {csv_files['itl1']}")
        logger.info(f"  • {csv_files['itl2']}")
        logger.info(f"  • {csv_files['itl3']}")
        logger.info(f"  • {csv_files['lad']}")
        logger.info("=" * 70)
        return

    logger.info("\n[3/5] Syncing to Supabase...")

    with psycopg2.connect(SUPABASE_URI) as pg:
        sync_table(pg, "macro_latest_all", csv_files["macro"])
        sync_table(pg, "itl1_latest_all", csv_files["itl1"])
        sync_table(pg, "itl2_latest_all", csv_files["itl2"])
        sync_table(pg, "itl3_latest_all", csv_files["itl3"])
        sync_table(pg, "lad_latest_all", csv_files["lad"])

    # Summary
    logger.info("\n[4/5] ✅ SYNC COMPLETE")
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