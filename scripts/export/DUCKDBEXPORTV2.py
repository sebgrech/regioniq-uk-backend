#!/usr/bin/env python3
"""RegionIQ → Supabase (TRUNCATE + COPY)"""

from dotenv import load_dotenv
load_dotenv()

import os
import duckdb
import psycopg2
from psycopg2 import sql
from pathlib import Path

# Config
EXPORT_DIR = Path("data/export")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = "data/lake/warehouse.duckdb"
SUPABASE_URI = os.getenv("SUPABASE_URI", "")

if not SUPABASE_URI:
    print("ERROR: SUPABASE_URI not set in .env")
    exit(1)

# Schema (matches itl1_latest_all exactly)
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
is_calculated BOOLEAN NOT NULL
"""

COLUMNS = [
    "region_code", "region_name", "region_level", "metric_id",
    "period", "value", "unit", "freq", "data_type",
    "ci_lower", "ci_upper", "vintage", "forecast_run_date",
    "forecast_version", "is_calculated"
]

print("="*70)
print("REGIONIQ → SUPABASE (TRUNCATE + COPY)")
print("="*70)

con = duckdb.connect(DB_PATH)

# ========= 1. MACRO_LATEST_ALL =========

print("\n1. Preparing macro_latest_all...")

con.execute("""
CREATE OR REPLACE TEMP VIEW macro_export AS
SELECT
  region_code, region_name, region_level, metric_id,
  CAST(period AS INT) AS period,
  CAST(value AS DOUBLE) AS value,
  unit, freq, data_type,
  CAST(ci_lower AS DOUBLE) AS ci_lower,
  CAST(ci_upper AS DOUBLE) AS ci_upper,
  NULL::TIMESTAMP AS vintage,
  CAST(forecast_run_date AS DATE) AS forecast_run_date,
  forecast_version,
  FALSE AS is_calculated
FROM gold.uk_macro_forecast
WHERE period IS NOT NULL AND metric_id IS NOT NULL;
""")

macro_csv = EXPORT_DIR / "macro_latest_all.csv"
con.execute(f"COPY macro_export TO '{macro_csv}' (HEADER, DELIMITER ',');")
print(f"✓ Exported {macro_csv}")

# ========= 2. ITL1_LATEST_ALL =========

print("\n2. Preparing itl1_latest_all...")

# Dimensions
con.execute("""
CREATE OR REPLACE TEMP VIEW dim_region AS
SELECT region_code, any_value(region_name) AS region_name, any_value(region_level) AS region_level
FROM silver.itl1_history GROUP BY 1;
""")

con.execute("""
CREATE OR REPLACE TEMP VIEW dim_metric AS
SELECT metric_id, any_value(unit) AS unit, any_value(freq) AS freq
FROM silver.itl1_history GROUP BY 1;
""")

calc_metrics = ['employment_rate', 'income_per_worker_gbp', 'productivity_gbp_per_job']
calc_vals = ",".join([f"('{m}')" for m in calc_metrics])
con.execute(f"CREATE OR REPLACE TEMP VIEW calc_metrics AS SELECT * FROM (VALUES {calc_vals}) AS t(metric_id);")

con.execute("""
CREATE OR REPLACE TEMP VIEW global_hist AS
SELECT MAX(CAST(period AS INT)) AS global_last_hist_year FROM silver.itl1_history;
""")

con.execute("""
CREATE OR REPLACE TEMP VIEW last_hist AS
SELECT region_code, metric_id, MAX(CAST(period AS INT)) AS last_hist_year
FROM silver.itl1_history GROUP BY 1,2;
""")

# Silver
con.execute("""
CREATE OR REPLACE TEMP VIEW silver_norm AS
SELECT
  s.region_code, s.region_name, s.region_level, s.metric_id,
  CAST(s.period AS INT) AS period, CAST(s.value AS DOUBLE) AS value,
  s.unit, s.freq, 'historical' AS data_type,
  NULL::DOUBLE AS ci_lower, NULL::DOUBLE AS ci_upper,
  CAST(s.vintage AS TIMESTAMP) AS vintage,
  NULL::DATE AS forecast_run_date, NULL::VARCHAR AS forecast_version,
  (s.metric_id IN (SELECT metric_id FROM calc_metrics)) AS is_calculated
FROM silver.itl1_history s WHERE s.period IS NOT NULL;
""")

# Gold
con.execute("""
CREATE OR REPLACE TEMP VIEW gold_filtered AS
WITH gh AS (SELECT global_last_hist_year FROM global_hist)
SELECT
  gl.region_code,
  COALESCE(gl.region_name, r.region_name) AS region_name,
  COALESCE(gl.region_level, r.region_level) AS region_level,
  gl.metric_id,
  CAST(gl.period AS INT) AS period,
  CAST(gl.value AS DOUBLE) AS value,
  COALESCE(gl.unit, m.unit) AS unit,
  COALESCE(gl.freq, m.freq) AS freq,
  CASE
    WHEN gl.metric_id IN (SELECT metric_id FROM calc_metrics)
         AND CAST(gl.period AS INT) <= (SELECT global_last_hist_year FROM gh)
      THEN 'historical'
    ELSE 'forecast'
  END AS data_type,
  CAST(gl.ci_lower AS DOUBLE) AS ci_lower,
  CAST(gl.ci_upper AS DOUBLE) AS ci_upper,
  NULL::TIMESTAMP AS vintage,
  CAST(gl.forecast_run_date AS DATE) AS forecast_run_date,
  gl.forecast_version,
  (gl.metric_id IN (SELECT metric_id FROM calc_metrics)) AS is_calculated
FROM gold.itl1_latest gl
LEFT JOIN dim_region r USING (region_code)
LEFT JOIN dim_metric m USING (metric_id)
LEFT JOIN last_hist lh USING (region_code, metric_id)
WHERE gl.period IS NOT NULL AND gl.metric_id IS NOT NULL
  AND (
    (gl.metric_id NOT IN (SELECT metric_id FROM calc_metrics)
     AND (lh.last_hist_year IS NULL OR CAST(gl.period AS INT) > lh.last_hist_year))
    OR (gl.metric_id IN (SELECT metric_id FROM calc_metrics))
  );
""")

# Unified
con.execute("""
CREATE OR REPLACE TEMP VIEW itl1_export AS
WITH ranked AS (
  SELECT u.*,
    ROW_NUMBER() OVER (
      PARTITION BY region_code, metric_id, period
      ORDER BY CASE WHEN u.data_type='historical' THEN 1 ELSE 2 END
    ) AS rn
  FROM (
    SELECT * FROM silver_norm
    UNION ALL
    SELECT * FROM gold_filtered
  ) u
)
SELECT
  region_code, region_name, region_level, metric_id,
  period, value, unit, freq, data_type,
  ci_lower, ci_upper, vintage, forecast_run_date,
  forecast_version, is_calculated
FROM ranked WHERE rn = 1
ORDER BY region_code, metric_id, period;
""")

itl1_csv = EXPORT_DIR / "itl1_latest_all.csv"
con.execute(f"COPY itl1_export TO '{itl1_csv}' (HEADER, DELIMITER ',');")
print(f"✓ Exported {itl1_csv}")

con.close()

# ========= SYNC TO SUPABASE =========

def sync_table(conn, table_name: str, csv_path: Path):
    cols = sql.SQL(",").join(map(sql.Identifier, COLUMNS))
    
    with conn, conn.cursor() as cur:
        # Create if not exists
        cur.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({DDL})")
        
        # TRUNCATE (atomic delete)
        cur.execute(sql.SQL("TRUNCATE {};").format(sql.Identifier(table_name)))
        
        # COPY (bulk load)
        copy_sql = sql.SQL("COPY {} ({}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE)").format(
            sql.Identifier(table_name), cols
        )
        
        with open(csv_path, "r") as f:
            cur.copy_expert(copy_sql.as_string(conn), f)
        
        # Verify
        cur.execute(sql.SQL("SELECT COUNT(*) FROM {};").format(sql.Identifier(table_name)))
        print(f"  ✅ {table_name}: {cur.fetchone()[0]:,} rows")

conn = psycopg2.connect(SUPABASE_URI)
sync_table(conn, "macro_latest_all", macro_csv)
sync_table(conn, "itl1_latest_all", itl1_csv)
conn.close()

print("\n" + "="*70)
print("✅ COMPLETE")
print("="*70)