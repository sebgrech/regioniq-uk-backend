from dotenv import load_dotenv
load_dotenv()  # load .env variables

import os
import duckdb
import psycopg2
from psycopg2 import sql
from pathlib import Path

# ========= Config =========
EXPORT_DIR = Path("data/export")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = "data/lake/warehouse.duckdb"

# Supabase (use Supavisor pooler URI, e.g. postgres://postgres.<PROJECT_REF>:<PWD>@aws-1-eu-west-2.pooler.supabase.com:6543/postgres)
SUPABASE_URI = os.getenv("SUPABASE_URI", "")

# Always include BOTH observed + calculated; flag with is_calculated
METRIC_WHITELIST = [
    "population_total",
    "gdhi_total_mn_gbp",
    "gdhi_per_head_gbp",   # keep only if observed in silver; remove here if derived locally
    "nominal_gva_mn_gbp",
    "emp_total_jobs",
]
CALC_METRICS = [
    "employment_rate",
    # "gdhi_per_head_gbp",  # add here only if it’s derived locally
]

# ========= Supabase DDL (15 columns total) =========
SCHEMA = "public"
TABLE_LATEST = "itl1_latest_all"
TABLE_ALL    = "itl1_all"

DDL_COMMON = """
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

DDL_LATEST = f"""
CREATE TABLE IF NOT EXISTS {SCHEMA}.{TABLE_LATEST} (
  {DDL_COMMON},
  CONSTRAINT {TABLE_LATEST}_pk PRIMARY KEY (region_code, metric_id, period)
);
"""

DDL_ALL = f"""
CREATE TABLE IF NOT EXISTS {SCHEMA}.{TABLE_ALL} (
  {DDL_COMMON}
);
"""

COLUMNS_ORDER = [
    "region_code","region_name","region_level","metric_id",
    "period","value","unit","freq","data_type",
    "ci_lower","ci_upper","vintage","forecast_run_date",
    "forecast_version","is_calculated"
]

# ========= DuckDB =========
con = duckdb.connect(DB_PATH)

# --- Dimensions from SILVER ---
con.execute("""
CREATE OR REPLACE TEMP VIEW dim_region AS
SELECT
  region_code,
  any_value(region_name)  AS region_name,
  any_value(region_level) AS region_level
FROM silver.itl1_history
GROUP BY 1;
""")

con.execute("""
CREATE OR REPLACE TEMP VIEW dim_metric AS
SELECT
  metric_id,
  any_value(unit) AS unit,
  any_value(freq) AS freq
FROM silver.itl1_history
GROUP BY 1;
""")

# --- Last historical year per series (from SILVER) ---
con.execute("""
CREATE OR REPLACE TEMP VIEW last_hist AS
SELECT
  region_code,
  metric_id,
  MAX(CAST(period AS INT)) AS last_hist_year
FROM silver.itl1_history
GROUP BY 1,2;
""")

# --- Global last historical year across SILVER (used to relabel calc metrics) ---
con.execute("""
CREATE OR REPLACE TEMP VIEW global_hist AS
SELECT MAX(CAST(period AS INT)) AS global_last_hist_year
FROM silver.itl1_history;
""")

# --- Whitelists / Calculated sets ---
if METRIC_WHITELIST:
    values = ",".join([f"('{m}')" for m in METRIC_WHITELIST])
    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW metric_whitelist AS
    SELECT * FROM (VALUES {values}) AS t(metric_id);
    """)
else:
    con.execute("CREATE OR REPLACE TEMP VIEW metric_whitelist AS SELECT metric_id WHERE 1=0;")

if CALC_METRICS:
    calc_values = ",".join([f"('{m}')" for m in CALC_METRICS])
    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW calc_metrics AS
    SELECT * FROM (VALUES {calc_values}) AS t(metric_id);
    """)
else:
    con.execute("CREATE OR REPLACE TEMP VIEW calc_metrics AS SELECT metric_id WHERE 1=0;")

# Always include both sets
con.execute("""
CREATE OR REPLACE TEMP VIEW allowed_metrics AS
SELECT metric_id FROM metric_whitelist
UNION
SELECT metric_id FROM calc_metrics;
""")

# --- SILVER (observed history), period as INT ---
con.execute("""
CREATE OR REPLACE TEMP VIEW silver_norm AS
SELECT
  s.region_code,
  s.region_name,
  s.region_level,
  s.metric_id,
  CAST(s.period AS INT)         AS period,
  CAST(s.value  AS DOUBLE)      AS value,
  s.unit,
  s.freq,
  'historical'                  AS data_type,
  NULL::DOUBLE                  AS ci_lower,
  NULL::DOUBLE                  AS ci_upper,
  CAST(s.vintage AS TIMESTAMP)  AS vintage,
  NULL::DATE                    AS forecast_run_date,
  NULL::VARCHAR                 AS forecast_version,
  (s.metric_id IN (SELECT metric_id FROM calc_metrics)) AS is_calculated
FROM silver.itl1_history s
JOIN allowed_metrics a USING (metric_id)
WHERE s.period IS NOT NULL;
""")

# --- GOLD (ALL VINTAGES), include both sets; relabel calc-history as 'historical' ---
con.execute("""
CREATE OR REPLACE TEMP VIEW gold_all_filtered AS
WITH gh AS (SELECT global_last_hist_year FROM global_hist)
SELECT
  g.region_code,
  COALESCE(g.region_name, r.region_name)    AS region_name,
  COALESCE(g.region_level, r.region_level)  AS region_level,
  g.metric_id,
  CAST(g.period AS INT)                     AS period,
  CAST(g.value  AS DOUBLE)                  AS value,
  COALESCE(g.unit, m.unit)                  AS unit,
  COALESCE(g.freq, m.freq)                  AS freq,
  CASE
    WHEN g.metric_id IN (SELECT metric_id FROM calc_metrics)
         AND CAST(g.period AS INT) <= (SELECT global_last_hist_year FROM gh)
      THEN 'historical'
    ELSE 'forecast'
  END                                       AS data_type,
  CAST(g.ci_lower AS DOUBLE)                AS ci_lower,
  CAST(g.ci_upper AS DOUBLE)                AS ci_upper,
  NULL::TIMESTAMP                           AS vintage,
  CAST(g.forecast_run_date AS DATE)         AS forecast_run_date,
  g.forecast_version,
  (g.metric_id IN (SELECT metric_id FROM calc_metrics)) AS is_calculated
FROM gold.itl1_forecast g
JOIN allowed_metrics a USING (metric_id)
LEFT JOIN dim_region r USING (region_code)
LEFT JOIN dim_metric m  USING (metric_id)
LEFT JOIN last_hist  lh USING (region_code, metric_id)
WHERE g.period IS NOT NULL
  AND (
        -- For non-calculated metrics: only keep periods strictly after last observed year
        (g.metric_id NOT IN (SELECT metric_id FROM calc_metrics)
         AND (lh.last_hist_year IS NULL OR CAST(g.period AS INT) > lh.last_hist_year))
        OR
        -- For calculated metrics: keep all periods; label via CASE above
        (g.metric_id IN (SELECT metric_id FROM calc_metrics))
      );
""")

# --- GOLD (LATEST VINTAGE), same relabel logic ---
con.execute("""
CREATE OR REPLACE TEMP VIEW gold_latest_filtered AS
WITH gh AS (SELECT global_last_hist_year FROM global_hist)
SELECT
  gl.region_code,
  COALESCE(gl.region_name, r.region_name)    AS region_name,
  COALESCE(gl.region_level, r.region_level)  AS region_level,
  gl.metric_id,
  CAST(gl.period AS INT)                     AS period,
  CAST(gl.value  AS DOUBLE)                  AS value,
  COALESCE(gl.unit, m.unit)                  AS unit,
  COALESCE(gl.freq, m.freq)                  AS freq,
  CASE
    WHEN gl.metric_id IN (SELECT metric_id FROM calc_metrics)
         AND CAST(gl.period AS INT) <= (SELECT global_last_hist_year FROM gh)
      THEN 'historical'
    ELSE 'forecast'
  END                                        AS data_type,
  CAST(gl.ci_lower AS DOUBLE)                AS ci_lower,
  CAST(gl.ci_upper AS DOUBLE)                AS ci_upper,
  NULL::TIMESTAMP                            AS vintage,
  CAST(gl.forecast_run_date AS DATE)         AS forecast_run_date,
  gl.forecast_version,
  (gl.metric_id IN (SELECT metric_id FROM calc_metrics)) AS is_calculated
FROM gold.itl1_latest gl
JOIN allowed_metrics a USING (metric_id)
LEFT JOIN dim_region r USING (region_code)
LEFT JOIN dim_metric m  USING (metric_id)
LEFT JOIN last_hist  lh USING (region_code, metric_id)
WHERE gl.period IS NOT NULL
  AND (
        (gl.metric_id NOT IN (SELECT metric_id FROM calc_metrics)
         AND (lh.last_hist_year IS NULL OR CAST(gl.period AS INT) > lh.last_hist_year))
        OR
        (gl.metric_id IN (SELECT metric_id FROM calc_metrics))
      );
""")

# --- Unified (history + ALL vintages) ---
# (Explicit column list to guarantee exact 15 columns/order in CSV)
con.execute(f"""
CREATE OR REPLACE TEMP VIEW itl1_all AS
SELECT
  region_code, region_name, region_level, metric_id,
  period, value, unit, freq, data_type,
  ci_lower, ci_upper, vintage, forecast_run_date,
  forecast_version, is_calculated
FROM (
  SELECT
    s.region_code, s.region_name, s.region_level, s.metric_id,
    s.period, s.value, s.unit, s.freq, s.data_type,
    s.ci_lower, s.ci_upper, s.vintage, s.forecast_run_date,
    s.forecast_version, s.is_calculated
  FROM silver_norm s
  UNION ALL
  SELECT
    g.region_code, g.region_name, g.region_level, g.metric_id,
    g.period, g.value, g.unit, g.freq, g.data_type,
    g.ci_lower, g.ci_upper, g.vintage, g.forecast_run_date,
    g.forecast_version, g.is_calculated
  FROM gold_all_filtered g
)
ORDER BY region_code, metric_id, period, forecast_run_date NULLS FIRST;
""")

# --- Unified (history + LATEST), prefer history if overlap ---
# IMPORTANT: Project only the 15 output columns (avoid leaking pref/rn).
con.execute(f"""
CREATE OR REPLACE TEMP VIEW itl1_latest_all AS
WITH ranked AS (
  SELECT
    u.*,
    CASE WHEN u.data_type='historical' THEN 1 ELSE 2 END AS pref,
    ROW_NUMBER() OVER (
      PARTITION BY region_code, metric_id, period
      ORDER BY CASE WHEN u.data_type='historical' THEN 1 ELSE 2 END
    ) AS rn
  FROM (
    SELECT * FROM silver_norm
    UNION ALL
    SELECT * FROM gold_latest_filtered
  ) u
)
SELECT
  region_code, region_name, region_level, metric_id,
  period, value, unit, freq, data_type,
  ci_lower, ci_upper, vintage, forecast_run_date,
  forecast_version, is_calculated
FROM ranked
WHERE rn = 1
ORDER BY region_code, metric_id, period;
""")

# ========= Export CSVs (exactly 15 columns) =========
csv_latest = EXPORT_DIR / "itl1_latest_all.csv"
csv_all    = EXPORT_DIR / "itl1_all.csv"
con.execute(f"COPY itl1_latest_all TO '{csv_latest}' (HEADER, DELIMITER ',');")
con.execute(f"COPY itl1_all        TO '{csv_all}'     (HEADER, DELIMITER ',');")

# Quick sanity: counts, ranges, calc shares
print("Row counts (itl1_latest_all):")
print(con.execute("""
SELECT data_type, COUNT(*) AS n_rows,
       MIN(period) AS min_y,
       MAX(period) AS max_y,
       SUM(CASE WHEN is_calculated THEN 1 ELSE 0 END) AS n_calc
FROM itl1_latest_all
GROUP BY data_type;
""").df())

print("Any forecast at/before last history year in latest_all (should be 0):")
print(con.execute("""
WITH lh AS (
  SELECT region_code, metric_id, MAX(period) AS last_hist_year
  FROM itl1_latest_all WHERE data_type='historical'
  GROUP BY 1,2
)
SELECT COUNT(*) AS bad_overlap
FROM itl1_latest_all u
JOIN lh USING (region_code, metric_id)
WHERE u.data_type='forecast' AND u.period <= lh.last_hist_year;
""").df())

con.close()
print("✅ Exported →", csv_latest, "and", csv_all)

# ========= Push to Supabase (automatic) =========
if not SUPABASE_URI:
    print("⚠️ SUPABASE_URI not set. Skipping Supabase sync.")
else:
    def ensure_tables_and_copy(conn, table_name: str, csv_path: Path):
        ddl = DDL_LATEST if table_name == TABLE_LATEST else DDL_ALL
        cols_csv = sql.SQL(",").join(map(sql.Identifier, COLUMNS_ORDER))
        with conn, conn.cursor() as cur:
            cur.execute(ddl)  # idempotent create
            cur.execute(sql.SQL("TRUNCATE {}.{};")
                        .format(sql.Identifier(SCHEMA), sql.Identifier(table_name)))
            copy_sql = sql.SQL("COPY {}.{} ({}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE)")
            copy_sql = copy_sql.format(sql.Identifier(SCHEMA),
                                       sql.Identifier(table_name),
                                       cols_csv)
            with open(csv_path, "r", encoding="utf-8") as f:
                cur.copy_expert(copy_sql.as_string(conn), f)
            cur.execute(sql.SQL("SELECT COUNT(*) FROM {}.{};")
                        .format(sql.Identifier(SCHEMA), sql.Identifier(table_name)))
            print(f"✅ Supabase sync → {table_name} ({cur.fetchone()[0]:,} rows)")

    try:
        conn = psycopg2.connect(SUPABASE_URI)
        ensure_tables_and_copy(conn, TABLE_LATEST, csv_latest)
        ensure_tables_and_copy(conn, TABLE_ALL,    csv_all)
        # Optional QA
        with conn, conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT MIN(period), MAX(period) FROM {}.{};")
                        .format(sql.Identifier(SCHEMA), sql.Identifier(TABLE_LATEST)))
            mn, mx = cur.fetchone()
            print(f"🧪 Supabase QA ({TABLE_LATEST}) period range: {mn}–{mx}")
        conn.close()
    except Exception as e:
        print("❌ Supabase sync failed:", repr(e))
        print("Hint: Check SUPABASE_URI, pooler mode, and privileges.")