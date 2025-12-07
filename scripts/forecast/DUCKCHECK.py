import duckdb
con = duckdb.connect("data/lake/warehouse.duckdb")

# What's in the table?
print("=== gold.itl1_forecast contents ===")
print(con.execute("""
    SELECT data_type, COUNT(*), MIN(period), MAX(period)
    FROM gold.itl1_forecast
    GROUP BY data_type
""").df())

# What forecast_run_dates exist?
print("\n=== forecast_run_dates ===")
print(con.execute("""
    SELECT forecast_run_date, data_type, COUNT(*)
    FROM gold.itl1_forecast
    GROUP BY forecast_run_date, data_type
    ORDER BY forecast_run_date DESC NULLS LAST
""").df())

# Check if metric_id column exists
print("\n=== Column names ===")
print(con.execute("SELECT * FROM gold.itl1_forecast LIMIT 0").df().columns.tolist())