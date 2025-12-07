import duckdb

con = duckdb.connect("data/lake/warehouse.duckdb")

# list all tables across schemas
print("All tables:", 
      con.execute("SELECT table_schema, table_name FROM information_schema.tables").fetchall())

# check if silver.itl1_history exists and row count
print("Silver history rows:", 
      con.execute("SELECT COUNT(*) FROM silver.itl1_history").fetchone()[0])
