# RegionIQ Backend Makefile
# =========================
# Provides ergonomic commands for common pipeline operations.
#
# Usage:
#   make install     # Install Python dependencies
#   make bootstrap   # First run on fresh infra
#   make run         # Production pipeline
#   make dev         # Local development (lenient)
#   make help        # Show available commands

.PHONY: help install bootstrap run dev clean test lint

# Default target
help:
	@echo "RegionIQ Backend Pipeline Commands"
	@echo "==================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install      Install Python dependencies"
	@echo ""
	@echo "Pipeline Modes:"
	@echo "  make bootstrap    First run on fresh infra (creates DuckDB, skips pre-snapshot)"
	@echo "  make run          Production steady-state run (strict invariants)"
	@echo "  make dev          Local development run (lenient, skips snapshots)"
	@echo ""
	@echo "Partial Runs:"
	@echo "  make ingest       Run ingest stage only"
	@echo "  make forecast     Run forecast stages only (macro → LAD)"
	@echo "  make export       Run export/sync stage only"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean        Remove cache and log files"
	@echo "  make status       Show DuckDB table status"
	@echo ""

# -------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------

install:
	pip install -r requirements.txt

# -------------------------------------------------------------------
# Pipeline Modes
# -------------------------------------------------------------------

bootstrap:
	@echo "🚀 Running pipeline in BOOTSTRAP mode (first run)..."
	PIPELINE_MODE=bootstrap python3 run_full_forecast_pipeline.py

run:
	@echo "🏭 Running pipeline in PROD mode..."
	python3 run_full_forecast_pipeline.py

dev:
	@echo "🔧 Running pipeline in LOCAL mode..."
	PIPELINE_MODE=local python3 run_full_forecast_pipeline.py

# -------------------------------------------------------------------
# Partial Runs (useful for debugging)
# -------------------------------------------------------------------

ingest:
	PIPELINE_MODE=local python3 run_full_forecast_pipeline.py --start-from ingest --stop-at transform

forecast:
	PIPELINE_MODE=local python3 run_full_forecast_pipeline.py --start-from macro --stop-at lad

export:
	PIPELINE_MODE=local python3 run_full_forecast_pipeline.py --start-from supabase --stop-at supabase

# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------

clean:
	@echo "Cleaning cache and temporary files..."
	rm -rf data/cache/*.pkl
	rm -rf __pycache__ scripts/**/__pycache__
	@echo "✓ Cleaned"

status:
	@echo "DuckDB Table Status:"
	@python3 -c "\
import duckdb; \
con = duckdb.connect('data/lake/warehouse.duckdb', read_only=True); \
tables = con.execute(\"SELECT table_schema, table_name FROM information_schema.tables WHERE table_schema NOT IN ('information_schema', 'pg_catalog') ORDER BY 1, 2\").fetchall(); \
print(); \
[print(f'  {s}.{t}') for s, t in tables]; \
con.close()" 2>/dev/null || echo "  (DuckDB not found - run 'make bootstrap' first)"

# -------------------------------------------------------------------
# Development Helpers
# -------------------------------------------------------------------

lint:
	@echo "Running linter..."
	python3 -m py_compile run_full_forecast_pipeline.py
	python3 -m py_compile scripts/ingest/Broad_LAD_T.py
	python3 -m py_compile scripts/manifest/pre_vintage_snapshot.py
	@echo "✓ Syntax OK"

test-nisra:
	@echo "Testing NISRA API response..."
	python3 -c "\
import requests; \
import pandas as pd; \
from io import StringIO; \
url = 'https://ws-data.nisra.gov.uk/public/api.restful/PxStat.Data.Cube_API.ReadDataset/LMSLGD/CSV/1.0/en'; \
resp = requests.get(url, timeout=60); \
text = resp.content.decode('utf-8-sig'); \
df = pd.read_csv(StringIO(text)); \
df.columns = [c.strip().strip('\"') for c in df.columns]; \
print('Columns:', list(df.columns)); \
print('Sample:'); \
print(df.head(3))"

