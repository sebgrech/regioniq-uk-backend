#!/usr/bin/env python3
# scripts/manifest/pre_vintage_snapshot.py
"""
Pre-Pipeline Vintage Snapshot

Captures baseline state of all silver/gold tables BEFORE pipeline runs.
Enables revision detection by comparing with post-run state.

Usage:
    python3 scripts/manifest/pre_vintage_snapshot.py --run-id 20251201_073212

Outputs:
    - metadata.table_vintage_baseline (DuckDB)
    - data/pipeline/{run_id}/pre_vintage.json

Mode-aware:
    In bootstrap/local modes (PIPELINE_MODE env), gracefully handles missing database.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import duckdb
except ImportError:
    print("ERROR: duckdb required")
    sys.exit(1)

# -----------------------------
# Config
# -----------------------------
DUCK_PATH = Path("data/lake/warehouse.duckdb")
PIPELINE_DIR = Path("data/pipeline")

# Pipeline mode (from orchestrator or env)
MODE = os.getenv("PIPELINE_MODE", "prod")
LENIENT_MODES = ("local", "bootstrap")

# Tables to track (schema.table format)
TRACKED_TABLES = [
    # Silver (history)
    "silver.uk_macro_history",
    "silver.itl1_history",
    "silver.itl2_history",
    "silver.itl3_history",
    # Gold (forecasts)
    "gold.uk_macro_forecast",
    "gold.itl1_forecast",
    "gold.itl2_forecast",
    "gold.itl3_forecast",
    "silver.lad_history",
    "gold.lad_forecast",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("pre_vintage")


# -----------------------------
# Hashing
# -----------------------------
def hash_table(con: duckdb.DuckDBPyConnection, table_name: str) -> dict | None:
    """
    Compute hash + stats for a table.
    Returns None if table doesn't exist.
    """
    try:
        # Check existence
        schema, table = table_name.split(".")
        exists = con.execute("""
            SELECT COUNT(*) FROM information_schema.tables 
            WHERE table_schema = ? AND table_name = ?
        """, [schema, table]).fetchone()[0]
        
        if not exists:
            return None
        
        # Get row count
        row_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        
        if row_count == 0:
            return {
                "table": table_name,
                "exists": True,
                "row_count": 0,
                "content_hash": "empty",
                "min_period": None,
                "max_period": None,
            }
        
        # Get period range if period column exists
        cols = [r[0] for r in con.execute(f"DESCRIBE {table_name}").fetchall()]
        min_period = max_period = None
        if "period" in cols:
            result = con.execute(f"SELECT MIN(period), MAX(period) FROM {table_name}").fetchone()
            min_period, max_period = result[0], result[1]
        
        # Compute content hash (hash of sorted CSV export)
        # This is deterministic regardless of insert order
        df = con.execute(f"SELECT * FROM {table_name} ORDER BY ALL").fetchdf()
        # Exclude metadata columns that change every run
        exclude_cols = ['vintage', 'created_at', 'updated_at', 'ingested_at']
        hash_cols = [c for c in df.columns if c not in exclude_cols]
        csv_bytes = df[hash_cols].to_csv(index=False).encode("utf-8")
        content_hash = hashlib.sha256(csv_bytes).hexdigest()[:32]
        
        return {
            "table": table_name,
            "exists": True,
            "row_count": row_count,
            "content_hash": content_hash,
            "min_period": int(min_period) if min_period else None,
            "max_period": int(max_period) if max_period else None,
        }
        
    except Exception as e:
        log.warning(f"Failed to hash {table_name}: {e}")
        return None


def ensure_metadata_schema(con: duckdb.DuckDBPyConnection):
    """Create metadata schema and baseline table."""
    con.execute("CREATE SCHEMA IF NOT EXISTS metadata")
    con.execute("""
        CREATE TABLE IF NOT EXISTS metadata.table_vintage_baseline (
            run_id TEXT NOT NULL,
            table_name TEXT NOT NULL,
            row_count INTEGER,
            content_hash TEXT,
            min_period INTEGER,
            max_period INTEGER,
            captured_at TIMESTAMP NOT NULL,
            PRIMARY KEY (run_id, table_name)
        )
    """)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Capture pre-pipeline table vintage")
    parser.add_argument("--run-id", required=True, help="Pipeline run ID (e.g., 20251201_073212)")
    args = parser.parse_args()
    
    run_id = args.run_id
    now = datetime.now(timezone.utc)
    
    log.info("=" * 60)
    log.info(f"PRE-VINTAGE SNAPSHOT — Run: {run_id} (mode={MODE})")
    log.info("=" * 60)
    
    # Handle missing database based on mode
    if not DUCK_PATH.exists():
        if MODE in LENIENT_MODES:
            log.warning(f"DuckDB not found: {DUCK_PATH}")
            log.info(f"Creating empty baseline (allowed in {MODE} mode)")
            
            # Ensure directory structure exists
            DUCK_PATH.parent.mkdir(parents=True, exist_ok=True)
            
            # Save empty baseline JSON
            run_dir = PIPELINE_DIR / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            
            summary = {
                "run_id": run_id,
                "captured_at": now.isoformat(),
                "mode": MODE,
                "first_run": True,
                "tables_found": 0,
                "tables_missing": len(TRACKED_TABLES),
                "baselines": [],
            }
            
            json_path = run_dir / "pre_vintage.json"
            with open(json_path, "w") as f:
                json.dump(summary, f, indent=2)
            
            log.info(f"Created empty baseline: {json_path}")
            log.info("=" * 60)
            return  # Exit successfully
        else:
            log.error(f"DuckDB not found: {DUCK_PATH}")
            log.error("Hint: Use --mode bootstrap for first run on fresh infra")
            sys.exit(1)
    
    con = duckdb.connect(str(DUCK_PATH))
    ensure_metadata_schema(con)
    
    results = []
    tables_found = 0
    tables_missing = 0
    
    for table_name in TRACKED_TABLES:
        info = hash_table(con, table_name)
        if info:
            tables_found += 1
            results.append(info)
            
            # Store in DuckDB
            con.execute("""
                INSERT OR REPLACE INTO metadata.table_vintage_baseline
                (run_id, table_name, row_count, content_hash, min_period, max_period, captured_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                run_id,
                table_name,
                info["row_count"],
                info["content_hash"],
                info["min_period"],
                info["max_period"],
                now,
            ])
            
            log.info(f"  ✓ {table_name}: {info['row_count']:,} rows, hash={info['content_hash'][:8]}...")
        else:
            tables_missing += 1
            log.info(f"  - {table_name}: not found (will be created)")
    
    con.close()
    
    # Save JSON summary
    run_dir = PIPELINE_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "run_id": run_id,
        "captured_at": now.isoformat(),
        "mode": MODE,
        "first_run": False,
        "tables_found": tables_found,
        "tables_missing": tables_missing,
        "baselines": results,
    }
    
    json_path = run_dir / "pre_vintage.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    log.info("-" * 60)
    log.info(f"Captured {tables_found} tables, {tables_missing} not yet created")
    log.info(f"Saved: {json_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()