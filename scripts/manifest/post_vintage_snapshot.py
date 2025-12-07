#!/usr/bin/env python3
# scripts/manifest/post_vintage_snapshot.py
"""
Post-Pipeline Vintage Snapshot

Compares current table state against pre-run baseline.
Detects which tables changed, by how much, and flags revisions.

Usage:
    python3 scripts/manifest/post_vintage_snapshot.py --run-id 20251201_073212

Requires:
    - pre_vintage_snapshot.py must have run first with same run_id

Outputs:
    - metadata.table_vintage_post (DuckDB)
    - metadata.table_vintage_diff (DuckDB)
    - data/pipeline/{run_id}/post_vintage.json
    - data/pipeline/{run_id}/vintage_diff.json
"""

import argparse
import hashlib
import json
import logging
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

# Must match pre_vintage_snapshot.py
TRACKED_TABLES = [
    "silver.uk_macro_history",
    "silver.itl1_history",
    "silver.itl2_history",
    "silver.itl3_history",
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
log = logging.getLogger("post_vintage")


# -----------------------------
# Hashing (same as pre)
# -----------------------------
def hash_table(con: duckdb.DuckDBPyConnection, table_name: str) -> dict | None:
    """Compute hash + stats for a table."""
    try:
        schema, table = table_name.split(".")
        exists = con.execute("""
            SELECT COUNT(*) FROM information_schema.tables 
            WHERE table_schema = ? AND table_name = ?
        """, [schema, table]).fetchone()[0]
        
        if not exists:
            return None
        
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
        
        cols = [r[0] for r in con.execute(f"DESCRIBE {table_name}").fetchall()]
        min_period = max_period = None
        if "period" in cols:
            result = con.execute(f"SELECT MIN(period), MAX(period) FROM {table_name}").fetchone()
            min_period, max_period = result[0], result[1]
        
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
    """Create metadata tables for post + diff."""
    con.execute("CREATE SCHEMA IF NOT EXISTS metadata")
    
    con.execute("""
        CREATE TABLE IF NOT EXISTS metadata.table_vintage_post (
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
    
    con.execute("""
        CREATE TABLE IF NOT EXISTS metadata.table_vintage_diff (
            run_id TEXT NOT NULL,
            table_name TEXT NOT NULL,
            status TEXT NOT NULL,
            rows_before INTEGER,
            rows_after INTEGER,
            rows_delta INTEGER,
            hash_before TEXT,
            hash_after TEXT,
            hash_changed BOOLEAN,
            period_extended BOOLEAN,
            max_period_before INTEGER,
            max_period_after INTEGER,
            PRIMARY KEY (run_id, table_name)
        )
    """)


def get_baseline(con: duckdb.DuckDBPyConnection, run_id: str, table_name: str) -> dict | None:
    """Fetch pre-run baseline for a table."""
    result = con.execute("""
        SELECT row_count, content_hash, min_period, max_period
        FROM metadata.table_vintage_baseline
        WHERE run_id = ? AND table_name = ?
    """, [run_id, table_name]).fetchone()
    
    if not result:
        return None
    
    return {
        "row_count": result[0],
        "content_hash": result[1],
        "min_period": result[2],
        "max_period": result[3],
    }


def compute_diff(table_name: str, before: dict | None, after: dict | None) -> dict:
    """Compare before/after state and produce diff record."""
    
    # Table didn't exist before, exists now
    if before is None and after is not None:
        return {
            "table": table_name,
            "status": "CREATED",
            "rows_before": None,
            "rows_after": after["row_count"],
            "rows_delta": after["row_count"],
            "hash_before": None,
            "hash_after": after["content_hash"],
            "hash_changed": True,
            "period_extended": False,
            "max_period_before": None,
            "max_period_after": after.get("max_period"),
        }
    
    # Table existed before, doesn't exist now (shouldn't happen normally)
    if before is not None and after is None:
        return {
            "table": table_name,
            "status": "DROPPED",
            "rows_before": before["row_count"],
            "rows_after": None,
            "rows_delta": -before["row_count"],
            "hash_before": before["content_hash"],
            "hash_after": None,
            "hash_changed": True,
            "period_extended": False,
            "max_period_before": before.get("max_period"),
            "max_period_after": None,
        }
    
    # Both exist — compare
    hash_changed = before["content_hash"] != after["content_hash"]
    rows_delta = (after["row_count"] or 0) - (before["row_count"] or 0)
    
    max_before = before.get("max_period")
    max_after = after.get("max_period")
    period_extended = (max_before and max_after and max_after > max_before)
    
    if hash_changed:
        status = "MODIFIED"
    else:
        status = "UNCHANGED"
    
    return {
        "table": table_name,
        "status": status,
        "rows_before": before["row_count"],
        "rows_after": after["row_count"],
        "rows_delta": rows_delta,
        "hash_before": before["content_hash"],
        "hash_after": after["content_hash"],
        "hash_changed": hash_changed,
        "period_extended": period_extended,
        "max_period_before": max_before,
        "max_period_after": max_after,
    }


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Capture post-pipeline table vintage and diff")
    parser.add_argument("--run-id", required=True, help="Pipeline run ID (must match pre-snapshot)")
    args = parser.parse_args()
    
    run_id = args.run_id
    now = datetime.now(timezone.utc)
    
    log.info("=" * 60)
    log.info(f"POST-VINTAGE SNAPSHOT — Run: {run_id}")
    log.info("=" * 60)
    
    if not DUCK_PATH.exists():
        log.error(f"DuckDB not found: {DUCK_PATH}")
        sys.exit(1)
    
    run_dir = PIPELINE_DIR / run_id
    if not run_dir.exists():
        log.warning(f"Run directory not found: {run_dir} — creating")
        run_dir.mkdir(parents=True, exist_ok=True)
    
    con = duckdb.connect(str(DUCK_PATH))
    ensure_metadata_schema(con)
    
    post_results = []
    diff_results = []
    
    tables_unchanged = 0
    tables_modified = 0
    tables_created = 0
    
    for table_name in TRACKED_TABLES:
        # Get current state
        after = hash_table(con, table_name)
        
        # Get baseline
        before = get_baseline(con, run_id, table_name)
        
        # Compute diff
        diff = compute_diff(table_name, before, after)
        
        # Store post state
        if after:
            post_results.append(after)
            con.execute("""
                INSERT OR REPLACE INTO metadata.table_vintage_post
                (run_id, table_name, row_count, content_hash, min_period, max_period, captured_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                run_id, table_name, after["row_count"], after["content_hash"],
                after.get("min_period"), after.get("max_period"), now
            ])
        
        # Store diff
        diff_results.append(diff)
        con.execute("""
            INSERT OR REPLACE INTO metadata.table_vintage_diff
            (run_id, table_name, status, rows_before, rows_after, rows_delta,
             hash_before, hash_after, hash_changed, period_extended,
             max_period_before, max_period_after)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            run_id, diff["table"], diff["status"], diff["rows_before"],
            diff["rows_after"], diff["rows_delta"], diff["hash_before"],
            diff["hash_after"], diff["hash_changed"], diff["period_extended"],
            diff["max_period_before"], diff["max_period_after"]
        ])
        
        # Log
        status_icon = {"UNCHANGED": "·", "MODIFIED": "Δ", "CREATED": "+", "DROPPED": "✗"}
        icon = status_icon.get(diff["status"], "?")
        
        if diff["status"] == "UNCHANGED":
            tables_unchanged += 1
            log.info(f"  {icon} {table_name}: unchanged")
        elif diff["status"] == "CREATED":
            tables_created += 1
            log.info(f"  {icon} {table_name}: CREATED ({diff['rows_after']:,} rows)")
        elif diff["status"] == "MODIFIED":
            tables_modified += 1
            delta_str = f"+{diff['rows_delta']}" if diff['rows_delta'] >= 0 else str(diff['rows_delta'])
            ext_str = " [period extended]" if diff["period_extended"] else ""
            log.info(f"  {icon} {table_name}: MODIFIED ({delta_str} rows){ext_str}")
        else:
            log.info(f"  {icon} {table_name}: {diff['status']}")
    
    con.close()
    
    # Save JSONs
    post_summary = {
        "run_id": run_id,
        "captured_at": now.isoformat(),
        "tables": post_results,
    }
    with open(run_dir / "post_vintage.json", "w") as f:
        json.dump(post_summary, f, indent=2)
    
    diff_summary = {
        "run_id": run_id,
        "captured_at": now.isoformat(),
        "tables_unchanged": tables_unchanged,
        "tables_modified": tables_modified,
        "tables_created": tables_created,
        "any_changes": (tables_modified + tables_created) > 0,
        "diffs": diff_results,
    }
    with open(run_dir / "vintage_diff.json", "w") as f:
        json.dump(diff_summary, f, indent=2)
    
    # Summary
    log.info("-" * 60)
    log.info(f"Unchanged: {tables_unchanged} | Modified: {tables_modified} | Created: {tables_created}")
    log.info(f"Saved: {run_dir / 'vintage_diff.json'}")
    log.info("=" * 60)
    
    # Exit code: 0 = no changes, 1 = changes detected (useful for CI)
    if tables_modified + tables_created > 0:
        log.info("Changes detected in this run.")
    else:
        log.info("No table changes detected.")


if __name__ == "__main__":
    main()