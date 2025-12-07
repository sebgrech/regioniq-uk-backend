#!/usr/bin/env python3
# scripts/utils/data_vintage.py
"""
Data Vintage Tracking for RegionIQ

Tracks upstream data changes by hashing raw NOMIS/ONS responses.
Enables "did the world change?" detection for weekly emails.

Usage in ingest scripts:
    from utils.data_vintage import VintageTracker
    
    tracker = VintageTracker()
    
    # After fetching raw data:
    raw_csv_text = response.text
    changed = tracker.record("nomis", "NM_185_1", raw_csv_text, n_rows=len(df), min_period=1997, max_period=2023)
    
    if changed:
        log.info("New data detected!")
    
    # At end of pipeline:
    summary = tracker.get_run_summary()
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import duckdb
    HAVE_DUCKDB = True
except ImportError:
    HAVE_DUCKDB = False

log = logging.getLogger("vintage")

DUCK_PATH = Path("data/lake/warehouse.duckdb")


def _compute_hash(content: str | bytes) -> str:
    """Compute stable SHA256 hash of content."""
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()[:16]  # First 16 chars is plenty


class VintageTracker:
    """
    Tracks data vintage for upstream change detection.
    
    Stores in metadata.data_vintage:
    - data_source: 'nomis', 'ons', etc.
    - dataset_id: 'NM_185_1', 'ONS_GVA_CP', etc.
    - data_hash: SHA256 of raw response
    - n_rows, min_period, max_period: basic stats
    - last_fetched_at: timestamp
    - previous_hash: for diff detection
    """
    
    def __init__(self, duck_path: Path = DUCK_PATH):
        self.duck_path = duck_path
        self.changes_this_run: list[dict] = []
        self.no_changes_this_run: list[str] = []
        self._ensure_schema()
    
    def _ensure_schema(self):
        """Create metadata schema and tables if needed."""
        if not HAVE_DUCKDB:
            log.warning("DuckDB not available; vintage tracking disabled")
            return
        
        con = duckdb.connect(str(self.duck_path))
        try:
            con.execute("CREATE SCHEMA IF NOT EXISTS metadata")
            con.execute("""
                CREATE TABLE IF NOT EXISTS metadata.data_vintage (
                    data_source TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    data_hash TEXT NOT NULL,
                    previous_hash TEXT,
                    n_rows INTEGER,
                    min_period INTEGER,
                    max_period INTEGER,
                    last_fetched_at TIMESTAMP NOT NULL,
                    PRIMARY KEY (data_source, dataset_id)
                )
            """)
            log.debug("Ensured metadata.data_vintage exists")
        finally:
            con.close()
    
    def record(
        self,
        data_source: str,
        dataset_id: str,
        raw_content: str | bytes,
        n_rows: Optional[int] = None,
        min_period: Optional[int] = None,
        max_period: Optional[int] = None,
    ) -> bool:
        """
        Record a dataset fetch and detect if it changed.
        
        Returns True if data changed (or is new), False if identical.
        """
        if not HAVE_DUCKDB:
            return True  # Assume changed if we can't track
        
        new_hash = _compute_hash(raw_content)
        now = datetime.now(timezone.utc)
        
        con = duckdb.connect(str(self.duck_path))
        try:
            # Get previous hash
            result = con.execute("""
                SELECT data_hash FROM metadata.data_vintage
                WHERE data_source = ? AND dataset_id = ?
            """, [data_source, dataset_id]).fetchone()
            
            previous_hash = result[0] if result else None
            changed = (previous_hash != new_hash)
            
            # Upsert new record
            con.execute("""
                INSERT OR REPLACE INTO metadata.data_vintage
                (data_source, dataset_id, data_hash, previous_hash, n_rows, min_period, max_period, last_fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [data_source, dataset_id, new_hash, previous_hash, n_rows, min_period, max_period, now])
            
            # Track for summary
            dataset_key = f"{data_source}:{dataset_id}"
            if changed:
                self.changes_this_run.append({
                    "source": data_source,
                    "dataset": dataset_id,
                    "previous_hash": previous_hash,
                    "new_hash": new_hash,
                    "n_rows": n_rows,
                    "period_range": f"{min_period}-{max_period}" if min_period and max_period else None,
                    "is_new": previous_hash is None,
                })
                log.info(f"{'NEW' if previous_hash is None else 'CHANGED'}: {dataset_key}")
            else:
                self.no_changes_this_run.append(dataset_key)
                log.debug(f"UNCHANGED: {dataset_key}")
            
            return changed
            
        finally:
            con.close()
    
    def get_run_summary(self) -> dict:
        """Get summary of all changes detected this run."""
        return {
            "new_data_detected": len(self.changes_this_run) > 0,
            "datasets_changed": len(self.changes_this_run),
            "datasets_unchanged": len(self.no_changes_this_run),
            "changes": self.changes_this_run,
            "unchanged": self.no_changes_this_run,
        }
    
    def get_all_vintages(self) -> list[dict]:
        """Get current state of all tracked datasets."""
        if not HAVE_DUCKDB:
            return []
        
        con = duckdb.connect(str(self.duck_path))
        try:
            df = con.execute("""
                SELECT * FROM metadata.data_vintage
                ORDER BY data_source, dataset_id
            """).fetchdf()
            return df.to_dict(orient="records")
        finally:
            con.close()


# Convenience function for quick integration
def track_vintage(
    data_source: str,
    dataset_id: str,
    raw_content: str | bytes,
    **kwargs
) -> bool:
    """
    One-shot vintage tracking (creates tracker, records, returns changed status).
    
    Use VintageTracker class directly if you need run summaries.
    """
    tracker = VintageTracker()
    return tracker.record(data_source, dataset_id, raw_content, **kwargs)


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    tracker = VintageTracker()
    
    # Simulate two fetches
    changed1 = tracker.record("test", "TEST_001", "some,csv,data\n1,2,3", n_rows=1, min_period=2020, max_period=2023)
    print(f"First fetch changed: {changed1}")  # True (new)
    
    changed2 = tracker.record("test", "TEST_001", "some,csv,data\n1,2,3", n_rows=1, min_period=2020, max_period=2023)
    print(f"Second fetch changed: {changed2}")  # False (same)
    
    changed3 = tracker.record("test", "TEST_001", "some,csv,data\n1,2,4", n_rows=1, min_period=2020, max_period=2023)
    print(f"Third fetch changed: {changed3}")  # True (modified)
    
    print("\nRun summary:", json.dumps(tracker.get_run_summary(), indent=2))