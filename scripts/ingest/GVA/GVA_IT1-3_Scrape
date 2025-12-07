#!/usr/bin/env python3
"""
Scraper for: ONS Regional gross value added (balanced) by industry: all ITL regions
URL: https://www.ons.gov.uk/economy/grossvalueaddedgva/datasets/nominalandrealregionalgrossvalueaddedbalancedbyindustry

Behavior (API-like):
- Fetch landing page
- Extract current XLSX link + release date
- Download file, compute SHA256
- Compare to last stored metadata; if new, save with versioned name and update _latest.json
- No parsing/cleaning here (that’s clean_gva.py’s job)

Run:
  python scripts/ingest/scrape_gva.py
Options:
  --outdir data/raw/gva
  --force         (download even if hash matches)
  --check-only    (don’t write anything; exit code 0=new/1=no change)
"""

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ONS_URL = ("https://www.ons.gov.uk/economy/grossvalueaddedgva/datasets/"
           "nominalandrealregionalgrossvalueaddedbalancedbyindustry")

DEFAULT_OUTDIR = Path("data/raw/gva")
LATEST_JSON = "_latest.json"

HEADERS = {
    "User-Agent": "RegionIQ-GVA-Scraper/1.0 (+https://regioniq.example)"
}


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )


def fetch_landing(url: str) -> dict:
    """Return dict with 'html', 'release_date'(date|None), 'download_url'(str)."""
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    html = r.text
    soup = BeautifulSoup(html, "html.parser")

    # Find current XLSX link (green 'xlsx (12.0 MB)' button)
    a = soup.find("a", href=re.compile(r"\.xlsx$", re.I))
    if not a:
        raise RuntimeError("Could not locate XLSX link on the page.")
    href = a.get("href")
    download_url = href if href.startswith("http") else f"https://www.ons.gov.uk{href}"

    # Try to read 'Release date: 17 April 2025'
    text = soup.get_text(" ", strip=True)
    m = re.search(r"Release date:\s+(\d{1,2}\s+\w+\s+\d{4})", text, flags=re.I)
    release_date = None
    if m:
        try:
            release_date = dt.datetime.strptime(m.group(1), "%d %B %Y").date()
        except Exception:
            pass

    return {"html": html, "release_date": release_date, "download_url": download_url}


def download_bytes(url: str) -> tuple[bytes, str]:
    """Download URL, return (content, sha256 hex)."""
    r = requests.get(url, headers=HEADERS, timeout=120)
    r.raise_for_status()
    blob = r.content
    sha = hashlib.sha256(blob).hexdigest()
    return blob, sha


def read_latest(meta_path: Path) -> dict | None:
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return None


def write_latest(meta_path: Path, payload: dict) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(payload, indent=2))


def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="Directory to save raw XLSX + metadata")
    ap.add_argument("--force", action="store_true", help="Force re-download even if unchanged")
    ap.add_argument("--check-only", action="store_true", help="Check for new vintage without writing files")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    meta_path = outdir / LATEST_JSON

    logging.info("Fetching ONS landing page…")
    landing = fetch_landing(ONS_URL)
    rel_date = landing["release_date"] or dt.date.today()
    dl_url = landing["download_url"]
    logging.info(f"Release date: {rel_date} | XLSX: {dl_url}")

    # Download file
    logging.info("Downloading current XLSX…")
    blob, sha = download_bytes(dl_url)
    logging.info(f"Downloaded {len(blob)/1_000_000:.1f} MB | sha256={sha[:12]}…")

    # Compare to last metadata
    latest = read_latest(meta_path)
    unchanged = (latest is not None) and (latest.get("sha256") == sha)

    if args.check_only:
        if unchanged:
            logging.info("No new vintage detected (hash unchanged).")
            sys.exit(1)  # 1 = no change
        else:
            logging.info("New vintage detected.")
            sys.exit(0)  # 0 = new

    if unchanged and not args.force:
        logging.info("No new vintage detected (hash unchanged). Skipping save.")
        return

    # Construct versioned filename: gva_YYYYMMDD_<sha8>.xlsx
    fname = f"gva_{rel_date.strftime('%Y%m%d')}_{sha[:8]}.xlsx"
    fpath = outdir / fname
    fpath.write_bytes(blob)
    logging.info(f"Saved: {fpath}")

    # Write/update metadata JSON
    meta = {
        "dataset_id": "ons_gva_balanced_by_industry",
        "ons_url": ONS_URL,
        "download_url": dl_url,
        "release_date": rel_date.isoformat(),
        "sha256": sha,
        "filename": fname,
        "filesize_bytes": len(blob),
        "scraped_at_utc": dt.datetime.utcnow().isoformat() + "Z"
    }
    write_latest(meta_path, meta)
    logging.info(f"Updated metadata: {meta_path}")


if __name__ == "__main__":
    main()
