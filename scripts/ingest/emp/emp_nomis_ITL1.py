# scripts/ingest/emp/emp_itl1_nomis.py
import os
import urllib.request
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw/emp")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# NOMIS API calls
URL_2009_15 = (
    "https://www.nomisweb.co.uk/api/v01/dataset/NM_172_1.data.csv"
    "?geography=2013265921...2013265931"  # ITL1 regions
    "&industry=37748736"                  # All industries
    "&employment_status=1"                # Total employment
    "&measure=1"                          # Jobs count
    "&measures=20100"                     # Value
)

URL_2015_23 = (
    "https://www.nomisweb.co.uk/api/v01/dataset/NM_189_1.data.csv"
    "?geography=2013265921...2013265931"
    "&industry=37748736"
    "&employment_status=1"
    "&measure=1"
    "&measures=20100"
)

OUT = RAW_DIR / "emp_itl1_nomis.csv"

def fetch_csv(url):
    """Download CSV from NOMIS into pandas DataFrame"""
    return pd.read_csv(url)

def main():
    print("⬇️ Fetching 2009–2015…")
    df1 = fetch_csv(URL_2009_15)
    print(f"   Got {df1.shape[0]} rows")

    print("⬇️ Fetching 2015–2023…")
    df2 = fetch_csv(URL_2015_23)
    print(f"   Got {df2.shape[0]} rows")

    # Concatenate (outer join not needed since schema matches)
    df = pd.concat([df1, df2], ignore_index=True)

    # Sort by geography + date if present
    sort_cols = [c for c in ["GEOGRAPHY_NAME", "DATE"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    # Save
    df.to_csv(OUT, index=False)
    print(f"✅ Saved merged employment dataset → {OUT}")

if __name__ == "__main__":
    main()

