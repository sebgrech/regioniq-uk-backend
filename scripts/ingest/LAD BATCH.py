# scripts/ingest/population_nomis_LAD_batch.py
import requests
import pandas as pd
from io import StringIO
from pathlib import Path
import math
import time

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# --- Full LAD code list (paste your set here or load from CSV) ---
lad_codes = [
"E06000001","E06000002","E06000003","E06000004","E06000005","E06000047","E06000057","E08000021","E08000022","E08000023","E08000024","E08000037","E06000006","E06000007","E06000008","E06000009","E06000049","E06000050","E06000063","E06000064","E08000001","E08000002","E08000003","E08000004","E08000005","E08000006","E08000007","E08000008","E08000009","E08000010","E08000011","E08000012","E08000013","E08000014","E08000015","E10000017","E06000010","E06000011","E06000012","E06000013","E06000014","E06000065","E08000016","E08000017","E08000018","E08000019","E08000032","E08000033","E08000034","E08000035","E08000036","E06000015","E06000016","E06000017","E06000018","E06000061","E06000062","E10000007","E10000018","E10000019","E10000024","E06000019","E06000020","E06000021","E06000051","E08000025","E08000026","E08000027","E08000028","E08000029","E08000030","E08000031","E10000028","E10000031","E10000034","E06000031","E06000032","E06000033","E06000034","E06000055","E06000056","E10000003","E10000012","E10000015","E10000020","E10000029","E09000001","E09000002","E09000003","E09000004","E09000005","E09000006","E09000007","E09000008","E09000009","E09000010","E09000011","E09000012","E09000013","E09000014","E09000015","E09000016","E09000017","E09000018","E09000019","E09000020","E09000021","E09000022","E09000023","E09000024","E09000025","E09000026","E09000027","E09000028","E09000029","E09000030","E09000031","E09000032","E09000033","E06000035","E06000036","E06000037","E06000038","E06000039","E06000040","E06000041","E06000042","E06000043","E06000044","E06000045","E06000046","E06000060","E10000011","E10000014","E10000016","E10000025","E10000030","E10000032","E06000022","E06000023","E06000024","E06000025","E06000026","E06000027","E06000030","E06000052","E06000053","E06000054","E06000058","E06000059","E06000066","E10000008","E10000013","W06000001","W06000002","W06000003","W06000004","W06000005","W06000006","W06000008","W06000009","W06000010","W06000011","W06000012","W06000013","W06000014","W06000015","W06000016","W06000018","W06000019","W06000020","W06000021","W06000022","W06000023","W06000024","S12000005","S12000006","S12000008","S12000010","S12000011","S12000013","S12000014","S12000017","S12000018","S12000019","S12000020","S12000021","S12000023","S12000026","S12000027","S12000028","S12000029","S12000030","S12000033","S12000034","S12000035","S12000036","S12000038","S12000039","S12000040","S12000041","S12000042","S12000045","S12000047","S12000048","S12000049","S12000050","N09000001","N09000002","N09000003","N09000004","N09000005","N09000006","N09000007","N09000008","N09000009","N09000010","N09000011"
]

# --- Config ---
BATCH_SIZE = 40   # keep under ~50 to avoid API failure
OUTPUT_FILE = RAW_DIR / "population_LAD_nomis.csv"

# API params (fixed)
BASE_URL = "https://www.nomisweb.co.uk/api/v01/dataset/NM_31_1.data.csv"
FIXED_PARAMS = "&sex=7&age=0,24,22,25,20,21&measures=20100"

def fetch_batch(batch_codes):
    geog_param = ",".join(batch_codes)
    url = f"{BASE_URL}?geography={geog_param}{FIXED_PARAMS}"
    r = requests.get(url)
    r.raise_for_status()
    # NOMIS sometimes sends TSV, sometimes CSV
    try:
        df = pd.read_csv(StringIO(r.text))
    except pd.errors.ParserError:
        df = pd.read_csv(StringIO(r.text), sep="\t")
    return df

def main():
    n_batches = math.ceil(len(lad_codes) / BATCH_SIZE)
    print(f"📍 Fetching {len(lad_codes)} LADs in {n_batches} batches...")

    all_dfs = []
    for i in range(n_batches):
        batch = lad_codes[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
        print(f"   → Batch {i+1}/{n_batches} ({len(batch)} LADs)")
        try:
            df = fetch_batch(batch)
            all_dfs.append(df)
            print(f"     ✅ Success, {len(df)} rows")
        except Exception as e:
            print(f"     ❌ Failed batch {i+1}: {e}")
        time.sleep(1)  # polite to NOMIS server

    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
        full_df.to_csv(OUTPUT_FILE, index=False)
        print(f"🎉 Done! Saved {len(full_df)} rows → {OUTPUT_FILE}")
    else:
        print("⚠️ No data retrieved!")

if __name__ == "__main__":
    main()
