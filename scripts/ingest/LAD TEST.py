# scripts/ingest/population_nomis_LAD.py
import requests
import pandas as pd
from io import StringIO
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# --- Step 1: pull the geography list ---
geo_url = "https://www.nomisweb.co.uk/api/v01/dataset/NM_31_1.geography.csv"
resp = requests.get(geo_url)
resp.raise_for_status()

# NOMIS sometimes serves tab-delimited "CSV"
try:
    geo_df = pd.read_csv(StringIO(resp.text))
except pd.errors.ParserError:
    geo_df = pd.read_csv(StringIO(resp.text), sep="\t")

# Filter: keep Local Authority Districts (district/unitary/metropolitan)
lad_df = geo_df[geo_df["geogdesc"].str.contains("district|unitary|metropolitan", 
                                               case=False, na=False)]

lad_codes = lad_df["geogcode"].astype(str).tolist()

print(f"📍 Found {len(lad_codes)} LAD codes")

# --- Step 2: fetch population data for these LADs ---
# (NOMIS won’t allow thousands of codes in one URL; normally you’d batch)
geog_param = ",".join(lad_codes[:50])   # <- demo with first 50; batch if needed

data_url = (
    "https://www.nomisweb.co.uk/api/v01/dataset/NM_31_1.data.csv"
    f"?geography={geog_param}"
    "&sex=7"
    "&age=0,24,22,25,20,21"
    "&measures=20100"
)

OUT = RAW_DIR / "population_LAD_nomis.csv"
r = requests.get(data_url)
r.raise_for_status()
OUT.write_text(r.text, encoding="utf-8")

print(f"✅ Downloaded LAD population data → {OUT}")
