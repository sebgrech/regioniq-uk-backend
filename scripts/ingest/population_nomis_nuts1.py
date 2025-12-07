# scripts/ingest/population_nomis_nuts1.py
import os
import urllib.request
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# NOMIS API: NUTS1, persons, selected age bands, value measure
URL = (
    "https://www.nomisweb.co.uk/api/v01/dataset/NM_31_1.data.csv"
    "?geography=2013265921...2013265932"
    "&sex=7"
    "&age=0,24,22,25,20,21"
    "&measures=20100"
)

OUT = RAW_DIR / "population_nuts1_nomis.csv"
urllib.request.urlretrieve(URL, OUT.as_posix())

print(f"✅ Downloaded NOMIS raw → {OUT}")
