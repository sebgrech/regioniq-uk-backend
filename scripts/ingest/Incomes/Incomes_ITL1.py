# scripts/ingest/gdhi_nomis_itl1.py
import os
import urllib.request
from pathlib import Path

RAW_DIR = Path("data/raw/incomes")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# NOMIS API: ITL1 regions, GDHI total + per head
URL = (
    "https://www.nomisweb.co.uk/api/v01/dataset/NM_185_1.data.csv"
    "?geography=2013265921...2013265932"   # ITL1 codes
    "&component_of_gdhi=0"                 # 0 = Total GDHI
    "&measure=1,2"                         # 1=GDHI (£m), 2=GDHI per head (£)
    "&measures=20100"                      # 20100 = value (£)
)

OUT = RAW_DIR / "gdhi_itl1_nomis.csv"
urllib.request.urlretrieve(URL, OUT.as_posix())

print(f"✅ Downloaded NOMIS GDHI raw → {OUT}")
