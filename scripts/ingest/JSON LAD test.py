# scripts/utils/fetch_lad_codes.py
import requests
import pandas as pd

# NOMIS geography definition for NM_31_1
url = "https://www.nomisweb.co.uk/api/v01/dataset/NM_31_1.geography.def.sdmx.json"

r = requests.get(url)
data = r.json()

# Walk through JSON → grab all codes
codes = data["structure"]["codelists"]["codelist"][0]["code"]

# Extract into DataFrame
rows = []
for c in codes:
    code_id = str(c["value"])
    desc = c["description"]["value"]
    geog_code = None
    for ann in c.get("annotations", {}).get("annotation", []):
        if ann["annotationtitle"] == "GeogCode":
            geog_code = ann["annotationtext"]
    rows.append({"id": code_id, "name": desc, "geogcode": geog_code})

df = pd.DataFrame(rows)

# 🔑 Filter for LADs: usually descriptions containing "district", "unitary", "borough", "city"
lad_df = df[df["name"].str.contains("district|unitary|borough|city", case=False, na=False)]

print(f"Found {len(lad_df)} LAD codes")
print(lad_df.head())

# Save to CSV for reuse
lad_df.to_csv("data/lad_codes.csv", index=False)
