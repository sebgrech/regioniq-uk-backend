import pandas as pd
import json
from pathlib import Path

# File paths
GEO_ITL1 = Path("data/geo/ITL1_simplified_clean.geojson")
FORECAST_LONG = Path("data/forecast/forecast_v3_long.csv")

# Load forecast CSV
df = pd.read_csv(FORECAST_LONG)
# Expecting: region, region_code, metric, value

# Load GeoJSON
with open(GEO_ITL1) as f:
    geojson = json.load(f)

# Use ITL125CD as the join key
geojson_key = "ITL125CD"

# --- Join forecasts into GeoJSON ---
for feature in geojson["features"]:
    code = feature["properties"][geojson_key]
    rows = df[df["region_code"] == code]

    if not rows.empty:
        # Map metrics into {metric: value}
        metrics = rows.set_index("metric")["value"].to_dict()
        feature["properties"].update(metrics)

# --- Save merged GeoJSON ---
OUTPUT = Path("data/geo/ITL1_forecast.geojson")
with open(OUTPUT, "w") as f:
    json.dump(geojson, f)

print(f"✅ Merged GeoJSON saved to {OUTPUT}")
