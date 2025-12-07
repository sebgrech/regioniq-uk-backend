import pandas as pd
from supabase_client import supabase
import numpy as np

# Load cleaned population
df = pd.read_csv("data/clean/population_ITL1_metrics_long.csv")

# Replace NaN, inf, -inf with None (JSON null)
df = df.replace([np.nan, np.inf, -np.inf], None)

# Convert DataFrame to list of dicts
data = df.to_dict(orient="records")

# Push to Supabase
response = supabase.table("population_ITL1_metrics_long").insert(data).execute()
print("Insert response:", response)



