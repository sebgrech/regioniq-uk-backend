import pandas as pd

lad_codes = pd.read_csv("/Users/gerardgrech/regioniq-uk/data/lookup/lad_codes.csv")["geogcode"].astype(str).tolist()
print(",".join([f'"{code}"' for code in lad_codes]))


