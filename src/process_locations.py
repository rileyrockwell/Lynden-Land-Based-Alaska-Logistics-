import pandas as pd
import numpy as np

print("Generating locations...")

np.random.seed(42)

# Step 1: create 100 locations
n = 100

df = pd.DataFrame({
    "location_id": [f"L{i}" for i in range(n)],
    "latitude": np.random.uniform(25, 65, n),
    "longitude": np.random.uniform(-125, -65, n)
})

# Step 2: save
df.to_parquet("data/raw/locations.parquet", index=False)

print("Locations created")