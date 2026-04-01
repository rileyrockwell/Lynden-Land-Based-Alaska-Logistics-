import pandas as pd
import numpy as np

print("Generating fuel dataset...")

# Step 1: Create dates
dates = pd.date_range(start="2023-01-01", periods=1000, freq="D")

# Step 2: Generate realistic fuel prices
np.random.seed(42)
prices = np.random.uniform(2.5, 5.0, len(dates))

# Step 3: Build dataframe
df = pd.DataFrame({
    "date": dates,
    "fuel_price_usd_per_gallon": prices
})

# Step 4: Save
df.to_parquet("data/raw/fuel.parquet", index=False)

print("Fuel dataset created successfully")