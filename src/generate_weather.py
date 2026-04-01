import pandas as pd
import numpy as np

print("Generating weather...")

np.random.seed(42)

locations = [f"L{i}" for i in range(100)]
dates = pd.date_range("2023-01-01", periods=365)

rows = []

for loc in locations:
    for date in dates:
        rows.append({
            "date": date,
            "location_id": loc,
            "temperature_f": np.random.uniform(-20, 100),
            "wind_speed_mph": np.random.uniform(0, 40),
            "precipitation_inches": max(0, np.random.normal(0.1, 0.2))
        })

df = pd.DataFrame(rows)

df.to_parquet("data/raw/weather.parquet", index=False)

print("Weather created")