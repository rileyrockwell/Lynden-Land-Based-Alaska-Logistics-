import pandas as pd
import numpy as np

print("Generating shipments...")

routes = pd.read_parquet("data/raw/routes.parquet")

N = 100000

sample = routes.sample(N, replace=True)

pickup = pd.to_datetime("2024-01-01") + pd.to_timedelta(
    np.random.randint(0, 365*24, N), unit="h"
)

transit = sample["distance_miles"] / 50

delivery = pickup + pd.to_timedelta(transit, unit="h")

df = pd.DataFrame({
    "shipment_id": [f"S{i}" for i in range(N)],
    "origin_id": sample["origin_id"].values,
    "destination_id": sample["destination_id"].values,
    "pickup_time": pickup,
    "delivery_time": delivery,
    "weight_lbs": np.random.uniform(100, 50000, N),
    "volume_cuft": np.random.uniform(10, 4000, N)
})

df.to_parquet("data/raw/shipments.parquet", index=False)

print("Shipments created")