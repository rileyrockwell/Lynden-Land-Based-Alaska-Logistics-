import pandas as pd
import numpy as np

print("Generating costs...")

shipments = pd.read_parquet("data/raw/shipments.parquet")
routes = pd.read_parquet("data/raw/routes.parquet")
fuel = pd.read_parquet("data/raw/fuel.parquet")

df = shipments.merge(
    routes,
    on=["origin_id", "destination_id"],
    how="left"
)

fuel_sample = fuel["fuel_price_usd_per_gallon"].sample(len(df), replace=True).values

df["distance_cost_usd"] = df["distance_miles"] * 1.5
df["fuel_cost_usd"] = df["distance_miles"] * 0.05 * fuel_sample
df["total_cost_usd"] = 50 + df["distance_cost_usd"] + df["fuel_cost_usd"]

out = df[[
    "shipment_id",
    "route_id",
    "total_cost_usd",
    "fuel_cost_usd",
    "distance_cost_usd"
]]

out.to_parquet("data/raw/costs.parquet", index=False)

print("Costs created")