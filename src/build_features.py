import pandas as pd

print("STARTING FEATURE BUILD...")

# -------------------------
# STEP 1: LOAD DATA
# -------------------------

shipments = pd.read_parquet("data/raw/shipments.parquet")
routes = pd.read_parquet("data/raw/routes.parquet")
costs = pd.read_parquet("data/raw/costs.parquet")
fuel = pd.read_parquet("data/raw/fuel.parquet")
weather = pd.read_parquet("data/raw/weather.parquet")

print("All datasets loaded")

# -------------------------
# STEP 2: BASIC CLEANING
# -------------------------

# Ensure datetime
shipments["pickup_time"] = pd.to_datetime(shipments["pickup_time"])
shipments["delivery_time"] = pd.to_datetime(shipments["delivery_time"])

fuel["date"] = pd.to_datetime(fuel["date"])
weather["date"] = pd.to_datetime(weather["date"])

# -------------------------
# STEP 3: JOIN CORE TABLES
# -------------------------

df = shipments.merge(
    routes,
    on=["origin_id", "destination_id"],
    how="left"
)

df = df.merge(
    costs,
    on=["shipment_id", "route_id"],
    how="left"
)

print("DEBUG 1 — After core joins:", df.shape)

print("Core joins complete")

# -------------------------
# STEP 4: FEATURE CREATION
# -------------------------

# Transit time (hours)
df["transit_time_hours"] = (
    (df["delivery_time"] - df["pickup_time"]).dt.total_seconds() / 3600
)

# Cost efficiency
df["cost_per_mile"] = df["total_cost_usd"] / df["distance_miles"]

# Weight density
df["weight_per_cuft"] = df["weight_lbs"] / df["volume_cuft"]

# -------------------------
# STEP 5: ADD FUEL (APPROX JOIN)
# -------------------------

# Align by nearest date
df["pickup_date"] = df["pickup_time"].dt.date
fuel["date"] = fuel["date"].dt.date

df = df.merge(
    fuel,
    left_on="pickup_date",
    right_on="date",
    how="left"
)

print("DEBUG 2 — After fuel join:", df.shape)

df = df.drop(columns=["date"])

print("Fuel joined")

# -------------------------
# STEP 6: ADD WEATHER (SIMPLE JOIN)
# -------------------------

weather["date"] = weather["date"].dt.date

df = df.merge(
    weather,
    left_on=["origin_id", "pickup_date"],
    right_on=["location_id", "date"],
    how="left"
)

print("DEBUG 3 — After weather join:", df.shape)

df = df.drop(columns=["location_id", "date"])

print("Weather joined")

# -------------------------
# STEP 7: FINAL CLEANING
# -------------------------

# Drop rows with critical nulls
df = df.dropna(subset=[
    "distance_miles",
    "total_cost_usd",
    "transit_time_hours"
])

print("DEBUG 4 — After cleaning:", df.shape)

# Replace infinities
df = df.replace([float("inf"), -float("inf")], None)

# Only drop rows missing CRITICAL fields
df = df.dropna(subset=[
    "distance_miles",
    "total_cost_usd",
    "transit_time_hours"
])

print("Rows after cleaning:", df.shape)

print("Final cleaning complete")

# -------------------------
# STEP 8: SAVE OUTPUT
# -------------------------

output_path = "data/processed/feature_table.parquet"
df.to_parquet(output_path, index=False)

print(f"FEATURE TABLE SAVED: {output_path}")
print(f"FINAL SHAPE: {df.shape}")