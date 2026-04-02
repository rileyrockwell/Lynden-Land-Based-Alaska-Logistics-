import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

print("STARTING MODEL TRAINING...")

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_parquet("data/processed/feature_table.parquet")

print("LOADED SHAPE:", df.shape)

# HARD STOP IF EMPTY
assert df.shape[0] > 0, "Dataset is empty"

# -------------------------
# DEFINE TARGET + FEATURES
# -------------------------
TARGET = "total_cost_usd"

NUMERIC_FEATURES = [
    "weight_lbs",
    "volume_cuft",
    "distance_miles",
    "fuel_cost_usd",
    "distance_cost_usd",
    "transit_time_hours",
    "cost_per_mile",
    "weight_per_cuft",
    "fuel_price_usd_per_gallon",
    "temperature_f",
    "wind_speed_mph",
    "precipitation_inches",
]

# -------------------------
# SELECT DATA
# -------------------------
df = df[NUMERIC_FEATURES + [TARGET]].copy()

print("AFTER COLUMN SELECT:", df.shape)

# -------------------------
# CLEAN (DO NOT DROP ROWS)
# -------------------------
print("NULL COUNTS:\n", df.isnull().sum())

df = df.fillna(0)

print("AFTER CLEANING:", df.shape)

# -------------------------
# SPLIT
# -------------------------
X = df[NUMERIC_FEATURES]
y = df[TARGET]

print("X SHAPE:", X.shape)
print("y SHAPE:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("TRAIN SHAPE:", X_train.shape)
print("TEST SHAPE:", X_test.shape)

# -------------------------
# TRAIN
# -------------------------
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("MODEL TRAINED")

# -------------------------
# EVALUATE
# -------------------------
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

print("MAE:", mae)

# -------------------------
# FEATURE IMPORTANCE
# -------------------------
importance = pd.DataFrame({
    "feature": NUMERIC_FEATURES,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("FEATURE IMPORTANCE:\n", importance)

# -------------------------
# SAVE OUTPUTS
# -------------------------
os.makedirs("outputs", exist_ok=True)

importance.to_csv("outputs/feature_importance.csv", index=False)

with open("outputs/model_results.txt", "w") as f:
    f.write(f"MAE: {mae}\n")

print("OUTPUTS SAVED")