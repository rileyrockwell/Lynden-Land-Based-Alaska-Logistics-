import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

print("STARTING MODEL TRAINING...")

# -------------------------
# STEP 1: LOAD DATA
# -------------------------

df = pd.read_parquet("data/processed/feature_table.parquet")

print("Data loaded:", df.shape)

# -------------------------
# STEP 2: SELECT FEATURES
# -------------------------

# Only numeric + useful columns
features = [
    "distance_miles",
    "weight_lbs",
    "volume_cuft",
    "transit_time_hours",
    "fuel_price_usd_per_gallon",
    "temperature_f",
    "wind_speed_mph",
    "precipitation_inches"
]

target = "total_cost_usd"

df = df[features + [target]].dropna()

print("Filtered data:", df.shape)

# -------------------------
# STEP 3: SPLIT DATA
# -------------------------

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# STEP 4: TRAIN MODEL
# -------------------------

model = RandomForestRegressor(n_estimators=50, random_state=42)

model.fit(X_train, y_train)

print("Model trained")

# -------------------------
# STEP 5: EVALUATE
# -------------------------

preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)

print("MAE:", mae)

# -------------------------
# STEP 6: FEATURE IMPORTANCE
# -------------------------

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

# -------------------------
# STEP 7: SAVE OUTPUTS
# -------------------------

# Save results
with open("outputs/model_results.txt", "w") as f:
    f.write(f"MAE: {mae}\n")

importance.to_csv("outputs/feature_importance.csv", index=False)

print("Outputs saved")