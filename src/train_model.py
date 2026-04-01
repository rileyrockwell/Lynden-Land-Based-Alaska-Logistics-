import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

print("STARTING MODEL TRAINING...")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_parquet("data/processed/feature_table.parquet")

print("INITIAL SHAPE:", df.shape)

# -------------------------
# SAFE FEATURE SET (GUARANTEED TO EXIST)
# -------------------------
features = [
    "distance_miles",
    "weight_lbs",
    "volume_cuft",
    "transit_time_hours"
]

target = "total_cost_usd"

df = df[features + [target]].copy()

print("AFTER COLUMN SELECT:", df.shape)

# -------------------------
# FINAL CLEANING (MINIMAL)
# -------------------------
df = df.dropna()

print("AFTER CLEANING:", df.shape)

# -------------------------
# SPLIT
# -------------------------
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# TRAIN
# -------------------------
model = RandomForestRegressor(n_estimators=50, random_state=42)
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
    "feature": features,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

# -------------------------
# SAVE OUTPUTS
# -------------------------
importance.to_csv("outputs/feature_importance.csv", index=False)

with open("outputs/model_results.txt", "w") as f:
    f.write(f"MAE: {mae}\n")

print("OUTPUTS SAVED")