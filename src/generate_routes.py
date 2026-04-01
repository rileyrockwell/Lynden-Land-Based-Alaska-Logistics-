import pandas as pd
from math import radians, sin, cos, sqrt, atan2

print("Generating routes...")

locations = pd.read_parquet("data/raw/locations.parquet")

def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

routes = []

for i, o in locations.iterrows():
    for j, d in locations.iterrows():
        if o["location_id"] != d["location_id"]:
            dist = haversine(o["latitude"], o["longitude"], d["latitude"], d["longitude"])

            routes.append({
                "route_id": f"{o['location_id']}_{d['location_id']}",
                "origin_id": o["location_id"],
                "destination_id": d["location_id"],
                "distance_miles": dist,
                "mode": "truck"
            })

df = pd.DataFrame(routes)

df.to_parquet("data/raw/routes.parquet", index=False)

print("Routes created")