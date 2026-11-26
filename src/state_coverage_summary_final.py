
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from pathlib import Path

# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "outputs" / "stations_state_standardized.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load standardised dataset
df = pd.read_csv(INPUT_FILE)

# Prepare coordinates
coordinates = df[["latitude", "longitude"]].to_numpy()
coordinates_rad = np.radians(coordinates)

# Haversine nearest-neighbour calculation
earth_radius_km = 6371.0088

tree = BallTree(coordinates_rad, metric="haversine")
distances, indices = tree.query(coordinates_rad, k=2)

# Second result is the nearest OTHER station
df["nearest_station_km"] = distances[:, 1] * earth_radius_km

# Calculate state-wise summary
summary = (
    df.groupby("state_cleaned")
    .agg(
        station_count=("nearest_station_km", "count"),
        mean_distance_km=("nearest_station_km", "mean"),
        median_distance_km=("nearest_station_km", "median"),
        maximum_distance_km=("nearest_station_km", "max"),
        stations_over_25km=("nearest_station_km",
                            lambda x: (x > 25).sum()),
    )
    .reset_index()
)

summary = summary.sort_values(
    "station_count", ascending=False
)

# Save outputs
df.to_csv(
    OUTPUT_DIR / "stations_with_nearest_distances_final.csv",
    index=False
)

summary.to_csv(
    OUTPUT_DIR / "state_coverage_summary_final.csv",
    index=False
)

# Print results
print("\n========== FINAL STATE COVERAGE SUMMARY ==========")
print("Total stations:", len(df))
print("Number of standardised labels:", df["state_cleaned"].nunique())

print("\nState-wise coverage statistics:")
print(summary.round(2).to_string(index=False))

print("\nStations requiring label review:")
print(
    df[df["state_mapping_status"] == "review_required"][
        ["state_original", "city", "name"]
    ].to_string(index=False)
)

print("\nSaved:")
print("outputs/stations_with_nearest_distances_final.csv")
print("outputs/state_coverage_summary_final.csv")
print("==================================================")
