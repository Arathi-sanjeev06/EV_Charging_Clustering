
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.neighbors import BallTree
from scipy.spatial import Delaunay

# ------------------------------------------------
# PATHS AND SETTINGS
# ------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

INPUT_FILE = DATA_DIR / "cleaned_ev_stations_revised.csv"

EARTH_RADIUS_KM = 6371.0088
DISTANCE_THRESHOLD_KM = 25
GRID_SIZE = 120

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("Stations loaded:", len(df))

lat = df["latitude"].to_numpy()
lon = df["longitude"].to_numpy()

# BallTree with Haversine requires radians
coords_rad = np.radians(np.column_stack((lat, lon)))

tree = BallTree(coords_rad, metric="haversine")

# ------------------------------------------------
# 1. NEAREST OTHER CHARGING STATION
# ------------------------------------------------

distances, indices = tree.query(coords_rad, k=2)

# First result is the station itself.
nearest_distance_km = distances[:, 1] * EARTH_RADIUS_KM
nearest_index = indices[:, 1]

df["nearest_station_distance_km"] = nearest_distance_km
df["nearest_station_name"] = (
    df.iloc[nearest_index]["name"].to_numpy()
)
df["nearest_station_state"] = (
    df.iloc[nearest_index]["state"].to_numpy()
)

df.to_csv(
    OUTPUT_DIR / "station_nearest_distances_revised.csv",
    index=False
)

# Overall station nearest-neighbour summary
station_summary = pd.DataFrame([{
    "Total_Stations": len(df),
    "Mean_Nearest_Distance_KM": np.mean(nearest_distance_km),
    "Median_Nearest_Distance_KM": np.median(nearest_distance_km),
    "Maximum_Nearest_Distance_KM": np.max(nearest_distance_km),
    "Stations_Over_25_KM": int(
        np.sum(nearest_distance_km > DISTANCE_THRESHOLD_KM)
    )
}])

station_summary.to_csv(
    OUTPUT_DIR / "station_distance_summary_revised.csv",
    index=False
)

# State-wise summary: see state_coverage_summary_final.py (uses cleaned state labels)

# ------------------------------------------------
# 3. GRID-BASED COVERAGE GAP ESTIMATE
# Grid is restricted to the convex hull of stations.
# It is NOT a representation of all Indian land area.
# ------------------------------------------------

lat_values = np.linspace(lat.min(), lat.max(), GRID_SIZE)
lon_values = np.linspace(lon.min(), lon.max(), GRID_SIZE)

grid_lon, grid_lat = np.meshgrid(lon_values, lat_values)

grid_points = np.column_stack((
    grid_lon.ravel(),
    grid_lat.ravel()
))

# Delaunay triangulation in longitude-latitude space
station_points_2d = np.column_stack((lon, lat))
hull = Delaunay(station_points_2d)

inside_hull = hull.find_simplex(grid_points) >= 0
grid_points = grid_points[inside_hull]

# Convert grid points to radians, ordered latitude then longitude
grid_rad = np.radians(
    np.column_stack((
        grid_points[:, 1],
        grid_points[:, 0]
    ))
)

grid_distances, _ = tree.query(grid_rad, k=1)

grid_nearest_km = grid_distances[:, 0] * EARTH_RADIUS_KM

beyond_threshold = (
    grid_nearest_km > DISTANCE_THRESHOLD_KM
)

total_grid_points = len(grid_nearest_km)
gap_points = int(np.sum(beyond_threshold))

gap_percentage = (
    gap_points / total_grid_points * 100
    if total_grid_points else 0
)

# Save grid-level results
grid_output = pd.DataFrame({
    "longitude": grid_points[:, 0],
    "latitude": grid_points[:, 1],
    "nearest_station_distance_km": grid_nearest_km,
    "beyond_25_km": beyond_threshold
})

grid_output.to_csv(
    OUTPUT_DIR / "coverage_gap_grid_revised.csv",
    index=False
)

# Summary
coverage_summary = pd.DataFrame([{
    "Grid_Points_Evaluated": total_grid_points,
    "Grid_Points_Beyond_25_KM": gap_points,
    "Percentage_Beyond_25_KM": gap_percentage,
    "Median_Nearest_Distance_KM": np.median(grid_nearest_km),
    "Maximum_Nearest_Distance_KM": np.max(grid_nearest_km),
    "Threshold_KM": DISTANCE_THRESHOLD_KM,
    "Grid_Size_Per_Axis": GRID_SIZE
}])

coverage_summary.to_csv(
    OUTPUT_DIR / "coverage_gap_summary_revised.csv",
    index=False
)

# ------------------------------------------------
# 4. PLOT COVERAGE GAP MAP
# ------------------------------------------------

plt.figure(figsize=(10, 7))

plt.scatter(
    grid_points[:, 0],
    grid_points[:, 1],
    c=grid_nearest_km,
    cmap="YlOrRd",
    s=5,
    alpha=0.7
)

plt.scatter(
    lon,
    lat,
    color="blue",
    s=8,
    label="EV charging stations"
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(
    "Estimated Distance to Nearest EV Charging Station"
)
plt.colorbar(
    label="Nearest station distance (km)"
)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "coverage_gap_map_revised.png",
    dpi=300
)

plt.close()

# ------------------------------------------------
# 5. PRINT RESULTS
# ------------------------------------------------

print("\n========== STATION DISTANCE SUMMARY ==========")
print(station_summary.to_string(index=False))


print("\n========== GRID COVERAGE SUMMARY ==========")
print(coverage_summary.to_string(index=False))

print("\nAll revised coverage outputs saved in:")
print(OUTPUT_DIR)

print("\nNOTE: Grid percentage applies only to sampled")
print("points inside the convex hull of station locations.")
print("It does not represent percentage of India's land area.")

print("\n========== ANALYSIS COMPLETED ==========")
