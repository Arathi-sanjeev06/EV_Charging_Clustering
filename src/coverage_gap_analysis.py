
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
from scipy.spatial import Delaunay

# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("data/cleaned_ev_stations.csv")
df = df.dropna(subset=["lattitude", "longitude"]).copy()

print("=" * 50)
print("IMPROVED EV COVERAGE GAP ANALYSIS")
print("=" * 50)
print("Charging stations:", len(df))

# ==========================================
# 2. BUILD GEOGRAPHICAL DISTANCE MODEL
# ==========================================

coordinates = df[["lattitude", "longitude"]].to_numpy()
station_radians = np.radians(coordinates)

earth_radius_km = 6371

tree = BallTree(station_radians, metric="haversine")

# Distance to the nearest OTHER station
distances, _ = tree.query(station_radians, k=2)

df["nearest_station_distance_km"] = (
    distances[:, 1] * earth_radius_km
)

df.to_csv(
    "outputs/station_nearest_distance.csv",
    index=False
)

print("\nNearest-other-station distance statistics:")
print(df["nearest_station_distance_km"].describe().round(2))

# Show the 10 stations with the largest nearest-station distances
print("\n10 stations with the largest nearest-station distances:")
print(
    df.nlargest(10, "nearest_station_distance_km")[
        ["name", "state", "city", "lattitude",
         "longitude", "nearest_station_distance_km"]
    ].to_string(index=False)
)

# ==========================================
# 3. CREATE CANDIDATE GRID
# ==========================================

grid_size = 120

latitudes = np.linspace(
    df["lattitude"].min(),
    df["lattitude"].max(),
    grid_size
)

longitudes = np.linspace(
    df["longitude"].min(),
    df["longitude"].max(),
    grid_size
)

longitude_grid, latitude_grid = np.meshgrid(
    longitudes, latitudes
)

grid_points = np.column_stack((
    latitude_grid.ravel(),
    longitude_grid.ravel()
))

# ==========================================
# 4. KEEP GRID POINTS INSIDE STATION CONVEX HULL
# ==========================================

# This removes points outside the convex hull of the stations.
# It does NOT determine whether a location is inhabited or accessible.

hull = Delaunay(coordinates)
inside_hull = hull.find_simplex(grid_points) >= 0

grid_points = grid_points[inside_hull]

print("\nGrid points inside the station convex hull:", len(grid_points))

# ==========================================
# 5. CALCULATE NEAREST-STATION DISTANCES
# ==========================================

grid_radians = np.radians(grid_points)

grid_distances, _ = tree.query(grid_radians, k=1)

nearest_km = grid_distances[:, 0] * earth_radius_km

grid_df = pd.DataFrame({
    "latitude": grid_points[:, 0],
    "longitude": grid_points[:, 1],
    "nearest_station_distance_km": nearest_km
})

# Exploratory threshold; not an official coverage standard
threshold_km = 25

grid_df["potential_gap"] = (
    grid_df["nearest_station_distance_km"] > threshold_km
)

grid_df.to_csv(
    "outputs/coverage_gap_grid.csv",
    index=False
)

# ==========================================
# 6. PRINT RESULTS
# ==========================================

gap_count = int(grid_df["potential_gap"].sum())
total_points = len(grid_df)

print("\nCOVERAGE GAP RESULTS")
print("-" * 40)
print("Grid points evaluated:", total_points)
print("Grid points beyond 25 km:", gap_count)

if total_points > 0:
    print(
        "Percentage beyond threshold:",
        round(100 * gap_count / total_points, 2),
        "%"
    )

print(
    "Median grid-point distance (km):",
    round(grid_df["nearest_station_distance_km"].median(), 2)
)

print(
    "Maximum grid-point distance (km):",
    round(grid_df["nearest_station_distance_km"].max(), 2)
)

# ==========================================
# 7. PLOT COVERAGE DISTANCES
# ==========================================

plt.figure(figsize=(11, 8))

scatter = plt.scatter(
    grid_df["longitude"],
    grid_df["latitude"],
    c=grid_df["nearest_station_distance_km"],
    cmap="YlOrRd",
    s=12,
    alpha=0.7
)

plt.scatter(
    df["longitude"],
    df["lattitude"],
    color="black",
    s=5,
    label="Recorded charging stations"
)

plt.colorbar(
    scatter,
    label="Distance to nearest station (km)"
)

plt.title("Potential EV Charging Coverage Gaps")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.grid(True, alpha=0.2)
plt.tight_layout()

plt.savefig(
    "outputs/coverage_gap_map.png",
    dpi=300
)

plt.show()

# ==========================================
# 8. SAVE SUMMARY
# ==========================================

summary = pd.DataFrame({
    "Metric": [
        "Number of charging stations",
        "Mean nearest-other-station distance (km)",
        "Median nearest-other-station distance (km)",
        "Maximum nearest-other-station distance (km)",
        "Grid points inside station convex hull",
        "Grid points beyond 25 km",
        "Percentage of evaluated grid points beyond 25 km",
        "Distance threshold (km)"
    ],
    "Value": [
        len(df),
        df["nearest_station_distance_km"].mean(),
        df["nearest_station_distance_km"].median(),
        df["nearest_station_distance_km"].max(),
        total_points,
        gap_count,
        round(100 * gap_count / total_points, 2)
        if total_points > 0 else 0,
        threshold_km
    ]
})

summary.to_csv(
    "outputs/coverage_gap_summary.csv",
    index=False
)

print("\nUpdated files saved in outputs:")
print("- station_nearest_distance.csv")
print("- coverage_gap_grid.csv")
print("- coverage_gap_map.png")
print("- coverage_gap_summary.csv")
print("\nAnalysis completed!")
