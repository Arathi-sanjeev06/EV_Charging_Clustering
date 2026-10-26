
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree

# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("data/cleaned_ev_stations.csv")
df = df.dropna(
    subset=["state", "lattitude", "longitude"]
).copy()

df["state"] = df["state"].astype(str).str.strip()

# ==========================================
# 2. STANDARDIZE STATE / REGION NAMES
# ==========================================

state_mapping = {
    "Maharashra": "Maharashtra",
    "Delhi NCR": "Delhi NCR",
    "Andhra pradesh": "Andhra Pradesh",
    "Andra Pradesh": "Andhra Pradesh",
    "AndhraPradesh": "Andhra Pradesh",
    "TamilNadu": "Tamil Nadu",
    "TAMIL NADU": "Tamil Nadu",
    "TamiNadu": "Tamil Nadu",
    "TELENGANA": "Telangana",
    "Chattisgarh": "Chhattisgarh",
    "chattisgarh": "Chhattisgarh",
    "WestBengal": "West Bengal",
    "Uttrakhand": "Uttarakhand",
    "Uttarkhand": "Uttarakhand",
    "Harayana": "Haryana",
    "Karala": "Kerala",
    "Jammu": "Jammu and Kashmir",
    "Jammu & Kashmir": "Jammu and Kashmir",
    "PUNJAB": "Punjab",
    "Pondicherry": "Puducherry",
    "Andaman": "Andaman and Nicobar Islands",
    "Hyderabadu00a0": "Telangana",
    "Hyderabad": "Telangana",
    "Kochi": "Kerala",
    "Ernakulam": "Kerala",
    "Bhubhaneswar": "Odisha",
    "Jajpur": "Odisha",
    "Hisar": "Haryana",
    "Rajahmundry": "Andhra Pradesh"
}

df["original_state"] = df["state"]
df["state"] = df["state"].replace(state_mapping)

# Flag entries that look like city names or need checking.
# Chikhali and Limbdi are not automatically changed here.
review_names = [
    "Chikhali",
    "Limbdi"
]

df["review_required"] = df["state"].isin(review_names)

print("=" * 60)
print("CLEANED STATE-WISE EV CHARGING STATION ANALYSIS")
print("=" * 60)

print("\nTotal records analysed:", len(df))

# ==========================================
# 3. SHOW RECORDS NEEDING MANUAL REVIEW
# ==========================================

review_df = df[df["review_required"]]

print("\nRecords requiring manual review:")
if len(review_df) > 0:
    print(
        review_df[
            ["name", "original_state", "city",
             "lattitude", "longitude"]
        ].to_string(index=False)
    )
else:
    print("No flagged records.")

# ==========================================
# 4. COUNT STATIONS BY CLEANED STATE
# ==========================================

state_counts = (
    df.groupby("state")
    .size()
    .reset_index(name="station_count")
    .sort_values("station_count", ascending=False)
)

print("\nSTATION COUNTS BY CLEANED STATE / REGION")
print("-" * 50)
print(state_counts.to_string(index=False))

# ==========================================
# 5. CALCULATE NEAREST-NEIGHBOUR DISTANCES
# ==========================================

coords = df[["lattitude", "longitude"]].to_numpy()

tree = BallTree(
    np.radians(coords),
    metric="haversine"
)

distances, _ = tree.query(
    np.radians(coords),
    k=2
)

df["nearest_station_distance_km"] = (
    distances[:, 1] * 6371
)

# ==========================================
# 6. SUMMARIZE DISTANCES BY CLEANED STATE
# ==========================================

state_summary = (
    df.groupby("state")
    .agg(
        station_count=("name", "size"),
        mean_nearest_distance_km=(
            "nearest_station_distance_km", "mean"
        ),
        median_nearest_distance_km=(
            "nearest_station_distance_km", "median"
        ),
        maximum_nearest_distance_km=(
            "nearest_station_distance_km", "max"
        )
    )
    .reset_index()
    .sort_values(
        "station_count",
        ascending=False
    )
)

print("\nNEAREST-NEIGHBOUR DISTANCES BY CLEANED STATE / REGION")
print("-" * 75)
print(state_summary.round(2).to_string(index=False))

# ==========================================
# 7. SAVE CLEANED RESULTS
# ==========================================

df.to_csv(
    "outputs/stations_with_nearest_distances.csv",
    index=False
)

state_counts.to_csv(
    "outputs/state_station_counts.csv",
    index=False
)

state_summary.to_csv(
    "outputs/state_nearest_distance_summary.csv",
    index=False
)

# ==========================================
# 8. PLOT STATION COUNTS
# ==========================================

plot_counts = state_counts.sort_values(
    "station_count"
)

plt.figure(figsize=(11, 8))
plt.barh(
    plot_counts["state"],
    plot_counts["station_count"]
)

plt.title("Recorded EV Charging Stations by State / Region")
plt.xlabel("Number of recorded stations")
plt.ylabel("State / region")
plt.tight_layout()

plt.savefig(
    "outputs/state_station_counts.png",
    dpi=300
)
plt.show()

# ==========================================
# 9. PLOT MEDIAN DISTANCES
# ==========================================

plot_distances = state_summary.sort_values(
    "median_nearest_distance_km"
)

plt.figure(figsize=(11, 8))
plt.barh(
    plot_distances["state"],
    plot_distances["median_nearest_distance_km"]
)

plt.title("Median Nearest-Station Distance by State / Region")
plt.xlabel("Distance (km)")
plt.ylabel("State / region")
plt.tight_layout()

plt.savefig(
    "outputs/state_median_distance.png",
    dpi=300
)
plt.show()

print("\nAnalysis completed!")
print("Updated CSV files and graphs are in the outputs folder.")
