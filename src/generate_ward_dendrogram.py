
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR, "outputs", "stations_state_standardized.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR, "outputs", "ward_dendrogram.png"
)

# Load revised dataset
df = pd.read_csv(INPUT_FILE)

df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

print("Stations loaded:", len(df))

# Convert coordinates to 3D unit sphere
lat_rad = np.radians(df["latitude"].to_numpy())
lon_rad = np.radians(df["longitude"].to_numpy())

X = np.column_stack([
    np.cos(lat_rad) * np.cos(lon_rad),
    np.cos(lat_rad) * np.sin(lon_rad),
    np.sin(lat_rad)
])

# Perform Ward hierarchical clustering
print("Calculating Ward linkage...")

Z = linkage(X, method="ward")

# Calculate a cut height corresponding to 5 clusters
cut_height = (Z[-5, 2] + Z[-4, 2]) / 2   # 4 merges above the line -> 5 clusters

# Draw readable truncated dendrogram
plt.figure(figsize=(16, 9))

dendrogram(
    Z,
    truncate_mode="lastp",
    p=30,
    show_contracted=True,
    color_threshold=cut_height
)

plt.axhline(
    y=cut_height,
    linestyle="--",
    label=f"Cut for 5 clusters (height {cut_height:.2f})"
)

plt.title("Ward Hierarchical Clustering Dendrogram (K=5)")
plt.xlabel("Clusters / Merged Station Groups")
plt.ylabel("Ward Linkage Distance")
plt.legend()
plt.tight_layout()

plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
plt.close()

print("Dendrogram saved:", OUTPUT_FILE)
print("Total stations:", len(df))
print("Dendrogram generated successfully.")
