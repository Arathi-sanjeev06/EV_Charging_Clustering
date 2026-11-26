
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from itertools import combinations
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score

# ------------------------------------------------
# PATHS
# ------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

INPUT_FILE = DATA_DIR / "cleaned_ev_stations_revised.csv"

df = pd.read_csv(INPUT_FILE)

latitude = np.radians(df["latitude"].to_numpy())
longitude = np.radians(df["longitude"].to_numpy())

# ------------------------------------------------
# CONVERT LATITUDE/LONGITUDE TO 3D SPHERICAL COORDINATES
# ------------------------------------------------

x = np.cos(latitude) * np.cos(longitude)
y = np.cos(latitude) * np.sin(longitude)
z = np.sin(latitude)

coordinates = np.column_stack((x, y, z))

print("Stations loaded:", len(df))
print("Coordinate representation: 3D unit sphere")

# ------------------------------------------------
# CLUSTERING: K-MEANS AND WARD
# ------------------------------------------------

results = []
kmeans_labels = {}
ward_labels = {}

for k in range(2, 11):

    print("\nTesting K =", k)

    # K-Means
    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    km_labels = km.fit_predict(coordinates)

    km_silhouette = silhouette_score(
        coordinates, km_labels
    )

    kmeans_labels[k] = km_labels

    # Ward hierarchical clustering
    ward = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward"
    )

    w_labels = ward.fit_predict(coordinates)

    ward_silhouette = silhouette_score(
        coordinates, w_labels
    )

    ward_labels[k] = w_labels

    results.append({
        "K": k,
        "KMeans_Silhouette": km_silhouette,
        "KMeans_Inertia": km.inertia_,
        "Ward_Silhouette": ward_silhouette
    })

    print("K-Means silhouette:", round(km_silhouette, 4))
    print("Ward silhouette:", round(ward_silhouette, 4))

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "geographic_clustering_comparison.csv",
    index=False
)

# ------------------------------------------------
# SELECT BEST K FOR EACH METHOD
# ------------------------------------------------

best_kmeans_row = results_df.loc[
    results_df["KMeans_Silhouette"].idxmax()
]

best_ward_row = results_df.loc[
    results_df["Ward_Silhouette"].idxmax()
]

best_kmeans_k = int(best_kmeans_row["K"])
best_ward_k = int(best_ward_row["K"])

print("\n========== BEST RESULTS ==========")
print("Best K-Means K:", best_kmeans_k)
print("Best K-Means silhouette:",
      round(best_kmeans_row["KMeans_Silhouette"], 4))

print("Best Ward K:", best_ward_k)
print("Best Ward silhouette:",
      round(best_ward_row["Ward_Silhouette"], 4))

# ------------------------------------------------
# K-MEANS STABILITY TEST AT K=4
# ------------------------------------------------

stability_labels = []

for seed in range(20):

    model = KMeans(
        n_clusters=4,
        random_state=seed,
        n_init=20
    )

    labels = model.fit_predict(coordinates)
    stability_labels.append(labels)

ari_scores = []

for a, b in combinations(stability_labels, 2):
    ari_scores.append(adjusted_rand_score(a, b))

print("\n========== STABILITY TEST ==========")
print("Number of random seeds: 20")
print("Pairwise ARI mean:", round(np.mean(ari_scores), 4))
print("Pairwise ARI standard deviation:",
      round(np.std(ari_scores), 4))

# ------------------------------------------------
# RANDOM LABEL BASELINE AT K=4
# Preserve the observed cluster sizes
# ------------------------------------------------

reference_labels = stability_labels[0]
rng = np.random.default_rng(42)

baseline_scores = []

for i in range(30):

    random_labels = rng.permutation(reference_labels)

    score = silhouette_score(
        coordinates,
        random_labels
    )

    baseline_scores.append(score)

print("\n========== RANDOM BASELINE ==========")
print("Random baseline mean silhouette:",
      round(np.mean(baseline_scores), 4))
print("Random baseline standard deviation:",
      round(np.std(baseline_scores), 4))

# ------------------------------------------------
# SAVE K-MEANS AND WARD CLUSTER ASSIGNMENTS
# Use K=4 for direct method comparison
# ------------------------------------------------

comparison_df = df.copy()

comparison_df["KMeans_Cluster_K4"] = kmeans_labels[4]
comparison_df["Ward_Cluster_K4"] = ward_labels[4]

comparison_df.to_csv(
    OUTPUT_DIR / "geographic_cluster_assignments.csv",
    index=False
)

# ------------------------------------------------
# CLUSTER SIZES AND CENTROIDS
# ------------------------------------------------

def save_cluster_details(labels, method):

    detail_rows = []

    for cluster in sorted(np.unique(labels)):

        points = coordinates[labels == cluster]

        center = points.mean(axis=0)
        center = center / np.linalg.norm(center)

        center_lat = np.degrees(
            np.arctan2(
                center[2],
                np.sqrt(center[0]**2 + center[1]**2)
            )
        )

        center_lon = np.degrees(
            np.arctan2(center[1], center[0])
        )

        detail_rows.append({
            "Method": method,
            "Cluster": int(cluster),
            "Station_Count": int(np.sum(labels == cluster)),
            "Centroid_Latitude": center_lat,
            "Centroid_Longitude": center_lon
        })

    return detail_rows


details = []

details += save_cluster_details(
    kmeans_labels[4], "K-Means"
)

details += save_cluster_details(
    ward_labels[4], "Ward"
)

pd.DataFrame(details).to_csv(
    OUTPUT_DIR / "geographic_cluster_details.csv",
    index=False
)

# ------------------------------------------------
# PLOT SILHOUETTE COMPARISON
# ------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    results_df["K"],
    results_df["KMeans_Silhouette"],
    marker="o",
    label="K-Means"
)

plt.plot(
    results_df["K"],
    results_df["Ward_Silhouette"],
    marker="s",
    label="Ward"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Geographic Clustering Method Comparison")
plt.xticks(range(2, 11))
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "geographic_silhouette_comparison.png",
    dpi=300
)

plt.close()

# ------------------------------------------------
# PLOT K-MEANS CLUSTERS AT K=4
# ------------------------------------------------

plt.figure(figsize=(9, 6))

plt.scatter(
    df["longitude"],
    df["latitude"],
    c=kmeans_labels[4],
    cmap="tab10",
    s=12
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("K-Means Geographic Clusters (K=4)")
plt.colorbar(label="Cluster")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "geographic_kmeans_clusters.png",
    dpi=300
)

plt.close()

print("\nAll revised clustering outputs saved in:")
print(OUTPUT_DIR)
print("\n========== ANALYSIS COMPLETED ==========")
