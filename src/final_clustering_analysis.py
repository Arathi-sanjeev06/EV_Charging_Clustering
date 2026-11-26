
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score

# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(
    BASE_DIR, "outputs", "stations_state_standardized.csv"
)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

BEST_K = 5
RANDOM_STATE = 42
N_INIT = 20
N_SEEDS = 20
N_BASELINE_RUNS = 30

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

required_columns = ["latitude", "longitude"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Missing column: {column}. Check the input CSV column names."
        )

df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

print("Stations loaded:", len(df))

# --------------------------------------------------
# 3. CONVERT LATITUDE/LONGITUDE TO 3D UNIT-SPHERE
# --------------------------------------------------

lat_rad = np.radians(df["latitude"].to_numpy())
lon_rad = np.radians(df["longitude"].to_numpy())

X = np.column_stack([
    np.cos(lat_rad) * np.cos(lon_rad),
    np.cos(lat_rad) * np.sin(lon_rad),
    np.sin(lat_rad)
])

# --------------------------------------------------
# 4. COMPARE K-MEANS AND WARD FOR K = 2 TO 10
# --------------------------------------------------

comparison_results = []

for k in range(2, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=N_INIT
    )
    kmeans_labels = kmeans.fit_predict(X)

    ward = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward"
    )
    ward_labels = ward.fit_predict(X)

    kmeans_silhouette = silhouette_score(X, kmeans_labels)
    ward_silhouette = silhouette_score(X, ward_labels)

    comparison_results.append({
        "K": k,
        "KMeans_Silhouette": kmeans_silhouette,
        "Ward_Silhouette": ward_silhouette
    })

    print(
        f"K={k}: K-Means={kmeans_silhouette:.4f}, "
        f"Ward={ward_silhouette:.4f}"
    )

comparison_df = pd.DataFrame(comparison_results)

comparison_df.to_csv(
    os.path.join(OUTPUT_DIR, "final_clustering_comparison.csv"),
    index=False
)

# --------------------------------------------------
# 5. FINAL CLUSTER ASSIGNMENTS USING K = 5
# --------------------------------------------------

final_kmeans = KMeans(
    n_clusters=BEST_K,
    random_state=RANDOM_STATE,
    n_init=N_INIT
)

df["kmeans_cluster"] = final_kmeans.fit_predict(X)

final_ward = AgglomerativeClustering(
    n_clusters=BEST_K,
    linkage="ward"
)

df["ward_cluster"] = final_ward.fit_predict(X)

df.to_csv(
    os.path.join(OUTPUT_DIR, "final_cluster_assignments_k5.csv"),
    index=False
)

print("\nFinal K-Means cluster sizes:")
print(df["kmeans_cluster"].value_counts().sort_index())

print("\nFinal Ward cluster sizes:")
print(df["ward_cluster"].value_counts().sort_index())

# --------------------------------------------------
# 6. CLUSTER SIZES AND GEOGRAPHIC CENTROIDS
# --------------------------------------------------

cluster_details = []

for cluster_id in sorted(df["kmeans_cluster"].unique()):

    cluster_data = df[df["kmeans_cluster"] == cluster_id]

    cluster_vectors = X[
        df["kmeans_cluster"].to_numpy() == cluster_id
    ]

    mean_vector = cluster_vectors.mean(axis=0)
    mean_vector = mean_vector / np.linalg.norm(mean_vector)

    centroid_lat = np.degrees(
        np.arctan2(
            mean_vector[2],
            np.sqrt(mean_vector[0] ** 2 + mean_vector[1] ** 2)
        )
    )

    centroid_lon = np.degrees(
        np.arctan2(mean_vector[1], mean_vector[0])
    )

    cluster_details.append({
        "cluster": int(cluster_id),
        "station_count": len(cluster_data),
        "centroid_latitude": centroid_lat,
        "centroid_longitude": centroid_lon
    })

cluster_details_df = pd.DataFrame(cluster_details)

cluster_details_df.to_csv(
    os.path.join(OUTPUT_DIR, "final_cluster_details_k5.csv"),
    index=False
)

print("\nFinal cluster details:")
print(cluster_details_df.to_string(index=False))

# --------------------------------------------------
# 7. K-MEANS STABILITY TEST: 20 RANDOM SEEDS
# --------------------------------------------------

seed_labels = []

for seed in range(N_SEEDS):

    model = KMeans(
        n_clusters=BEST_K,
        random_state=seed,
        n_init=N_INIT
    )

    labels = model.fit_predict(X)
    seed_labels.append(labels)

ari_scores = []

for labels_a, labels_b in itertools.combinations(seed_labels, 2):
    ari_scores.append(
        adjusted_rand_score(labels_a, labels_b)
    )

stability_mean = float(np.mean(ari_scores))
stability_std = float(np.std(ari_scores))

stability_df = pd.DataFrame({
    "test": ["KMeans K=5 stability"],
    "number_of_seeds": [N_SEEDS],
    "number_of_pairwise_comparisons": [len(ari_scores)],
    "mean_ARI": [stability_mean],
    "std_ARI": [stability_std]
})

stability_df.to_csv(
    os.path.join(OUTPUT_DIR, "final_clustering_stability_k5.csv"),
    index=False
)

print("\nK-Means stability test:")
print("Random seeds:", N_SEEDS)
print("Pairwise comparisons:", len(ari_scores))
print(f"Mean ARI: {stability_mean:.4f}")
print(f"Standard deviation: {stability_std:.4f}")

# --------------------------------------------------
# 8. RANDOM-LABEL REFERENCE AT K = 5
# --------------------------------------------------

reference_labels = df["kmeans_cluster"].to_numpy()
random_silhouettes = []

for seed in range(N_BASELINE_RUNS):

    rng = np.random.default_rng(seed)
    random_labels = rng.permutation(reference_labels)

    score = silhouette_score(X, random_labels)
    random_silhouettes.append(score)

baseline_mean = float(np.mean(random_silhouettes))
baseline_std = float(np.std(random_silhouettes))

baseline_df = pd.DataFrame({
    "reference": ["Randomly permuted K-Means labels"],
    "K": [BEST_K],
    "runs": [N_BASELINE_RUNS],
    "mean_silhouette": [baseline_mean],
    "std_silhouette": [baseline_std]
})

baseline_df.to_csv(
    os.path.join(OUTPUT_DIR, "random_label_reference_k5.csv"),
    index=False
)

print("\nRandom-label reference:")
print(f"Mean silhouette: {baseline_mean:.4f}")
print(f"Standard deviation: {baseline_std:.4f}")

# --------------------------------------------------
# 9. PLOT SILHOUETTE COMPARISON
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    comparison_df["K"],
    comparison_df["KMeans_Silhouette"],
    marker="o",
    label="K-Means"
)

plt.plot(
    comparison_df["K"],
    comparison_df["Ward_Silhouette"],
    marker="s",
    label="Ward Hierarchical"
)

plt.xlabel("Number of clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Comparison: K-Means and Ward")
plt.xticks(range(2, 11))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "final_silhouette_comparison.png"),
    dpi=300
)
plt.close()

# --------------------------------------------------
# 10. PLOT FINAL K-MEANS CLUSTERS
# --------------------------------------------------

plt.figure(figsize=(9, 7))

scatter = plt.scatter(
    df["longitude"],
    df["latitude"],
    c=df["kmeans_cluster"],
    cmap="tab10",
    s=18,
    alpha=0.75
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Geographical Clusters of EV Charging Stations (K=5)")
plt.colorbar(scatter, label="Cluster")
plt.grid(True, alpha=0.2)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "final_kmeans_clusters_k5.png"),
    dpi=300
)
plt.close()

# --------------------------------------------------
# 11. FINAL SUMMARY
# --------------------------------------------------

print("\n========== FINAL CLUSTERING SUMMARY ==========")
print("Total stations:", len(df))
print("Selected K:", BEST_K)

print(
    "K-Means silhouette:",
    round(
        comparison_df.loc[
            comparison_df["K"] == BEST_K,
            "KMeans_Silhouette"
        ].iloc[0],
        4
    )
)

print(
    "Ward silhouette:",
    round(
        comparison_df.loc[
            comparison_df["K"] == BEST_K,
            "Ward_Silhouette"
        ].iloc[0],
        4
    )
)

print("Mean stability ARI:", round(stability_mean, 4))
print("Random-label reference silhouette:", round(baseline_mean, 4))

print("\nFiles saved in outputs folder.")
print("==============================================")
