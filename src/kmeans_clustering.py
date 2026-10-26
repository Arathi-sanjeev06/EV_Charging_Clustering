
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. LOAD THE CLEANED DATASET
# ==========================================

df = pd.read_csv("data/cleaned_ev_stations.csv")

print("EV CHARGING STATION CLUSTERING")
print("=" * 40)
print("Number of stations:", len(df))

# ==========================================
# 2. SELECT GEOGRAPHICAL COORDINATES
# ==========================================

coordinates = df[["lattitude", "longitude"]].copy()

# Standardize the coordinates
scaler = StandardScaler()
X_scaled = scaler.fit_transform(coordinates)

# ==========================================
# 3. EVALUATE DIFFERENT K VALUES
# ==========================================

k_values = range(2, 11)
inertia_values = []
silhouette_values = []

print("\nEvaluating K-means for K = 2 to 10...\n")

for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    inertia = model.inertia_
    silhouette = silhouette_score(X_scaled, labels)

    inertia_values.append(inertia)
    silhouette_values.append(silhouette)

    print(
        f"K = {k} | "
        f"Inertia = {inertia:.2f} | "
        f"Silhouette Score = {silhouette:.4f}"
    )

# ==========================================
# 4. SAVE THE ELBOW METHOD GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    inertia_values,
    marker="o"
)

plt.title("Elbow Method for K-means Clustering")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(list(k_values))
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("outputs/elbow_method.png", dpi=300)
plt.close()

# ==========================================
# 5. SAVE THE SILHOUETTE SCORE GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    silhouette_values,
    marker="o"
)

plt.title("Silhouette Scores for K-means")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(list(k_values))
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("outputs/silhouette_scores.png", dpi=300)
plt.close()

# ==========================================
# 6. SELECT THE BEST K
# ==========================================

best_k = list(k_values)[silhouette_values.index(max(silhouette_values))]

print("\n" + "=" * 40)
print("BEST K-MEANS RESULT")
print("=" * 40)

print("Best K by silhouette score:", best_k)
print(
    "Best silhouette score:",
    round(max(silhouette_values), 4)
)

# ==========================================
# 7. TRAIN THE FINAL K-MEANS MODEL
# ==========================================

final_model = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df["cluster"] = final_model.fit_predict(X_scaled)

# ==========================================
# 8. DISPLAY STATIONS IN EACH CLUSTER
# ==========================================

print("\nNumber of stations in each cluster:")

print(
    df["cluster"]
    .value_counts()
    .sort_index()
)

# ==========================================
# 9. SAVE THE CLUSTERED DATASET
# ==========================================

df.to_csv(
    "outputs/kmeans_clustered_stations.csv",
    index=False
)

# ==========================================
# 10. SAVE THE EVALUATION RESULTS
# ==========================================

evaluation = pd.DataFrame({
    "K": list(k_values),
    "Inertia": inertia_values,
    "Silhouette_Score": silhouette_values
})

evaluation.to_csv(
    "outputs/kmeans_evaluation.csv",
    index=False
)

# ==========================================
# 11. VISUALIZE THE FINAL CLUSTERS
# ==========================================

plt.figure(figsize=(10, 7))

plt.scatter(
    df["longitude"],
    df["lattitude"],
    c=df["cluster"],
    cmap="viridis",
    s=15,
    alpha=0.7
)

plt.title(f"K-means Clustering of EV Stations (K={best_k})")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("outputs/kmeans_clusters.png", dpi=300)
plt.show()

# ==========================================
# 12. FINISH
# ==========================================

print("\nAll files saved successfully!")
print("Check your outputs folder for:")

print("- elbow_method.png")
print("- silhouette_scores.png")
print("- kmeans_clusters.png")
print("- kmeans_clustered_stations.csv")
print("- kmeans_evaluation.csv")
