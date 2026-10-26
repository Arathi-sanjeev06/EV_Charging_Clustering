
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

# 1. Load the cleaned dataset
df = pd.read_csv("data/cleaned_ev_stations.csv")

# 2. Select latitude and longitude
coordinates = df[["lattitude", "longitude"]].copy()

# 3. Standardize the coordinates
scaler = StandardScaler()
X_scaled = scaler.fit_transform(coordinates)

# 4. Perform hierarchical clustering
k = 4

model = AgglomerativeClustering(
    n_clusters=k,
    linkage="ward"
)

df["cluster"] = model.fit_predict(X_scaled)

# 5. Calculate silhouette score
score = silhouette_score(X_scaled, df["cluster"])

print("Hierarchical Clustering Results")
print("--------------------------------")
print("Number of stations:", len(df))
print("Number of clusters:", k)
print("Silhouette Score:", round(score, 4))

# 6. Display the number of stations in each cluster
print("\nStations in each cluster:")
print(df["cluster"].value_counts().sort_index())

# 7. Save clustered dataset
df.to_csv("outputs/hierarchical_clustered_stations.csv", index=False)

# 8. Plot the geographical clusters
plt.figure(figsize=(10, 7))

plt.scatter(
    df["longitude"],
    df["lattitude"],
    c=df["cluster"],
    cmap="viridis",
    s=15,
    alpha=0.7
)

plt.title("Hierarchical Clustering of EV Charging Stations")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/hierarchical_clusters.png", dpi=300)
plt.show()

# 9. Create a dendrogram using a sample of stations
sample_size = min(250, len(X_scaled))
sample_data = X_scaled[:sample_size]

linked = linkage(sample_data, method="ward")

plt.figure(figsize=(12, 6))
dendrogram(linked, truncate_mode="lastp", p=25)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Cluster Groups")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig("outputs/hierarchical_dendrogram.png", dpi=300)
plt.show()

print("\nFiles saved successfully:")
print("- outputs/hierarchical_clustered_stations.csv")
print("- outputs/hierarchical_clusters.png")
print("- outputs/hierarchical_dendrogram.png")
