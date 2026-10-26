
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ==========================================
# 1. LOAD CLEANED DATA
# ==========================================

df = pd.read_csv("data/cleaned_ev_stations.csv")

coordinates = df[["lattitude", "longitude"]].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(coordinates)

print("=" * 55)
print("K-MEANS VS HIERARCHICAL CLUSTERING")
print("=" * 55)

# ==========================================
# 2. K-MEANS CLUSTERING
# ==========================================

k_values = range(2, 11)
results = []

for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    results.append({
        "Method": "K-means",
        "Number_of_Clusters": k,
        "Silhouette_Score": score
    })

# ==========================================
# 3. HIERARCHICAL CLUSTERING
# ==========================================

for k in k_values:
    model = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward"
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    results.append({
        "Method": "Hierarchical",
        "Number_of_Clusters": k,
        "Silhouette_Score": score
    })

# ==========================================
# 4. DISPLAY COMPARISON
# ==========================================

results_df = pd.DataFrame(results)

print("\nSilhouette score comparison:")
print(
    results_df.pivot(
        index="Number_of_Clusters",
        columns="Method",
        values="Silhouette_Score"
    ).round(4)
)

# ==========================================
# 5. FIND BEST RESULT FOR EACH METHOD
# ==========================================

best_indices = results_df.groupby("Method")[
    "Silhouette_Score"
].idxmax()

best_results = results_df.loc[best_indices].reset_index(drop=True)

print("\nBest result for each method:")
print(best_results.round(4).to_string(index=False))

# ==========================================
# 6. SAVE RESULTS
# ==========================================

results_df.to_csv(
    "outputs/clustering_method_comparison.csv",
    index=False
)

best_results.to_csv(
    "outputs/best_clustering_comparison.csv",
    index=False
)

print("\nSaved:")
print("- outputs/clustering_method_comparison.csv")
print("- outputs/best_clustering_comparison.csv")
print("\nComparison completed!")

