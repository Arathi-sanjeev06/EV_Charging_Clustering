"""Spatial null model and bootstrap stability for K-Means (K=5) on the unit sphere."""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"; OUTPUT_DIR.mkdir(exist_ok=True)
K, N_INIT, SEED, N_NULL, N_BOOT = 5, 20, 42, 30, 100

def to_sphere(lat, lon):
    p, l = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(p)*np.cos(l), np.cos(p)*np.sin(l), np.sin(p)])

df = pd.read_csv(BASE_DIR / "data" / "cleaned_ev_stations_revised.csv")
X = to_sphere(df["latitude"].to_numpy(), df["longitude"].to_numpy())
rng = np.random.default_rng(SEED)

ref = KMeans(K, random_state=SEED, n_init=N_INIT).fit(X)
observed = silhouette_score(X, ref.labels_)

# Spatial null: random points over the same latitude/longitude bounding box
lat_lo, lat_hi = df["latitude"].min(), df["latitude"].max()
lon_lo, lon_hi = df["longitude"].min(), df["longitude"].max()
null = []
for _ in range(N_NULL):
    P = to_sphere(rng.uniform(lat_lo, lat_hi, len(df)), rng.uniform(lon_lo, lon_hi, len(df)))
    null.append(silhouette_score(P, KMeans(K, random_state=SEED, n_init=N_INIT).fit_predict(P)))

# Bootstrap: resample stations with replacement, refit, compare with full-data labels
boot_sil, boot_ari = [], []
for b in range(N_BOOT):
    idx = rng.integers(0, len(X), len(X))
    m = KMeans(K, random_state=b, n_init=N_INIT).fit(X[idx])
    boot_sil.append(silhouette_score(X[idx], m.labels_))
    boot_ari.append(adjusted_rand_score(ref.labels_, m.predict(X)))

out = pd.DataFrame([
    ["observed_silhouette_K5", observed, np.nan, 1],
    ["spatial_null_silhouette_K5", np.mean(null), np.std(null), N_NULL],
    ["bootstrap_silhouette_K5", np.mean(boot_sil), np.std(boot_sil), N_BOOT],
    ["bootstrap_ARI_vs_full_data", np.mean(boot_ari), np.std(boot_ari), N_BOOT],
], columns=["metric", "mean", "std", "runs"])
out.to_csv(OUTPUT_DIR / "robustness_checks.csv", index=False)
print(out.round(4).to_string(index=False))
