"""Map of all cleaned charging station records (final dataset)."""
import pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
(BASE_DIR / "outputs").mkdir(exist_ok=True)
df = pd.read_csv(BASE_DIR / "data" / "cleaned_ev_stations_revised.csv")
plt.figure(figsize=(8, 7))
plt.scatter(df["longitude"], df["latitude"], s=10, alpha=0.6)
plt.xlabel("Longitude"); plt.ylabel("Latitude")
plt.title(f"Recorded EV charging stations ({len(df):,} records)")
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(BASE_DIR / "outputs" / "station_distribution.png", dpi=300)
print("Saved outputs/station_distribution.png")
