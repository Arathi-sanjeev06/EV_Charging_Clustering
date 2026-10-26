
import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned dataset
df = pd.read_csv("data/cleaned_ev_stations.csv")

# Plot charging station locations
plt.figure(figsize=(10, 7))
plt.scatter(
    df["longitude"],
    df["lattitude"],
    alpha=0.6,
    s=15
)

plt.title("Geographical Distribution of EV Charging Stations in India")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig("station_distribution.png", dpi=300)
plt.show()

print("Map plot saved as station_distribution.png")
