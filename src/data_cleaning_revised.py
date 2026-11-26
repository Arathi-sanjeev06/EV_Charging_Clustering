
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "ev.csv"
OUTPUT_FILE = DATA_DIR / "cleaned_ev_stations_revised.csv"

# Load original dataset
df = pd.read_csv(INPUT_FILE)

original_rows = len(df)
original_columns = len(df.columns)

print("========== DATA CLEANING ==========")
print("Original shape:", df.shape)

# Remove unnecessary spaces from text columns
for col in ["name", "state", "city", "address"]:
    df[col] = df[col].astype("string").str.strip()

# Remove exact duplicate rows
duplicates_removed = int(df.duplicated().sum())
df = df.drop_duplicates().copy()

print("Exact duplicates removed:", duplicates_removed)

# Convert coordinates to numeric values
# A trailing comma in a coordinate is treated as formatting noise.
df["lattitude"] = pd.to_numeric(
    df["lattitude"].astype("string").str.strip().str.rstrip(","),
    errors="coerce"
)

df["longitude"] = pd.to_numeric(
    df["longitude"].astype("string").str.strip().str.rstrip(","),
    errors="coerce"
)

# Correct only clearly reversed latitude/longitude pairs.
# Broad screening ranges for India: latitude 6–38, longitude 68–98.
swapped = (
    df["lattitude"].between(68, 98)
    & df["longitude"].between(6, 38)
)

swap_count = int(swapped.sum())

df.loc[swapped, ["lattitude", "longitude"]] = (
    df.loc[swapped, ["longitude", "lattitude"]].to_numpy()
)

print("Coordinate pairs swapped:", swap_count)

# Identify missing and invalid coordinates
missing_coordinates = (
    df["lattitude"].isna() | df["longitude"].isna()
)

valid_coordinates = (
    df["lattitude"].between(6, 38)
    & df["longitude"].between(68, 98)
)

invalid_coordinates = ~valid_coordinates

missing_count = int(missing_coordinates.sum())
invalid_count = int(invalid_coordinates.sum())

# Save excluded records separately for transparency
excluded = df[invalid_coordinates].copy()
excluded_file = DATA_DIR / "excluded_invalid_coordinates.csv"
excluded.to_csv(excluded_file, index=False)

# Keep only records with valid coordinates
df = df[valid_coordinates].copy()

# Rename the misspelled latitude column
df = df.rename(columns={"lattitude": "latitude"})

# Save revised cleaned dataset
df.to_csv(OUTPUT_FILE, index=False)

# Print cleaning summary
print("\n========== CLEANING SUMMARY ==========")
print("Original rows:", original_rows)
print("Original columns:", original_columns)
print("Duplicate rows removed:", duplicates_removed)
print("Rows with missing coordinates:", missing_count)
print("Rows with invalid coordinates:", invalid_count)
print("Coordinate pairs corrected:", swap_count)
print("Final cleaned rows:", len(df))
print("Final columns:", len(df.columns))

print("\nMissing values in final dataset:")
print(df.isnull().sum())

print("\nFinal latitude range:",
      df["latitude"].min(), "to", df["latitude"].max())

print("Final longitude range:",
      df["longitude"].min(), "to", df["longitude"].max())

print("\nCleaned file saved to:", OUTPUT_FILE)
print("Excluded records saved to:", excluded_file)
print("====================================")
