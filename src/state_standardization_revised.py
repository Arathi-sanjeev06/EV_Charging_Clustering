
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

INPUT_FILE = DATA_DIR / "cleaned_ev_stations_revised.csv"

df = pd.read_csv(INPUT_FILE)

# Preserve original state labels
df["state_original"] = df["state"]

# Standardise spaces and capitalisation
df["state_normalized"] = (
    df["state"]
    .astype(str)
    .str.replace("\u00a0", " ", regex=False)
    .str.strip()
    .str.lower()
)

# Correct clear spelling variations
aliases = {
    "andhra pradesh": "Andhra Pradesh",
    "andhra pradesh ": "Andhra Pradesh",
    "andhraPradesh".lower(): "Andhra Pradesh",
    "andra pradesh": "Andhra Pradesh",

    "tamil nadu": "Tamil Nadu",
    "tamilnadu": "Tamil Nadu",
    "taminadu": "Tamil Nadu",

    "maharashra": "Maharashtra",
    "punjab": "Punjab",
    "telengana": "Telangana",

    "chattisgarh": "Chhattisgarh",
    "uttarkhand": "Uttarakhand",
    "uttrakhand": "Uttarakhand",
    "harayana": "Haryana",

    "westbengal": "West Bengal",

    "jammu": "Jammu and Kashmir",
    "jammu & kashmir": "Jammu and Kashmir",
    "jammu and kashmir": "Jammu and Kashmir",

    "pondicherry": "Puducherry",

    "karala": "Kerala",
    "hyderabadu00a0": "Telangana"
}

# Clearly identifiable city/locality labels
# These are inferred from the place name, not verified from
# an authoritative station-address lookup.
location_aliases = {
    "bhubhaneswar": "Odisha",
    "jajpur": "Odisha",
    "ernakulam": "Kerala",
    "kochi": "Kerala",
    "hisar": "Haryana",
    "limbdi": "Gujarat",
    "hyderabad": "Telangana",
    "rajahmundry": "Andhra Pradesh"
}

# Known official state/UT names
official_names = {
    "andaman": "Andaman",
    "assam": "Assam",
    "bihar": "Bihar",
    "chandigarh": "Chandigarh",
    "chhattisgarh": "Chhattisgarh",
    "delhi": "Delhi",
    "delhi ncr": "Delhi NCR",
    "goa": "Goa",
    "gujarat": "Gujarat",
    "haryana": "Haryana",
    "himachal pradesh": "Himachal Pradesh",
    "jharkhand": "Jharkhand",
    "karnataka": "Karnataka",
    "kerala": "Kerala",
    "madhya pradesh": "Madhya Pradesh",
    "maharashtra": "Maharashtra",
    "odisha": "Odisha",
    "puducherry": "Puducherry",
    "punjab": "Punjab",
    "rajasthan": "Rajasthan",
    "sikkim": "Sikkim",
    "telangana": "Telangana",
    "tripura": "Tripura",
    "uttar pradesh": "Uttar Pradesh",
    "uttarakhand": "Uttarakhand",
    "west bengal": "West Bengal",
    "andhra pradesh": "Andhra Pradesh",
    "jammu and kashmir": "Jammu and Kashmir"
}

def standardize_state(value):

    if value in aliases:
        return aliases[value], "corrected_spelling"

    if value in location_aliases:
        return location_aliases[value], "inferred_from_place_name"

    if value in official_names:
        return official_names[value], "verified_name_format"

    return value.title(), "review_required"


mapped = df["state_normalized"].apply(standardize_state)

df["state_cleaned"] = mapped.apply(lambda x: x[0])
df["state_mapping_status"] = mapped.apply(lambda x: x[1])

# Keep Delhi NCR as a separate regional label:
# it is not a single state/UT and should not be silently
# assigned to Delhi.

# Save complete station data
station_file = OUTPUT_DIR / "stations_state_standardized.csv"
df.to_csv(station_file, index=False)

# Save records needing manual review
review = df[
    df["state_mapping_status"] == "review_required"
]

review_file = OUTPUT_DIR / "state_labels_needing_review.csv"
review.to_csv(review_file, index=False)

# Count standardized labels
counts = (
    df.groupby(["state_cleaned", "state_mapping_status"])
    .size()
    .reset_index(name="Station_Count")
    .sort_values("Station_Count", ascending=False)
)

counts.to_csv(
    OUTPUT_DIR / "standardized_state_counts.csv",
    index=False
)

print("========== STATE STANDARDIZATION ==========")
print("Total stations:", len(df))
print("Distinct original labels:", df["state_original"].nunique())
print("Final labels:", df["state_cleaned"].nunique())

print("\nMapping status:")
print(df["state_mapping_status"].value_counts().to_string())

print("\nStandardized state counts:")
print(counts.to_string(index=False))

print("\nLabels requiring review:")
if len(review):
    print(
        review[["state_original", "city", "name"]]
        .to_string(index=False)
    )
else:
    print("None")

print("\nSaved:", station_file)
print("Review file:", review_file)
print("============================================")
