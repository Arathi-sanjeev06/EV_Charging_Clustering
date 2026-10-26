# Spatial Clustering of EV Charging Stations to Find Coverage Gaps

## 1. Project Overview

Electric vehicles (EVs) are an important part of the transition towards cleaner transportation. However, the availability of charging infrastructure is not evenly distributed across geographical areas.

This project uses spatial clustering and distance-based analysis to study the distribution of electric vehicle charging stations in India. It applies K-Means and Hierarchical Clustering to identify spatial patterns and uses nearest-neighbour distances and grid-based analysis to explore potential charging infrastructure coverage gaps.

The project supports **United Nations Sustainable Development Goals (SDGs) 7 and 11**, which focus on affordable and clean energy and sustainable cities and communities.

## 2. Objectives

The main objectives of this project are:

- To examine the geographical distribution of EV charging stations in India.
- To apply K-Means and Hierarchical Clustering to identify spatial groupings.
- To compare clustering methods using silhouette scores.
- To calculate nearest-station distances and identify locations that may be underserved.
- To conduct state-wise analysis to explore differences in station distribution.
- To visualize spatial patterns and potential coverage gaps.

## 3. Dataset

The project uses the *Electric Vehicle Charging Stations in India* dataset obtained from Kaggle.

The dataset contains information such as station name, state, city, address, latitude, longitude, and station type.

### Data preprocessing

The following preprocessing steps were performed:

- Converted latitude and longitude values into numeric format.
- Removed exact duplicate records.
- Removed records with missing or invalid coordinates.
- Applied geographical coordinate bounds to exclude invalid locations.
- Standardized selected state-name variations for state-wise analysis.

After preprocessing, **1,331 charging station records** remained in the cleaned dataset.

The original dataset contained 1,547 records. The cleaned dataset is stored in `data/cleaned_ev_stations.csv`.

## 4. Methodology

### 4.1 Exploratory Data Analysis

The geographical distribution of charging stations was examined using visualizations and state-wise station counts.

### 4.2 K-Means Clustering

K-Means was applied to latitude and longitude coordinates to identify spatial clusters. Different values of K were evaluated using silhouette scores to compare cluster separation and cohesion.

### 4.3 Hierarchical Clustering

Hierarchical Clustering was used as a second approach to examine geographical groupings. Its results were compared with those of K-Means for cluster counts from 2 to 10.

### 4.4 Coverage-Gap Analysis

Nearest-neighbour distances were calculated using geographic distance calculations based on the haversine formula. A spatial grid was also used to estimate distances from sampled locations to the nearest charging station.

A 25 km distance threshold was used as an exploratory indicator of potentially distant charging infrastructure. This threshold is an analytical choice for this project, not an official accessibility standard.

### 4.5 State-Wise Analysis

Station counts and nearest-neighbour distances were summarized by state or recorded region to examine differences in the spatial distribution of charging stations.

## 5. Results and Key Findings

### 5.1 Clustering Comparison

The best silhouette scores among the tested cluster counts were:

| Method | Number of clusters | Silhouette score |
|---|---:|---:|
| K-Means | 4 | 0.6469 |
| Hierarchical Clustering | 4 | 0.6219 |

K-Means achieved the higher silhouette score among the tested configurations. This indicates better cluster separation under the chosen representation and evaluation method; it does not establish that K-Means is universally superior for all spatial datasets.

### 5.2 Station Distribution

The cleaned dataset contained 1,331 station records. Maharashtra had the highest station count in the state-wise summary, with 225 records.

Among recorded states with at least 20 stations, Odisha had a mean nearest-neighbour distance of approximately 21.14 km, followed by Rajasthan at approximately 15.44 km. These results suggest that station spacing varies considerably across the dataset.

### 5.3 Potential Coverage Gaps

The nearest-neighbour analysis produced a median distance of approximately 1.00 km and a mean distance of approximately 7.34 km between a station and its nearest other station.

Some stations had very large nearest-neighbour distances. Such cases may indicate sparse local infrastructure, geographic separation, data-quality issues, or isolated locations that require further investigation.

The grid-based analysis also identified sampled locations more than 25 km from a recorded station. These locations are potential areas for further investigation, not confirmed areas without EV charging access.

## 6. Visualizations and Outputs

The project includes visualizations and CSV result tables.

### Figures

Stored in `outputs/figures/`:

- Station distribution map
- K-Means elbow-method plot
- Silhouette-score comparison
- K-Means cluster visualization
- Hierarchical clustering visualization
- Hierarchical dendrogram
- Coverage-gap map
- State-wise station-count chart
- State-wise median nearest-distance chart

### Tables

Stored in `outputs/tables/`:

- Clustering evaluation results
- K-Means and Hierarchical Clustering comparison
- Cluster assignments
- Coverage-gap grid results
- Coverage-gap summary
- State-wise station counts and distance summaries
- Nearest-neighbour distance results

## 7. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- SciPy

## 8. Project Structure

```text
EV_Charging_Clustering/
├── data/
│   ├── ev.csv
│   └── cleaned_ev_stations.csv
├── outputs/
│   ├── figures/
│   └── tables/
├── src/
│   ├── kmeans_clustering.py
│   ├── hierarchical_clustering.py
│   ├── coverage_gap_analysis.py
│   ├── state_wise_analysis.py
│   ├── compare_clustering.py
│   └── main.py
├── .gitignore
├── requirements.txt
└── README.md
```

## 9. How to Run the Project

### Prerequisites

Install Python and the required libraries.

From the project root directory, install dependencies with:

```bash
pip install -r requirements.txt
```

Run the analysis scripts using:

```bash
python src/kmeans_clustering.py
python src/hierarchical_clustering.py
python src/coverage_gap_analysis.py
python src/state_wise_analysis.py
python src/compare_clustering.py
```

On Windows, you may use `py` instead of `python`, for example:

```powershell
py src/compare_clustering.py
```

Run each script from the project root directory so that relative file paths resolve correctly.

The scripts may regenerate the associated result tables and figures.

## 10. Limitations

- The dataset represents recorded charging stations and may not include every operational station in India.
- Geographic coordinates and station attributes may contain errors or inconsistencies.
- Straight-line geographic distance does not represent road-network travel distance or actual travel time.
- Clustering latitude and longitude identifies spatial patterns but does not directly measure population demand, charging capacity, station reliability, or EV ownership.
- Grid-based coverage estimates depend on the selected grid, geographical boundary, and distance threshold.
- State-wise summaries may be affected by small sample sizes, duplicated coordinates, and inconsistent regional labels.

Therefore, the findings should be interpreted as an exploratory spatial analysis rather than a definitive assessment of EV charging accessibility.

## 11. Sustainable Development Goals

This project relates to:

- **SDG 7 — Affordable and Clean Energy:** Supporting the study of infrastructure needed for electric mobility.
- **SDG 11 — Sustainable Cities and Communities:** Exploring the geographical distribution of charging infrastructure to inform future planning.

## 12. Conclusion

This project demonstrates how spatial clustering, nearest-neighbour distance calculations, and geographical visualization can be used to examine the distribution of EV charging stations in India.

The comparison of clustering methods identifies K-Means with four clusters as the best-performing configuration among those tested using silhouette scores. State-wise summaries and distance-based analysis highlight substantial variation in station spacing and identify locations that may warrant further investigation.

Future work could incorporate population density, EV registrations, road networks, charging capacity, station operating status, and travel-time analysis to produce a more realistic assessment of charging accessibility.

## Data Source

The dataset was obtained from Kaggle: *Electric Vehicle Charging Stations in India*.

Please refer to the original dataset page for source information and licence details.
