# Spatial Clustering of EV Charging Stations to Identify Coverage Gaps in India

Exploratory analysis of 1,334 cleaned charging station records: K-Means and Ward clustering on a 3D unit-sphere
representation of latitude/longitude, plus Haversine nearest-station distance and convex-hull grid analysis.
Author: Arathi Sanjeev (Roll no. 24020570009), University of Delhi.

## Data
- Source: Kaggle dataset "Electric Vehicle Charging Stations in India" by saketpradhan.
  URL: https://www.kaggle.com/datasets/saketpradhan/electric-vehicle-charging-stations-in-india
  Licence: CC0 1.0 Public Domain (https://creativecommons.org/publicdomain/zero/1.0/). Accessed: 28 November 2026.
- `data/ev.csv` - raw file (1,547 rows, 7 columns: name, state, city, address, lattitude, longitude, type).
- `data/cleaned_ev_stations_revised.csv` - cleaned file produced by `src/data_cleaning_revised.py` (1,334 rows).
- `data/excluded_invalid_coordinates.csv` - records removed for missing or out-of-range coordinates.

## How to run (from the project root)
```bash
pip install -r requirements.txt
python src/run_all.py          # on Windows you can use: py src/run_all.py
```
`run_all.py` runs, in order: `data_cleaning_revised.py`, `state_standardization_revised.py`,
`final_clustering_analysis.py`, `generate_ward_dendrogram.py`, `robustness_checks.py`,
`coverage_gap_revised.py`, `state_coverage_summary_final.py`, `plot_station_distribution.py`.
All tables and figures are written to `outputs/`. Seeds are fixed (`random_state=42`, `n_init=20`).

## Pipeline and results
| Step | Result |
|---|---|
| Cleaning | 1,547 rows -> 1,334 (205 exact duplicates, 6 missing coordinates, 2 outside 6-38 N / 68-98 E removed; 2 swapped lat/lon pairs corrected) |
| State labels | 58 raw labels -> 30 (1,069 verified, 256 spelling corrected, 8 inferred from place name, 1 flagged: Chikhali) |
| Representation | latitude/longitude -> 3D unit sphere (x, y, z) before clustering |
| Model selection | K = 2..10; best silhouette at K = 5: K-Means 0.6507, Ward 0.6459 (`final_clustering_comparison.csv`) |
| Stability | 20 seeds, 190 pairwise ARI: mean 1.0000, sd 0.0000; random-label silhouette -0.0437 (sd 0.0191) |
| Spatial null / bootstrap | null (random points in bounding box) 0.3866 (sd 0.0074); bootstrap silhouette 0.6514 (sd 0.0071), ARI 0.9982 (sd 0.0026) |
| Nearest-station distance | mean 7.33 km, median 1.01 km, max 1,150.09 km; 88 stations (6.60%) > 25 km; 264 records at 0.00 km |
| Convex-hull grid (120 x 120) | 9,943 points, 8,954 (90.05%) > 25 km; median 102.27 km, max 708.03 km |

## Outputs used in the paper
Figure 1 `final_silhouette_comparison.png`; Figure 2 `final_kmeans_clusters_k5.png`; Figure 3 `ward_dendrogram.png`;
Figure 4 `coverage_gap_map_revised.png`; station map `station_distribution.png`.
Table 3 `state_coverage_summary_final.csv`; Table 4 `final_clustering_comparison.csv`;
Table 5 `final_cluster_details_k5.csv`; per-station cluster membership `final_cluster_assignments_k5.csv`.

## Limitations
- Station coordinates only: no demand, traffic, road network or charger capacity.
- Haversine distance is straight-line, not road distance; the 25 km threshold is a highway-planning benchmark, not a universal standard.
- The grid covers the convex hull of the stations, which includes sea and territory outside India; the 90.05% is not a share of India's land area.
- State labels come from the source column; Delhi NCR is kept as a separate label and one record (Chikhali) needs manual review.
- The dataset may not contain every operational charging station in India.
