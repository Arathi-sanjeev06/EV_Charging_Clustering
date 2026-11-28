"""Run the whole pipeline from the project root:  python src/run_all.py"""
import subprocess, sys
from pathlib import Path
SRC = Path(__file__).resolve().parent
STEPS = ["data_cleaning_revised.py", "state_standardization_revised.py",
         "final_clustering_analysis.py", "generate_ward_dendrogram.py",
         "robustness_checks.py", "coverage_gap_revised.py",
         "state_coverage_summary_final.py", "plot_station_distribution.py"]
for step in STEPS:
    print(f"\n>>> {step}")
    subprocess.run([sys.executable, str(SRC / step)], check=True)
print("\nAll steps finished. Results are in outputs/")
