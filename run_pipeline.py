"""
run_pipeline.py
================
Runs the full pipeline in the correct order and stops with a clear message
when an input is missing. Fisheries Directorate files must be downloaded
manually; see the README.

Usage:
    python run_pipeline.py            # caly pipeline: ingest -> merge -> train
    python run_pipeline.py --skip-copernicus   # skip Copernicus download
                                                # when ocean data already exists
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(name: str, cmd: list[str]) -> bool:
    print(f"\n{'='*70}\n>>> {name}\n{'='*70}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nERROR: step '{name}' exited with code {result.returncode}. Stopping.")
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-copernicus", action="store_true",
                        help="Skip ocean-data download and use existing data/ocean_features.csv")
    args = parser.parse_args()

    py = sys.executable

    raw_fiskeridir_dir = Path("data/raw/fiskeridir")
    has_source_files = any(raw_fiskeridir_dir.glob("*.csv")) or any(raw_fiskeridir_dir.glob("*.xlsx"))
    if not has_source_files:
        print(
            "\nMISSING source files in data/raw/fiskeridir/.\n"
            "Download them manually; the Fisheries Directorate does not provide the required API.\n"
            "See the README instructions. Stopping the pipeline.\n"
        )
        sys.exit(1)

    if not run_step("1/5 ingest_fiskeridir.py", [py, "src/ingest_fiskeridir.py"]):
        sys.exit(1)

    if not args.skip_copernicus:
        if not run_step("2/5 ingest_copernicus.py", [py, "src/ingest_copernicus.py"]):
            sys.exit(1)
    else:
        print("\n>>> Skipping ingest_copernicus.py (--skip-copernicus)")
        if not Path("data/ocean_features.csv").exists():
            print("ERROR: data/ocean_features.csv is missing, but --skip-copernicus requires it.")
            sys.exit(1)

    if not run_step("3/5 merge_sources.py", [py, "src/merge_sources.py"]):
        sys.exit(1)

    if not run_step("4/5 feature_selection.py (Boruta)", [py, "src/feature_selection.py"]):
        sys.exit(1)

    if not run_step("5/5 train.py", [py, "src/train.py"]):
        sys.exit(1)

    print(
        "\n" + "=" * 70 +
        "\nDONE. Start the dashboard with: streamlit run app.py\n" +
        "=" * 70
    )


if __name__ == "__main__":
    main()
