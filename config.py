"""
config.py
=========
Single place for constants used by the full pipeline (ingest -> merge ->
features -> train -> app). Change ports, species, and years here only.
"""

from __future__ import annotations

# --- Data range ----------------------------------------------------------
# Fisheries Directorate data: 2013-2026 (2026 is the current incomplete year).
START_DATE = "2013-01-01"
END_DATE = "2025-12-31"

# --- Ports: name -> (latitude, longitude) of port center ------------------
# Keys use uppercase, matching the "Landingskommune" format in source data.
# The same spelling is used throughout ingestion, merging, and features.
# Existing ocean_features.csv files from before this convention may need
# their port names normalized or regenerated.
PORT_COORDS = {
    "TROMSØ": (69.6492, 18.9553),
    "ÅLESUND": (62.4722, 6.1495),
    "BODØ": (67.2804, 14.4049),
    "VÅGAN": (68.2341, 14.5683),  # municipality containing Svolvaer; Landingskommune
                                   # stores the municipality, not the port city name
}

PORTS = list(PORT_COORDS.keys())

# --- Species: canonical name -> variants found in source data -------------
SPECIES_FILTER = {
    "torsk": ["torsk", "cod", "gadus morhua"],
    "sild": ["sild", "herring", "clupea"],
    "sei": ["sei", "saithe", "pollachius"],
}

SPECIES = list(SPECIES_FILTER.keys())

# --- Training time split (must stay consistent in train.py and app.py) -----
# The test set must contain the newest available years to imitate forecasting.
TEST_YEARS = [2024, 2025]
CALIB_YEARS = [2023]

# --- Quantile regression --------------------------------------------------
import numpy as np  # noqa: E402

QUANTILES = np.array([0.1, 0.5, 0.9])
TARGET_COVERAGE = 0.8

# --- Copernicus Marine: dataset identifiers -------------------------------
SST_DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"  # GLOBAL_MULTIYEAR_PHY_001_030
SST_VARIABLE = "thetao"

CHL_DATASET_ID = "cmems_mod_glo_bgc_my_0.25deg_P1D-m"  # GLOBAL_MULTIYEAR_BGC_001_029
CHL_VARIABLE = "chl"

WAVE_DATASET_ID = "cmems_mod_glo_wav_my_0.2deg_PT3H-i"  # GLOBAL_MULTIYEAR_WAV_001_032
WAVE_VARIABLE = "VHM0"  # significant wave height (Hm0), in metres

# Above this weekly maximum wave height (m), mark the week as stormy.
STORM_WAVE_THRESHOLD_M = 4.0

BBOX_MARGIN = 0.3  # degrees; margin around the port point for Copernicus queries

# --- Boruta-lite: walk-forward candidate feature selection ----------------
BORUTA_MIN_TRAIN_YEARS = 3
BORUTA_ITERATIONS = 15
BORUTA_ITERATION_HIT_THRESHOLD = 0.6
BORUTA_RECENT_FOLDS_REQUIRED = 3
BORUTA_OVERALL_FOLD_CONFIRM_FRACTION = 0.7
BORUTA_RF_PARAMS = {"n_estimators": 250, "max_depth": 7, "min_samples_leaf": 5, "n_jobs": -1}
