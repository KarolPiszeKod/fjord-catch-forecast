"""
ingest_copernicus.py
=====================
Downloads ocean data for four ports from Copernicus Marine Service and
aggregates it weekly:
- sea surface temperature (SST), weekly mean;
- chlorophyll-a, weekly mean;
- significant wave height (VHM0), weekly maximum. Peak events are preserved
    rather than smoothed by a mean; merge_sources.py marks storm weeks when VHM0
    exceeds config.STORM_WAVE_THRESHOLD_M.

Requires a free account at https://data.marine.copernicus.eu/register.
On the first run, credentials can be entered in the terminal and are stored
locally in ~/.copernicusmarine. Alternatively, set
COPERNICUSMARINE_SERVICE_USERNAME and COPERNICUSMARINE_SERVICE_PASSWORD in
.env; the library supports this for non-interactive Docker or CI runs.

Datasets (identifiers and variables are also listed in config.py):
- Physics (SST): GLOBAL_MULTIYEAR_PHY_001_030, variable "thetao".
- Biogeochemistry (chlorophyll): GLOBAL_MULTIYEAR_BGC_001_029, variable "chl".
- Waves (WAVERYS): GLOBAL_MULTIYEAR_WAV_001_032, variable "VHM0".

The "_my_" multi-year reanalysis products usually lag real time by months.
Live forecasting requires suitable "_anfc_" analysis-and-forecast products;
check availability for Norway and update config.py if needed. In its current
form, this script builds training data from 2013-2025 history.

Grid-point selection note: Norwegian fishing ports often lie in narrow fjords.
At the datasets' 0.083-0.25 degree resolution, the nearest geographic point
may fall on a land-mask pixel with only NaN values. _weekly_series therefore
searches the downloaded BBOX_MARGIN area for the nearest grid point with
valid data instead of blindly using sel(..., method="nearest").
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import copernicusmarine
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (  # noqa: E402
    BBOX_MARGIN,
    CHL_DATASET_ID,
    CHL_VARIABLE,
    END_DATE,
    PORT_COORDS,
    SST_DATASET_ID,
    SST_VARIABLE,
    START_DATE,
    WAVE_DATASET_ID,
    WAVE_VARIABLE,
)

try:
    from dotenv import load_dotenv
    load_dotenv()  # Allows Copernicus credentials in .env instead of manual entry.
except ImportError:
    pass


def fetch_variable_for_port(dataset_id: str, variable: str, lat: float, lon: float,
                             start_date: str, end_date: str, out_dir: Path) -> xr.Dataset:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{variable}_{lat:.2f}_{lon:.2f}.nc"
    copernicusmarine.subset(
        dataset_id=dataset_id,
        variables=[variable],
        minimum_longitude=lon - BBOX_MARGIN,
        maximum_longitude=lon + BBOX_MARGIN,
        minimum_latitude=lat - BBOX_MARGIN,
        maximum_latitude=lat + BBOX_MARGIN,
        start_datetime=f"{start_date}T00:00:00",
        end_datetime=f"{end_date}T00:00:00",
        minimum_depth=0,
        maximum_depth=1,
        output_directory=str(out_dir),
        output_filename=out_file.name,
        overwrite=True,
    )
    return xr.open_dataset(out_file)


def _nearest_valid_point(ds: xr.Dataset, variable: str, lat: float, lon: float) -> xr.DataArray:
    """Find the nearest grid point in the downloaded area with valid data.

    The geographically nearest point may be land because the grid is coarse
    relative to narrow fjords.
    """
    da = ds[variable]
    if "depth" in da.dims:
        da = da.isel(depth=0)

    has_data = da.notnull().any(dim="time")

    lat_grid, lon_grid = np.meshgrid(da.latitude.values, da.longitude.values, indexing="ij")
    dist = np.sqrt((lat_grid - lat) ** 2 + (lon_grid - lon) ** 2)
    dist = np.where(has_data.values, dist, np.inf)

    if np.all(np.isinf(dist)):
        raise ValueError(
            f"No valid water data exists in the downloaded area for {variable} "
            f"at ({lat},{lon}). Increase BBOX_MARGIN in config.py."
        )

    idx_flat = np.argmin(dist)
    i, j = np.unravel_index(idx_flat, dist.shape)
    found_lat = da.latitude.values[i]
    found_lon = da.longitude.values[j]

    return da.sel(latitude=found_lat, longitude=found_lon)


def _weekly_series(ds: xr.Dataset, variable: str, lat: float, lon: float, agg: str) -> pd.DataFrame:
    """Select the nearest valid water point and return weekly ISO aggregates."""
    point = _nearest_valid_point(ds, variable, lat, lon)
    df = point.to_dataframe(name=variable).reset_index()[["time", variable]]
    df["time"] = pd.to_datetime(df["time"])
    iso = df["time"].dt.isocalendar()
    df["year"] = iso["year"]
    df["week"] = iso["week"]
    grouped = df.groupby(["year", "week"])[variable]
    weekly = (grouped.max() if agg == "max" else grouped.mean()).reset_index()
    return weekly


def build_ocean_features(start_date: str = START_DATE, end_date: str = END_DATE) -> pd.DataFrame:
    rows = []
    tmp_dir = Path("data/raw/copernicus_tmp")

    for port, (lat, lon) in PORT_COORDS.items():
        print(f"\n=== Port {port} ({lat}, {lon}) ===")

        print("Downloading SST...")
        sst_ds = fetch_variable_for_port(SST_DATASET_ID, SST_VARIABLE, lat, lon, start_date, end_date, tmp_dir)
        sst_weekly = _weekly_series(sst_ds, SST_VARIABLE, lat, lon, agg="mean")
        sst_weekly = sst_weekly.rename(columns={SST_VARIABLE: "sst_celsius"})

        print("Downloading chlorophyll...")
        chl_ds = fetch_variable_for_port(CHL_DATASET_ID, CHL_VARIABLE, lat, lon, start_date, end_date, tmp_dir)
        chl_weekly = _weekly_series(chl_ds, CHL_VARIABLE, lat, lon, agg="mean")
        chl_weekly = chl_weekly.rename(columns={CHL_VARIABLE: "chlorophyll_a"})

        print("Downloading wave height (VHM0)...")
        wave_ds = fetch_variable_for_port(WAVE_DATASET_ID, WAVE_VARIABLE, lat, lon, start_date, end_date, tmp_dir)
        wave_weekly = _weekly_series(wave_ds, WAVE_VARIABLE, lat, lon, agg="max")
        wave_weekly = wave_weekly.rename(columns={WAVE_VARIABLE: "wave_height_max"})

        merged = sst_weekly.merge(chl_weekly, on=["year", "week"], how="outer")
        merged = merged.merge(wave_weekly, on=["year", "week"], how="outer")
        merged["port"] = port
        rows.append(merged)

    result = pd.concat(rows, ignore_index=True)
    return result


if __name__ == "__main__":
    df = build_ocean_features()
    Path("data").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/ocean_features.csv", index=False)
    print(f"\nSaved {len(df)} rows -> data/ocean_features.csv")
    print(df.head(10))
    print("\nMissing values per column:")
    print(df.isna().sum())
