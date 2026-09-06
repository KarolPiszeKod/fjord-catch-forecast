"""
merge_sources.py
==================
Merges:
- data/catch_data_real.csv from ingest_fiskeridir.py;
- data/ocean_features.csv from ingest_copernicus.py.

Writes data/raw_catch_data.csv using the schema expected by features.py,
train.py, and app.py:
year, week, port, species, tons, quota_pct_used, sst_celsius,
    chlorophyll_a, wave_height_max, storm, date, n_vessels, dominant_gear

Derived columns are calculated from real source data:
- quota_pct_used is a catch-based proxy for cumulative yearly volume, not
    official port-level quota usage.
- storm is based on maximum weekly significant wave height (VHM0) and
    config.STORM_WAVE_THRESHOLD_M.

Fishing effort:
n_vessels is carried from catch_data_real.csv as the unique vessel count per
port x species x week. Missing legacy values are filled with 0.
dominant_gear is carried unchanged and missing values are filled with "unknown".

Incomplete or out-of-range years are removed using config.START_DATE/END_DATE.
Most importantly, filtering uses ISO "year", not calendar "date".
The reason is that ingest_fiskeridir.py calculates year and week with
pd.dt.isocalendar(), so year is ISO 8601 rather than calendar year. A late
December date can therefore belong to ISO week 1 of the following year. Date
filtering could admit such a row even when its ISO year was outside the
training and evaluation design. Filtering by ISO year keeps this module,
feature_selection.py, and train.py consistent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import STORM_WAVE_THRESHOLD_M, START_DATE, END_DATE  # noqa: E402


def compute_quota_proxy(catch: pd.DataFrame) -> pd.DataFrame:
    catch = catch.sort_values(["port", "species", "year", "week"]).copy()
    catch["cum_tons_ytd_before"] = (
        catch.groupby(["port", "species", "year"])["tons"].cumsum() - catch["tons"]
    )
    annual_total = catch.groupby(["port", "species", "year"])["tons"].transform("sum")
    catch["quota_pct_used"] = (catch["cum_tons_ytd_before"] / annual_total).clip(0, 1).fillna(0)
    return catch.drop(columns=["cum_tons_ytd_before"])


def main():
    catch_path = Path("data/catch_data_real.csv")
    ocean_path = Path("data/ocean_features.csv")

    if not catch_path.exists():
        print(f"MISSING {catch_path} - run src/ingest_fiskeridir.py first")
        return
    if not ocean_path.exists():
        print(f"MISSING {ocean_path} - run src/ingest_copernicus.py first")
        return

    catch = pd.read_csv(catch_path)
    ocean = pd.read_csv(ocean_path)

    has_effort = "n_vessels" in catch.columns
    has_gear = "dominant_gear" in catch.columns

    catch = compute_quota_proxy(catch)

    merged = catch.merge(ocean, on=["port", "year", "week"], how="left")

    # Fill missing ocean values, such as weeks without satellite coverage,
    # from the nearest available value within each port; use the global mean
    # if no value remains.
    merged = merged.sort_values(["port", "year", "week"])
    for col in ["sst_celsius", "chlorophyll_a", "wave_height_max"]:
        merged[col] = merged.groupby("port")[col].transform(lambda s: s.ffill().bfill())
        merged[col] = merged[col].fillna(merged[col].mean())

    merged["storm"] = (merged["wave_height_max"] > STORM_WAVE_THRESHOLD_M).astype(int)

    merged["date"] = pd.to_datetime(
        merged["year"].astype(str) + "-W" + merged["week"].astype(str).str.zfill(2) + "-1",
        format="%G-W%V-%u",
    )

    # --- Remove years outside [START_DATE, END_DATE] from config.py ---------
    # Filter by ISO "year", not calendar "date", to keep split labels aligned.
    n_before_range_filter = len(merged)
    start_year, end_year = pd.Timestamp(START_DATE).year, pd.Timestamp(END_DATE).year
    out_of_range = merged[(merged["year"] < start_year) | (merged["year"] > end_year)]
    if len(out_of_range):
        dropped_years = sorted(out_of_range["year"].unique().tolist())
        print(
            f"\nRemoving {len(out_of_range)} rows outside config.START_DATE/END_DATE "
            f"({start_year} .. {end_year}); ISO years: {dropped_years}."
        )
    merged = merged[(merged["year"] >= start_year) & (merged["year"] <= end_year)].copy()

    final_cols = ["year", "week", "port", "species", "tons", "quota_pct_used",
                  "sst_celsius", "chlorophyll_a", "wave_height_max", "storm", "date"]

    if has_effort:
        merged["n_vessels"] = merged["n_vessels"].fillna(0).astype(int)
        final_cols.append("n_vessels")
    if has_gear:
        merged["dominant_gear"] = merged["dominant_gear"].fillna("unknown")
        final_cols.append("dominant_gear")
    merged = merged[final_cols].sort_values(["port", "species", "year", "week"]).reset_index(drop=True)

    Path("data").mkdir(parents=True, exist_ok=True)
    merged.to_csv("data/raw_catch_data.csv", index=False)
    print(
        f"\nSaved {len(merged)} rows ({n_before_range_filter} before year filtering) "
        "-> data/raw_catch_data.csv"
    )
    print(merged.head(10))
    print(
        f"\nStorm weeks (VHM0 > {STORM_WAVE_THRESHOLD_M} m): "
        f"{int(merged['storm'].sum())} of {len(merged)} ({merged['storm'].mean()*100:.1f}%)"
    )
    print(f"Date range after filtering: {merged['date'].min().date()} .. {merged['date'].max().date()}")
    print(f"ISO year range after filtering: {sorted(merged['year'].unique().tolist())}")
    if not has_effort:
        print("\nWARNING: 'n_vessels' is missing from catch_data_real.csv; use the updated "
              "ingest_fiskeridir.py to include the effort data.")
    if not has_gear:
                print("WARNING: 'dominant_gear' is missing from catch_data_real.csv; use the updated "
              "ingest_fiskeridir.py to include gear data.")
    print("\nMissing values after merging (should be zero except source gaps):")
    print(merged.isna().sum())


if __name__ == "__main__":
    main()
