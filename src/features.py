"""
features.py
============
Feature engineering for Fjord Catch Forecast.

Key design rule: every feature must be known BEFORE the forecast week (zero
data leakage). Therefore:
- SST, chlorophyll, and wave_height_max represent the current week only because
    Copernicus Marine also provides "_anfc_" analysis+forecast products.
    This project trains on historical "_my_" reanalysis; a live deployment must
    switch ingest_copernicus.py to suitable "_anfc_" products.
- quota_pct_used is provided by merge_sources.py as a pre-catch value.
- lag_tons_{1..4} and seasonal_avg_hist use historical data only.
- Effort features (n_vessels_*, dominant_gear_*) are lagged or rolling so the
    current week's outcome is not exposed.

Features built:
- lag_tons_1..4: catch volume from the previous 1-4 weeks (per port x species).
- seasonal_avg_hist: historical average for the same calendar week, using
    earlier years only (expanding, no future leakage).
- roll_mean_4: lagged rolling average over the previous four weeks.
- quota_pct_used, ocean features, and storm: passed through from merge_sources.py.
- week_sin and week_cos: cyclic week encoding instead of raw 1-52.
- sild_week_sin and sild_week_cos: species_sild seasonal interactions.
- sild_lag_tons_{21..31}: seasonal sild lags from the previous year.
- n_vessels_lag_1: vessel count from the previous week.
- n_vessels_roll_mean_4: lagged four-week rolling average.
- n_vessels_vs_hist: deviation from the expanding historical weekly average.
- low_effort_flag: 1 when n_vessels is below half the historical average.
- dominant_gear_*: one-hot encoding for the most common gear categories.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["port", "species", "year", "week"]).reset_index(drop=True)

    group_cols = ["port", "species"]

    # --- Lag features (tons) ---------------------------------------------
    for lag in [1, 2, 3, 4]:
        df[f"lag_tons_{lag}"] = df.groupby(group_cols)["tons"].shift(lag)

    # --- Rolling mean on a lagged series to avoid the current tons value ----
    df["roll_mean_4"] = df.groupby(group_cols)["tons"].transform(
        lambda s: s.shift(1).rolling(window=4, min_periods=2).mean()
    )

    # --- Historical seasonal average using earlier years only --------------
    seasonal_lookup = (
        df.groupby(group_cols + ["week", "year"])["tons"].mean().reset_index()
    )

    seasonal_lookup = seasonal_lookup.sort_values(group_cols + ["week", "year"])
    seasonal_lookup["seasonal_avg_hist"] = seasonal_lookup.groupby(
        group_cols + ["week"]
    )["tons"].transform(lambda s: s.expanding().mean().shift(1))

    df = df.merge(
        seasonal_lookup[group_cols + ["week", "year", "seasonal_avg_hist"]],
        on=group_cols + ["week", "year"],
        how="left",
    )

    # --- Effort features: n_vessels ---------------------------------------
    # n_vessels_lag_1: vessel count from the previous week
    df["n_vessels_lag_1"] = df.groupby(group_cols)["n_vessels"].shift(1)

    # n_vessels_roll_mean_4: lagged four-week rolling average
    df["n_vessels_roll_mean_4"] = df.groupby(group_cols)["n_vessels"].transform(
        lambda s: s.shift(1).rolling(window=4, min_periods=2).mean()
    )

    # n_vessels_vs_hist: deviation from the historical average for this week
    vessels_seasonal_lookup = (
        df.groupby(group_cols + ["week", "year"])["n_vessels"].mean().reset_index()
    )
    vessels_seasonal_lookup = vessels_seasonal_lookup.sort_values(
        group_cols + ["week", "year"]
    )
    vessels_seasonal_lookup["n_vessels_seasonal_avg_hist"] = (
        vessels_seasonal_lookup.groupby(group_cols + ["week"])["n_vessels"]
        .transform(lambda s: s.expanding().mean().shift(1))
    )

    df = df.merge(
        vessels_seasonal_lookup[
            group_cols + ["week", "year", "n_vessels_seasonal_avg_hist"]
        ],
        on=group_cols + ["week", "year"],
        how="left",
    )

    df["n_vessels_vs_hist"] = (
        df["n_vessels"] - df["n_vessels_seasonal_avg_hist"]
    ) / df["n_vessels_seasonal_avg_hist"].replace(0, np.nan)

    # low_effort_flag: 1 when n_vessels is below half the historical average
    df["low_effort_flag"] = (
        df["n_vessels"] < 0.5 * df["n_vessels_seasonal_avg_hist"]
    ).astype(int)

    # --- Dominant gear (one-hot) -----------------------------------------
    # dominant_gear is a category string, for example "TRAWL" or "LINE".
    df = pd.get_dummies(df, columns=["dominant_gear"], prefix="gear")

    # --- Cyclic week encoding ----------------------------------------------
    df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52)

    # --- species_sild seasonal interactions -------------------------------
    # Only sild receives the seasonal values; other species receive zero.
    df["sild_week_sin"] = np.where(
        df["species"] == "sild",
        df["week_sin"],
        0
    )
    df["sild_week_cos"] = np.where(
        df["species"] == "sild",
        df["week_cos"],
        0
    )

    # --- Seasonal sild lags (weeks 21-31 from the previous year) -----------
    # These are 52-week lags for the same week in the previous year.
    # They are calculated only for sild; other species receive zero.
    for lag_week in range(21, 32):  # weeks 21-31
        lag_col = f"sild_lag_tons_{lag_week}"
        # First calculate the 52-week lag per port and species.
        df[f"_temp_lag_{lag_week}"] = df.groupby(group_cols)["tons"].shift(52)
        # Keep the lag only for sild; other species receive zero.
        df[lag_col] = np.where(
            df["species"] == "sild",
            df[f"_temp_lag_{lag_week}"],
            0
        )
        # Remove the temporary column.
        df = df.drop(columns=[f"_temp_lag_{lag_week}"])

    # --- One-hot encoding for port and species -----------------------------
    df = pd.get_dummies(df, columns=["port", "species"], prefix=["port", "species"])

    return df


FEATURE_COLUMNS = [
    "lag_tons_1",
    "lag_tons_2",
    "lag_tons_3",
    "lag_tons_4",
    "roll_mean_4",
    "seasonal_avg_hist",
    "quota_pct_used",
    "sst_celsius",
    "chlorophyll_a",
    "wave_height_max",
    "storm",
    "week_sin",
    "week_cos",
    # Seasonal features for sild
    "sild_week_sin",
    "sild_week_cos",
    "sild_lag_tons_21",
    "sild_lag_tons_22",
    "sild_lag_tons_23",
    "sild_lag_tons_24",
    "sild_lag_tons_25",
    "sild_lag_tons_26",
    "sild_lag_tons_27",
    "sild_lag_tons_28",
    "sild_lag_tons_29",
    "sild_lag_tons_30",
    "sild_lag_tons_31",
    # Effort features
    "n_vessels_lag_1",
    "n_vessels_roll_mean_4",
    "n_vessels_vs_hist",
    "low_effort_flag",
    # gear_* columns are added dynamically after one-hot encoding.
]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    dummy_cols = [c for c in df.columns if c.startswith("port_") or c.startswith("species_")]
    gear_cols = [c for c in df.columns if c.startswith("gear_")]
    return FEATURE_COLUMNS + dummy_cols + gear_cols


if __name__ == "__main__":
    raw = pd.read_csv("data/raw_catch_data.csv", parse_dates=["date"])
    feat = build_features(raw)
    n_before = len(feat)
    feat_clean = feat.dropna(subset=["lag_tons_4", "seasonal_avg_hist"])
    print(f"Rows before warm-up cleaning: {n_before}; after removing warm-up NaN: {len(feat_clean)}")
    feat.to_csv("data/features.csv", index=False)
    print("Saved data/features.csv")
    print("\nFeature columns:", get_feature_columns(feat))
    print(feat_clean.head())