"""
train.py
========
Training a weekly catch-volume forecast (port x species) with uncertainty
intervals using XGBoost quantile regression.

    The model trains on log1p(tons) and converts predictions back to tonnes.
    The split is chronological: train < calibration < test.

Note:
- USE_CONDITIONAL_FEATURES enables features conditionally confirmed by Boruta,
    currently primarily n_vessels_lag_1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_pinball_loss


sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from features import build_features, get_feature_columns  # noqa: E402
from config import CALIB_YEARS, QUANTILES, TARGET_COVERAGE, TEST_YEARS  # noqa: E402


# -------------------------------------------------------------------------
# EXPERIMENT: conditionally confirmed Boruta features
# -------------------------------------------------------------------------
# False = use only confirmed features from models/confirmed_features.json.
# True  = add features from models/conditional_features.json, for example
#         n_vessels_lag_1 (0.83 fold_win_fraction).
#
# Set this explicitly when comparing models with and without conditional features.
USE_CONDITIONAL_FEATURES = True

# Production v1 trains on the full merged dataset. Conditional features remain
# enabled because they add the Boruta-confirmed effort lag n_vessels_lag_1 when
# the selection files are available.
DATA_PATH = Path("data/raw_catch_data.csv")


# -------------------------------------------------------------------------
# Data split
# -------------------------------------------------------------------------
def time_based_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return three disjoint chronological sets: train < calibration < test."""
    train = df[~df["year"].isin(TEST_YEARS + CALIB_YEARS)].copy()
    calib = df[df["year"].isin(CALIB_YEARS)].copy()
    test = df[df["year"].isin(TEST_YEARS)].copy()
    return train, calib, test


# -------------------------------------------------------------------------
# XGBoost: quantile regression
# -------------------------------------------------------------------------
def train_xgb_quantile(
    X_train: pd.DataFrame,
    y_train_log: np.ndarray,
    X_valid: pd.DataFrame | None = None,
    y_valid_log: np.ndarray | None = None,
) -> xgb.Booster:
    """Train one XGBoost model for the quantiles in QUANTILES."""
    train_matrix = xgb.QuantileDMatrix(X_train, y_train_log)
    evals: list[tuple[xgb.QuantileDMatrix, str]] = [(train_matrix, "train")]

    if X_valid is not None and y_valid_log is not None:
        valid_matrix = xgb.QuantileDMatrix(X_valid, y_valid_log, ref=train_matrix)
        evals.append((valid_matrix, "valid"))

    params = {
        "objective": "reg:quantileerror",
        "tree_method": "hist",
        "quantile_alpha": QUANTILES,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_child_weight": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5.0,
        "seed": 42,
    }

    booster = xgb.train(
        params,
        train_matrix,
        num_boost_round=500,
        evals=evals if X_valid is not None else [],
        early_stopping_rounds=30 if X_valid is not None else None,
        verbose_eval=False,
    )

    if X_valid is not None:
        print(
            f"  Early stopping: used {booster.best_iteration + 1}/500 rounds "
            f"(best_score={booster.best_score:.4f})"
        )
        if booster.best_iteration >= 490:
            print(
                "  WARNING: the model used almost all 500 rounds. "
                "Consider stronger regularization if overfitting is observed."
            )

    return booster


def predict_quantiles(booster: xgb.Booster, X: pd.DataFrame) -> dict[float, np.ndarray]:
    """Return quantile predictions in tonnes rather than log1p scale."""
    predictions_log = booster.inplace_predict(X)
    predictions_log = np.asarray(predictions_log)

    if predictions_log.ndim == 1:
        predictions_log = predictions_log.reshape(-1, 1)

    # Protect against axis ordering differences between XGBoost versions.
    if predictions_log.shape[0] == len(QUANTILES) and predictions_log.shape[1] != len(QUANTILES):
        predictions_log = predictions_log.T

    predictions_tons = np.expm1(predictions_log)
    predictions_tons = np.clip(predictions_tons, 0, None)

    return {quantile: predictions_tons[:, i] for i, quantile in enumerate(QUANTILES)}


# -------------------------------------------------------------------------
# Conformalized Quantile Regression (CQR)
# -------------------------------------------------------------------------
def conformalize_per_species(
    feature_df: pd.DataFrame,
    predictions: dict[float, np.ndarray],
    y_true: np.ndarray,
    target_coverage: float = TARGET_COVERAGE,
) -> dict[str, float]:
    """Calculate a per-species CQR correction with a global fallback."""
    lower = predictions[0.1]
    upper = predictions[0.9]
    conformity = np.maximum(lower - y_true, y_true - upper)

    result: dict[str, float] = {}
    species_columns = [column for column in feature_df.columns if column.startswith("species_")]

    for column in species_columns:
        mask = feature_df[column].astype(bool).to_numpy()
        if mask.sum() >= 5:
            species_name = column.replace("species_", "")
            result[species_name] = float(np.quantile(conformity[mask], target_coverage))

    result["_global"] = float(np.quantile(conformity, target_coverage))
    return result


def apply_conformal(
    feature_df: pd.DataFrame,
    predictions: dict[float, np.ndarray],
    conformal_by_species: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Expand q10-q90 by the appropriate species-level CQR correction."""
    correction = np.full(len(feature_df), conformal_by_species["_global"], dtype=float)
    species_columns = [column for column in feature_df.columns if column.startswith("species_")]

    for column in species_columns:
        species_name = column.replace("species_", "")
        if species_name in conformal_by_species:
            mask = feature_df[column].astype(bool).to_numpy()
            correction[mask] = conformal_by_species[species_name]

    lower = np.clip(predictions[0.1] - correction, 0, None)
    upper = predictions[0.9] + correction
    return lower, upper


# -------------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------------
def evaluate(
    y_true: np.ndarray,
    lower_calibrated: np.ndarray,
    upper_calibrated: np.ndarray,
    predictions: dict[float, np.ndarray],
) -> dict[str, float | int]:
    """Calculate global tonne-scale metrics; pinball loss remains on log scale."""
    metrics: dict[str, float | int] = {}
    y_log = np.log1p(y_true)

    for quantile in QUANTILES:
        pred_log = np.log1p(predictions[quantile])
        metrics[f"pinball_loss_log_q{int(quantile * 100)}"] = float(
            mean_pinball_loss(y_log, pred_log, alpha=quantile)
        )

    metrics["mae_median_tons"] = float(mean_absolute_error(y_true, predictions[0.5]))

    # MAPE is unstable near zero, so calculate it only for actual catch >= 1 tonne.
    mape_min_tons = 1.0
    mask = y_true >= mape_min_tons
    metrics["mape_median_pct_n_rows"] = int(mask.sum())
    metrics["mape_excluded_near_zero_rows"] = int((~mask).sum())

    if mask.sum() > 0:
        absolute_percentage_error = np.abs(y_true[mask] - predictions[0.5][mask]) / y_true[mask]
        metrics["mape_median_pct"] = float(np.mean(absolute_percentage_error) * 100)
        metrics["mdape_median_pct"] = float(np.median(absolute_percentage_error) * 100)

    lower_raw = predictions[0.1]
    upper_raw = predictions[0.9]
    metrics["coverage_80pct_interval_raw"] = float(
        np.mean((y_true >= lower_raw) & (y_true <= upper_raw))
    )
    metrics["mean_interval_width_raw_tons"] = float(np.mean(upper_raw - lower_raw))
    metrics["median_interval_width_raw_tons"] = float(np.median(upper_raw - lower_raw))

    metrics["coverage_80pct_interval_calibrated"] = float(
        np.mean((y_true >= lower_calibrated) & (y_true <= upper_calibrated))
    )
    metrics["mean_interval_width_calibrated_tons"] = float(
        np.mean(upper_calibrated - lower_calibrated)
    )
    metrics["median_interval_width_calibrated_tons"] = float(
        np.median(upper_calibrated - lower_calibrated)
    )

    return metrics


def evaluate_per_segment(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    lower_calibrated: np.ndarray,
    upper_calibrated: np.ndarray,
    predictions: dict[float, np.ndarray],
    mape_min_tons: float = 1.0,
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    """Calculate metrics separately per species and port."""
    output: dict[str, dict[str, dict[str, float | int | None]]] = {}

    for group_name, prefix in [("species", "species_"), ("port", "port_")]:
        group_output: dict[str, dict[str, float | int | None]] = {}
        columns = [column for column in test_df.columns if column.startswith(prefix)]

        for column in columns:
            name = column.replace(prefix, "")
            mask = test_df[column].astype(bool).to_numpy()
            if mask.sum() == 0:
                continue

            y_segment = y_true[mask]
            pred_segment = predictions[0.5][mask]
            lower_segment = lower_calibrated[mask]
            upper_segment = upper_calibrated[mask]

            valid_mape = y_segment >= mape_min_tons
            if valid_mape.sum() > 0:
                ape = np.abs(y_segment[valid_mape] - pred_segment[valid_mape]) / y_segment[valid_mape]
                mape = float(np.mean(ape) * 100)
                mdape = float(np.median(ape) * 100)
            else:
                mape = None
                mdape = None

            group_output[name] = {
                "n": int(mask.sum()),
                "median_actual_tons": float(np.median(y_segment)),
                "mae_tons": float(mean_absolute_error(y_segment, pred_segment)),
                "mape_pct": mape,
                "mdape_pct": mdape,
                "n_excluded_near_zero": int((~valid_mape).sum()),
                "coverage_80pct": float(
                    np.mean((y_segment >= lower_segment) & (y_segment <= upper_segment))
                ),
            }

        output[group_name] = group_output

    return output


# -------------------------------------------------------------------------
# Boruta features
# -------------------------------------------------------------------------
def select_feature_columns(feat: pd.DataFrame) -> list[str]:
    """Return confirmed Boruta features and optional conditional features."""
    all_feature_columns = get_feature_columns(feat)
    confirmed_path = Path("models/confirmed_features.json")
    conditional_path = Path("models/conditional_features.json")

    if not confirmed_path.exists():
        print("WARNING: models/confirmed_features.json is missing; using the full feature store.")
        return all_feature_columns

    with open(confirmed_path, encoding="utf-8") as file:
        confirmed_data = json.load(file)

    # Support both safe formats: a JSON list or a dictionary with a features key.
    if isinstance(confirmed_data, list):
        confirmed = confirmed_data
    else:
        confirmed = confirmed_data.get("features", confirmed_data.get("confirmed", []))

    feature_columns = [column for column in all_feature_columns if column in confirmed]
    print(f"Boruta: using {len(feature_columns)} confirmed features.")

    conditional_to_add: list[str] = []
    if USE_CONDITIONAL_FEATURES:
        if conditional_path.exists():
            with open(conditional_path, encoding="utf-8") as file:
                conditional_data = json.load(file)

            if isinstance(conditional_data, list):
                conditional = conditional_data
            else:
                conditional = conditional_data.get("features", conditional_data.get("conditional", []))

            conditional_to_add = [
                column
                for column in conditional
                if column in all_feature_columns and column not in feature_columns
            ]
            feature_columns.extend(conditional_to_add)
            print(f"Boruta: added {len(conditional_to_add)} conditional features.")
            print(f"         Enabled conditional features: {conditional_to_add}")
        else:
            print("WARNING: USE_CONDITIONAL_FEATURES=True, but models/conditional_features.json is missing.")

    excluded = [column for column in all_feature_columns if column not in feature_columns]
    if excluded:
        print(f"         excluded: {excluded}")

    return feature_columns


# -------------------------------------------------------------------------
# Main program
# -------------------------------------------------------------------------
def main() -> None:
    data_path = DATA_PATH
    if not data_path.exists():
        print(
            f"MISSING {data_path} - run ingest_fiskeridir.py, "
            "ingest_copernicus.py, and then merge_sources.py first."
        )
        return

    raw = pd.read_csv(data_path, parse_dates=["date"])
    feat = build_features(raw)
    feat = feat.dropna(subset=["lag_tons_4", "seasonal_avg_hist"]).reset_index(drop=True)

    if feat.empty:
        print("No data remains after removing warm-up rows.")
        return

    feature_columns = select_feature_columns(feat)
    if not feature_columns:
        print("No features remain for training after Boruta selection.")
        return

    train_df, calib_df, test_df = time_based_split(feat)
    print(f"Train: {len(train_df)} rows (years {sorted(train_df['year'].unique())})")
    print(f"Calibration: {len(calib_df)} rows (years {sorted(calib_df['year'].unique())})")
    print(f"Test:  {len(test_df)} rows (years {sorted(test_df['year'].unique())})")

    if train_df.empty or calib_df.empty or test_df.empty:
        print("\nWARNING: one of the splits is empty. Check TEST_YEARS/CALIB_YEARS in config.py.")
        return

    X_train = train_df[feature_columns]
    y_train = train_df["tons"].to_numpy()
    X_calib = calib_df[feature_columns]
    y_calib = calib_df["tons"].to_numpy()
    X_test = test_df[feature_columns]
    y_test = test_df["tons"].to_numpy()

    y_train_log = np.log1p(y_train)

    # Validation is the last 15% of training data, never a random sample.
    split_point = int(len(train_df) * 0.85)
    X_fit = X_train.iloc[:split_point]
    y_fit_log = y_train_log[:split_point]
    X_valid = X_train.iloc[split_point:]
    y_valid_log = y_train_log[split_point:]

    booster = train_xgb_quantile(X_fit, y_fit_log, X_valid, y_valid_log)

    predictions_calib = predict_quantiles(booster, X_calib)
    conformal_by_species = conformalize_per_species(calib_df, predictions_calib, y_calib)
    print(f"\nCQR correction per species (tonnes): {conformal_by_species}")

    predictions_test = predict_quantiles(booster, X_test)
    lower_calibrated, upper_calibrated = apply_conformal(
        test_df, predictions_test, conformal_by_species
    )

    metrics = evaluate(y_test, lower_calibrated, upper_calibrated, predictions_test)
    print("\n--- Test-set metrics (tonne scale, not log scale) ---")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: {value}")

    segment_metrics = evaluate_per_segment(
        test_df,
        y_test,
        lower_calibrated,
        upper_calibrated,
        predictions_test,
    )

    print("\n--- Metrics per species ---")
    for name, metric in segment_metrics["species"].items():
        mape_text = f"{metric['mape_pct']:6.1f}%" if metric["mape_pct"] is not None else "  n/a "
        mdape_text = f"{metric['mdape_pct']:6.1f}%" if metric["mdape_pct"] is not None else "  n/a "
        print(
            f"  {name:10s} n={metric['n']:4d}  actual_median={metric['median_actual_tons']:8.2f}t  "
            f"MAE={metric['mae_tons']:8.2f}t  MAPE={mape_text}  MdAPE={mdape_text}  "
            f"coverage80%={metric['coverage_80pct'] * 100:5.1f}%"
        )

    print("\n--- Metrics per port ---")
    for name, metric in segment_metrics["port"].items():
        mape_text = f"{metric['mape_pct']:6.1f}%" if metric["mape_pct"] is not None else "  n/a "
        mdape_text = f"{metric['mdape_pct']:6.1f}%" if metric["mdape_pct"] is not None else "  n/a "
        print(
            f"  {name:10s} n={metric['n']:4d}  actual_median={metric['median_actual_tons']:8.2f}t  "
            f"MAE={metric['mae_tons']:8.2f}t  MAPE={mape_text}  MdAPE={mdape_text}  "
            f"coverage80%={metric['coverage_80pct'] * 100:5.1f}%"
        )

    importance = pd.Series(booster.get_score(importance_type="gain")).sort_values(ascending=False)
    print("\n--- Top 8 features by gain importance (log space) ---")
    print(importance.head(8))

    Path("models").mkdir(parents=True, exist_ok=True)
    model_suffix = "_with_conditional" if USE_CONDITIONAL_FEATURES else ""

    # Zawsze zapisujemy podstawowe nazwy, aby app.py mogl dzialac bez zmian.
    booster.save_model("models/xgb_quantile_model.json")
    with open("models/feature_columns.json", "w", encoding="utf-8") as file:
        json.dump(feature_columns, file, ensure_ascii=False, indent=2)
    with open("models/metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    with open("models/metrics_by_segment.json", "w", encoding="utf-8") as file:
        json.dump(segment_metrics, file, ensure_ascii=False, indent=2)
    with open("models/conformal_Q.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "conformal_by_species": conformal_by_species,
                "target_is_log1p": True,
                "use_conditional_features": USE_CONDITIONAL_FEATURES,
                "feature_columns": feature_columns,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    importance.to_csv("models/feature_importance.csv", header=["gain"])

    # Save an additional experiment copy so a later run does not overwrite it.
    if model_suffix:
        booster.save_model(f"models/xgb_quantile_model{model_suffix}.json")
        with open(f"models/metrics{model_suffix}.json", "w", encoding="utf-8") as file:
            json.dump(metrics, file, ensure_ascii=False, indent=2)
        importance.to_csv(f"models/feature_importance{model_suffix}.csv", header=["gain"])

    print("\nSaved: models/xgb_quantile_model.json, feature_columns.json, metrics.json,")
    print("       metrics_by_segment.json, conformal_Q.json, feature_importance.csv")
    if model_suffix:
        print("Additional copies with conditional features saved in models/*_with_conditional.*")


if __name__ == "__main__":
    main()
