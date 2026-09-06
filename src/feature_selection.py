"""
feature_selection.py
=====================
Boruta-lite: candidate feature selection with shadow features, repeated over
several rolling time windows (walk-forward) before train.py is run.

Why this is needed: tree gain/MDI importance is systematically biased toward
continuous variables with many unique values over discrete variables, whether
or not they carry real signal. With enough flexibility and a small sample, a
tree can find a nonlinear split in noise that happens to improve one training
window.

Algorithm for each fold:
1. Add independently shuffled copies of every feature as shadow_* columns.
2. Train a RandomForest on real and shadow features together.
3. A feature wins an iteration when its importance exceeds every shadow feature.
4. Repeat N_ITER times with a new shuffle each time.
5. A feature wins a fold when its iteration hit rate reaches the threshold.

Folds are WALK-FORWARD expanding windows and never include TEST_YEARS or
CALIB_YEARS from config.py.

THREE DECISION CATEGORIES (not only confirmed/rejected):
A feature is evaluated using two independent conditions:
  (a) wins every one of the BORUTA_RECENT_FOLDS_REQUIRED newest folds, a hard
      condition that the feature works now.
  (b) wins >= BORUTA_OVERALL_FOLD_CONFIRM_FRACTION of all folds, filtering out
      pure flukes.
Feature statuses:
    - "CONFIRMED": meets (a) and (b), saved to confirmed_features.json, and
        always used by train.py.
    - "CONDITIONALLY_CONFIRMED": meets (b) but not (a), saved to
        conditional_features.json, and used only when USE_CONDITIONAL_FEATURES is
        explicitly enabled.
    - "REJECTED": does not meet (b) and is indistinguishable from noise in most
        of the history.
Project example: n_vessels_lag_1 had fold_win_fraction=0.83 (>= 0.7 overall),
but lost one of the three newest folds, so it receives a conditional status
rather than being automatically rejected.

Source years are logged with ANSI colors and explicit categories: warm-up
(used for annual lags), excluded (TEST/CALIB), and Boruta pool.

Important Windows encoding note: output JSON files are opened explicitly with
encoding="utf-8" so port names containing "Ø" are written reliably.

Output:
- models/confirmed_features.json - confirmed features used by default.
- models/conditional_features.json - conditional features enabled explicitly.
- models/boruta_report.json - complete audit trail.

Run with: python src/feature_selection.py
Run it before train.py when using the confirmed feature set instead of the full
feature store.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from features import build_features, get_feature_columns  # noqa: E402
from config import (  # noqa: E402
    TEST_YEARS, CALIB_YEARS,
    BORUTA_MIN_TRAIN_YEARS, BORUTA_ITERATIONS, BORUTA_ITERATION_HIT_THRESHOLD,
    BORUTA_RECENT_FOLDS_REQUIRED, BORUTA_OVERALL_FOLD_CONFIRM_FRACTION, BORUTA_RF_PARAMS,
)

REPORT_PATH = Path("models/boruta_report.json")
CONFIRMED_PATH = Path("models/confirmed_features.json")
CONDITIONAL_PATH = Path("models/conditional_features.json")

# --- Kolory ANSI do konsoli (dziala w PowerShell 7+/Windows Terminal/bash) --
_GRAY = "\033[90m"
_ORANGE = "\033[38;5;208m"
_YELLOW = "\033[93m"
_GREEN = "\033[92m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

STATUS_CONFIRMED = "CONFIRMED"
STATUS_CONDITIONAL = "CONDITIONALLY_CONFIRMED"
STATUS_REJECTED = "REJECTED"


def _classify_years(all_source_years: list[int], warmup_years: list[int],
                     available_years: list[int]) -> dict[int, str]:
    classification = {}
    for y in sorted(all_source_years):
        if y in warmup_years:
            classification[y] = "warm-up"
        elif y in (TEST_YEARS + CALIB_YEARS):
            classification[y] = "excluded (TEST/CALIB)"
        elif y in available_years:
            classification[y] = "Boruta pool"
        else:
            classification[y] = "unknown category (check manually)"
    return classification


def _print_year_legend(classification: dict[int, str]) -> None:
    print(f"\n{_BOLD}Source-year classification:{_RESET}")
    for y, cat in classification.items():
        if cat == "warm-up":
            color = _GRAY
        elif cat.startswith("excluded"):
            color = _ORANGE
        elif cat == "Boruta pool":
            color = _GREEN
        else:
            color = ""
        print(f"  {color}{y}: {cat}{_RESET}")
    print(
        f"  {_GRAY}warm-up{_RESET} = used to calculate annual lags and not evaluated; "
        f"this is required for zero data leakage.\n"
        f"  {_ORANGE}excluded (TEST/CALIB){_RESET} = reserved in config.py and never seen by Boruta.\n"
        f"  {_GREEN}Boruta pool{_RESET} = used for walk-forward feature selection below.\n"
    )


def make_walk_forward_folds(years: list[int]) -> list[list[int]]:
    years = sorted(years)
    folds = []
    for end_idx in range(BORUTA_MIN_TRAIN_YEARS, len(years) + 1):
        folds.append(years[:end_idx])
    return folds


def run_boruta_fold(fold_df: pd.DataFrame, feature_cols: list[str], seed_base: int) -> dict:
    X = fold_df[feature_cols]
    y = fold_df["tons"]
    hits = {c: 0 for c in feature_cols}

    for it in range(BORUTA_ITERATIONS):
        rng = np.random.RandomState(seed_base * 1000 + it)
        shadow = pd.DataFrame(
            {f"shadow_{c}": rng.permutation(X[c].values) for c in feature_cols},
            index=X.index,
        )
        combined = pd.concat([X, shadow], axis=1)

        model = RandomForestRegressor(random_state=seed_base * 1000 + it, **BORUTA_RF_PARAMS)
        model.fit(combined, y)

        importance = pd.Series(model.feature_importances_, index=combined.columns)
        shadow_cols = [c for c in combined.columns if c.startswith("shadow_")]
        shadow_max = importance[shadow_cols].max()

        for c in feature_cols:
            if importance[c] > shadow_max:
                hits[c] += 1

    return {c: hits[c] / BORUTA_ITERATIONS for c in feature_cols}


def main():
    data_path = Path("data/raw_catch_data.csv")
    if not data_path.exists():
        print(f"MISSING {data_path} - run the ingestion pipeline and merge_sources.py first.")
        return

    raw = pd.read_csv(data_path, parse_dates=["date"])
    all_source_years = sorted(raw["year"].unique().tolist())

    feat = build_features(raw)
    feat_before_dropna_years = sorted(feat["year"].unique().tolist())
    feat = feat.dropna(subset=["lag_tons_4", "lag_tons_52", "lag_tons_104", "seasonal_avg_hist"]).reset_index(drop=True)
    feat_after_dropna_years = sorted(feat["year"].unique().tolist())
    warmup_years = [y for y in feat_before_dropna_years if y not in feat_after_dropna_years]

    pool = feat[~feat["year"].isin(TEST_YEARS + CALIB_YEARS)].copy()
    available_years = sorted(pool["year"].unique().tolist())

    year_classification = _classify_years(all_source_years, warmup_years, available_years)
    _print_year_legend(year_classification)

    if len(available_years) < BORUTA_MIN_TRAIN_YEARS + 1:
        print(
            f"{_ORANGE}Too few years outside TEST_YEARS/CALIB_YEARS ({available_years}) "
            f"for a meaningful walk-forward run (need >= {BORUTA_MIN_TRAIN_YEARS + 1}). "
            f"Skipping selection; train.py will use the full feature store.{_RESET}"
        )
        return

    feature_cols = get_feature_columns(feat)
    folds = make_walk_forward_folds(available_years)
    print(f"{_GREEN}Years in Boruta pool: {available_years}{_RESET}")
    print(f"Number of walk-forward folds: {len(folds)} (expanding window, minimum {BORUTA_MIN_TRAIN_YEARS} years)")
    print(f"Candidates to test: {len(feature_cols)}")
    print(f"Shuffle iterations per fold: {BORUTA_ITERATIONS}\n")

    per_fold_results = []
    for i, fold_years in enumerate(folds):
        fold_df = pool[pool["year"].isin(fold_years)]
        print(f"Fold {i+1}/{len(folds)}: years {fold_years[0]}-{fold_years[-1]} ({len(fold_df)} rows)...")
        hit_rates = run_boruta_fold(fold_df, feature_cols, seed_base=i + 1)
        won_fold = {c: hit_rates[c] >= BORUTA_ITERATION_HIT_THRESHOLD for c in feature_cols}
        per_fold_results.append({"years": fold_years, "hit_rates": hit_rates, "won_fold": won_fold})

    n_folds = len(per_fold_results)
    n_recent = min(BORUTA_RECENT_FOLDS_REQUIRED, n_folds)
    recent_folds = per_fold_results[-n_recent:]

    global_report = {}
    confirmed = []
    conditional = []
    for c in feature_cols:
        fold_wins = sum(1 for f in per_fold_results if f["won_fold"][c])
        fold_win_fraction = fold_wins / n_folds
        won_all_recent = all(f["won_fold"][c] for f in recent_folds)
        meets_overall = fold_win_fraction >= BORUTA_OVERALL_FOLD_CONFIRM_FRACTION

        if won_all_recent and meets_overall:
            status = STATUS_CONFIRMED
            confirmed.append(c)
        elif meets_overall and not won_all_recent:
            status = STATUS_CONDITIONAL
            conditional.append(c)
        else:
            status = STATUS_REJECTED

        global_report[c] = {
            "fold_win_fraction": round(fold_win_fraction, 3),
            "won_all_recent_folds": won_all_recent,
            "meets_overall_threshold": meets_overall,
            "fold_hit_rates": [round(f["hit_rates"][c], 3) for f in per_fold_results],
            "status": status,
        }

    print(f"\n{_BOLD}--- Boruta result (fraction of folds where a feature beat noise) ---{_RESET}")
    for c, r in sorted(global_report.items(), key=lambda kv: -kv[1]["fold_win_fraction"]):
        status = r["status"]
        color = {STATUS_CONFIRMED: _GREEN, STATUS_CONDITIONAL: _YELLOW, STATUS_REJECTED: _ORANGE}[status]
        print(f"  {color}{c:30s} fold_win_fraction={r['fold_win_fraction']:.2f}  [{status}]{_RESET}")

    n_rejected = len(feature_cols) - len(confirmed) - len(conditional)
    print(
        f"\nConfirmed: {len(confirmed)}/{len(feature_cols)}  "
        f"Conditionally confirmed: {len(conditional)}/{len(feature_cols)}  "
        f"Rejected: {n_rejected}/{len(feature_cols)}"
    )
    if conditional:
        print(
            f"\n{_YELLOW}Conditionally confirmed features (signal in most history, but not all "
            f"of the {n_recent} newest folds): {conditional}{_RESET}"
        )
        print(
            f"{_YELLOW}They are not enabled by default. Set USE_CONDITIONAL_FEATURES = True "
            f"in train.py to compare models.{_RESET}"
        )

    Path("models").mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "all_source_years": all_source_years,
                "year_classification": year_classification,
                "warmup_years_used_for_lags_only": warmup_years,
                "test_years_excluded": TEST_YEARS,
                "calib_years_excluded": CALIB_YEARS,
                "available_years": available_years,
                "n_folds": n_folds,
                "iteration_hit_threshold": BORUTA_ITERATION_HIT_THRESHOLD,
                "recent_folds_required": BORUTA_RECENT_FOLDS_REQUIRED,
                "overall_fold_confirm_fraction": BORUTA_OVERALL_FOLD_CONFIRM_FRACTION,
                "features": global_report,
            },
            f, indent=2, ensure_ascii=False,
        )
    with open(CONFIRMED_PATH, "w", encoding="utf-8") as f:
        json.dump(confirmed, f, indent=2, ensure_ascii=False)
    with open(CONDITIONAL_PATH, "w", encoding="utf-8") as f:
        json.dump(conditional, f, indent=2, ensure_ascii=False)

        print(f"\nSaved: {REPORT_PATH} (full audit trail), {CONFIRMED_PATH} (confirmed features), "
            f"{CONDITIONAL_PATH} (conditional features, manually enabled)")
    if n_rejected:
        rejected = [c for c, r in global_report.items() if r["status"] == STATUS_REJECTED]
        print(f"{_ORANGE}Rejected (not distinguishable from random noise on these data): {rejected}{_RESET}")


if __name__ == "__main__":
    main()
