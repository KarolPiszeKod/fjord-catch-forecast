"""
ingest_fiskeridir.py
====================
Parses CSV/XLSX files manually downloaded from the Fisheries Directorate open
data site and aggregates them to weekly catch volume per port, species, year,
and week.

Output: data/catch_data_real.csv
Columns: year, week, port, species, tons, n_vessels, dominant_gear
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PORTS, SPECIES_FILTER  # noqa: E402


try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


RAW_DIR = Path("data/raw/fiskeridir")
OUT_PATH = Path("data/catch_data_real.csv")
SOURCE_UNIT_IS_KG = True

COLUMN_MAP: dict[str, str | None] = {
    "tons": "Rundvekt",
    "species": "Art - FDIR",
    "port": "Landingskommune",
    "date": "Landingsdato",
    "vessel_id": "Fartøy ID",
    "gear": "Redskap",
}

CANDIDATE_KEYWORDS = {
    "tons": ["rundvekt", "produktvekt"],
    "species": ["art - fdir", "art fao", "hovedart"],
    "port": ["landingskommune", "lossehavn"],
    "date": ["landingsdato", "siste fangstdato"],
    "vessel_id": ["fartøy id", "fartoy id", "fartøyid", "fartoyid"],
    "gear": ["redskap"],
}

PORT_FILTER = [str(port).strip().upper() for port in PORTS]


def _progress_iter(items, desc: str):
    if _HAS_TQDM:
        yield from tqdm(items, desc=desc, unit="file")
        return

    total = len(items)
    bar_width = 30
    for i, item in enumerate(items, start=1):
        filled = int(bar_width * i / total)
        bar = "#" * filled + "-" * (bar_width - filled)
        print(f"\r{desc} [{bar}] {i}/{total}", end="", flush=True)
        yield item
    print()


def _load_one_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(
                path,
                sep=";",
                decimal=",",
                encoding="utf-8",
                low_memory=False,
                on_bad_lines="skip",
            )
            if df.shape[1] <= 1:
                raise ValueError("Invalid separator")
        except Exception:
            df = pd.read_csv(
                path,
                sep=",",
                encoding="utf-8",
                low_memory=False,
                on_bad_lines="skip",
            )
    elif path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")
    return df


def _auto_detect_column(columns: list[str], key: str) -> str | None:
    for column in columns:
        low = str(column).lower()
        if any(keyword in low for keyword in CANDIDATE_KEYWORDS[key]):
            return column
    return None


def resolve_columns(df: pd.DataFrame) -> dict[str, str | None]:
    resolved: dict[str, str | None] = {}

    for key, manual_value in COLUMN_MAP.items():
        if manual_value in df.columns:
            resolved[key] = manual_value
            continue

        print(
            f"  WARNING: column '{manual_value}' does not exist for key '{key}'; "
            "trying automatic detection."
        )
        detected = _auto_detect_column(list(df.columns), key)

        if detected is not None:
            print(f"  [automatic detection] '{key}' -> '{detected}'")
            resolved[key] = detected
        elif key in ("vessel_id", "gear"):
            print(f"  WARNING: optional column for '{key}' is missing. Values will be missing.")
            resolved[key] = None
        else:
            raise ValueError(
                f"Could not identify the required column for '{key}'. "
                f"Available columns: {list(df.columns)}."
            )

    return resolved


def classify_species(raw_value: object) -> str | None:
    if not isinstance(raw_value, str):
        return None

    low = raw_value.lower()
    for canonical, variants in SPECIES_FILTER.items():
        if any(str(variant).lower() in low for variant in variants):
            return canonical
    return None


def process_file(path: Path) -> pd.DataFrame:
    print(f"\nProcessing {path.name}...")
    source = _load_one_file(path)
    cols = resolve_columns(source)

    # Select columns first, then call .copy() to keep the operation explicit.
    out = source[[
        cols["tons"],
        cols["species"],
        cols["port"],
        cols["date"],
    ]].copy()
    out.columns = ["tons_raw", "species_raw", "port", "date"]

    vessel_col = cols.get("vessel_id")
    gear_col = cols.get("gear")
    out["vessel_id"] = source[vessel_col] if vessel_col is not None else pd.NA
    out["gear"] = source[gear_col] if gear_col is not None else pd.NA

    out["date"] = pd.to_datetime(out["date"], errors="coerce", dayfirst=True)
    out = out.dropna(subset=["date"])

    out["species"] = out["species_raw"].apply(classify_species)
    out = out[out["species"].notna()].copy()

    out = out[out["port"].notna()].copy()
    out["port"] = out["port"].astype(str).str.strip().str.upper()

    unmatched_ports = sorted(
        set(out.loc[~out["port"].isin(PORT_FILTER), "port"].dropna().astype(str))
    )
    if unmatched_ports:
        print(
            f"  Notice: PORT_FILTER did not match {len(unmatched_ports)} unique port values "
            f"(first 10): {unmatched_ports[:10]}"
        )

    out = out[out["port"].isin(PORT_FILTER)].copy()
    out["tons_raw"] = pd.to_numeric(out["tons_raw"], errors="coerce")
    out = out.dropna(subset=["tons_raw"])
    out["tons"] = out["tons_raw"] / 1000.0 if SOURCE_UNIT_IS_KG else out["tons_raw"]

    iso = out["date"].dt.isocalendar()
    out["year"] = iso["year"].astype(int)
    out["week"] = iso["week"].astype(int)

    out["vessel_id"] = out["vessel_id"].astype("string").str.strip()
    out["gear"] = out["gear"].astype("string").str.strip()
    out.loc[out["vessel_id"] == "", "vessel_id"] = pd.NA
    out.loc[out["gear"] == "", "gear"] = pd.NA

    print(f"  Rows after filtering: {len(out)}")
    if out.empty:
        print(
            "  WARNING: no rows passed the filters; check COLUMN_MAP, "
            "SPECIES_FILTER, and PORT_FILTER in config.py."
        )

    return out[["year", "week", "port", "species", "tons", "vessel_id", "gear"]]


def _dominant_value(values: pd.Series) -> object:
    values = values.dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        return pd.NA

    counts = values.value_counts()
    max_count = counts.max()
    return sorted(counts[counts == max_count].index)[0]


def _count_unique_vessels(values: pd.Series) -> int:
    values = values.dropna().astype(str).str.strip()
    values = values[values != ""]
    return int(values.nunique())


def main() -> None:
    files = (
        list(RAW_DIR.glob("*.csv"))
        + list(RAW_DIR.glob("*.xlsx"))
        + list(RAW_DIR.glob("*.xls"))
    )

    if not files:
        print(f"NO FILES found in {RAW_DIR}. Download them manually from the Fisheries Directorate.")
        return

    files = sorted(files)
    if not _HAS_TQDM:
        print("(tip: pip install tqdm provides a progress bar with ETA)\n")

    all_rows: list[pd.DataFrame] = []
    for file_path in _progress_iter(files, desc="Processing files"):
        try:
            all_rows.append(process_file(file_path))
        except (ValueError, KeyError, UnicodeError) as exc:
            print(f"  ERROR in {file_path.name}: {exc}")

    if not all_rows:
        print("\nNo file could be processed; check COLUMN_MAP.")
        return

    combined = pd.concat(all_rows, ignore_index=True)
    for column in ("vessel_id", "gear"):
        if column not in combined.columns:
            combined[column] = pd.NA

    group_cols = ["year", "week", "port", "species"]
    weekly = (
        combined.groupby(group_cols, dropna=False)
        .agg(
            tons=("tons", "sum"),
            n_vessels=("vessel_id", _count_unique_vessels),
            dominant_gear=("gear", _dominant_value),
        )
        .reset_index()
    )

    weekly["n_vessels"] = weekly["n_vessels"].fillna(0).astype(int)
    weekly["dominant_gear"] = weekly["dominant_gear"].astype("string")

    Path("data").mkdir(parents=True, exist_ok=True)
    weekly.to_csv(OUT_PATH, index=False, encoding="utf-8")

    print(f"\nSaved {len(weekly)} rows (port x species x year x week) -> {OUT_PATH}")
    print("\nPreview:")
    print(weekly.head(10).to_string(index=False))
    print("\nZakres lat:", sorted(weekly["year"].unique()))
    print("Ports:", sorted(weekly["port"].unique()))
    print("Species:", sorted(weekly["species"].unique()))
    print(f"Rows with n_vessels > 0: {int((weekly['n_vessels'] > 0).sum())}/{len(weekly)}")
    print(f"Rows with dominant_gear: {int(weekly['dominant_gear'].notna().sum())}/{len(weekly)}")

    missing_ports = set(PORT_FILTER) - set(weekly["port"].unique())
    if missing_ports:
        print(
            f"\nWARNING: no data for ports: {sorted(missing_ports)}; "
            "the model will operate only for ports with data."
        )


if __name__ == "__main__":
    main()