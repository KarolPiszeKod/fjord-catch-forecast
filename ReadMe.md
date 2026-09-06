# Fjord Catch Forecast (FCF)

FCF is a small forecasting project for weekly landed catch in Norwegian ports. It gives a median forecast plus a sensible uncertainty range for each port, species, and ISO week. It is mainly a place to explore historical forecasts and uncertainty, not a crystal ball for operational decisions. No, it will not tell you exactly how many tonnes will land next Tuesday, but it should give you a useful range and a reasonable guess.

## What's inside?

- Landing records from the Norwegian Fisheries Directorate.
- Ocean features from Copernicus Marine, including temperature, chlorophyll, and wave height.
- An XGBoost quantile model with q10, q50, and q90 predictions.
- Conformalized Quantile Regression (CQR) for calibrated 80% intervals.
- A Streamlit dashboard for exploring forecasts, errors, and uncertainty.
- Reports, experiment archives, and an optional Docker setup for running the dashboard locally.

## Quickstart: run locally in 5 minutes

Run these commands from the repository root.

1. Clone the repository and open its directory.
2. Create and activate a virtual environment if you want an isolated setup:

    ```powershell
    python -m venv .venv
    .venv\Scripts\activate
    ```

3. Install the dependencies:

    ```powershell
    pip install -r requirements.txt
    ```

4. Run the pipeline in order:

    ```powershell
    python src/ingest_fiskeridir.py
    python src/ingest_copernicus.py
    python src/merge_sources.py
    python src/features.py
    python src/feature_selection.py
    python src/train.py
    ```

5. Start the dashboard:

    ```powershell
    streamlit run app.py
    ```

The dashboard will be available at `http://localhost:8501`.

Raw data and generated models are local artifacts. They are ignored by Git, so every user needs to obtain or generate them locally. The Copernicus step may also require account configuration.

## Run with Docker (optional, but nice)

1. Install and start Docker Desktop.
2. From the repository root, run:

    ```powershell
    docker compose up --build
    ```
3. Open `http://localhost:8501`.
4. Stop the dashboard with:

    ```powershell
    docker compose down
    ```

Docker is an optional local deployment for the dashboard. Data and model files are not baked into the image; the Compose file mounts the local `data/` and `models/` folders at runtime. If you like containers, this is the project's happy place.

For a direct Docker command on Windows PowerShell:

```powershell
docker build -t fjord-catch-forecast .
docker run --rm -p 8501:8501 -v "${PWD}/data:/app/data" -v "${PWD}/models:/app/models" fjord-catch-forecast
```

For startup diagnostics:

```powershell
docker compose logs -f fcf
```

You still need to generate the data and model artifacts locally with the Python pipeline first. `.dockerignore` keeps raw data, generated datasets, model files, and secrets out of the image.

## What does the model actually do?

The model learns from historical data from 2014-2022. It uses 2023 as a calibration year and tests its forecasts on 2024-2025. For each port, species, and week, it predicts a median catch volume and an 80% uncertainty interval rather than one overly confident number.

The features include lagged catch volumes, seasonal averages, lagged fleet-effort signals such as `n_vessels_lag_1`, and ocean conditions. The target is transformed with `log1p(tons)` before training. CQR then widens the q10-q90 interval when needed so the reported coverage is better calibrated.

## How good is it?

The full-data v1 baseline currently reports:

| Metric | Result |
|---|---:|
| MAE | 249.57 t |
| MdAPE | 50.66% |
| Calibrated 80% coverage | 80.55% |

MAE is the average absolute error in tonnes. MdAPE describes the typical relative error, while coverage tells us how often actual landings fall inside the calibrated interval. It is not perfect, but it is a solid baseline for a first version.

The normal-segment experiment is reported separately because it is diagnostic, not deployable:

| Variant | MAE | MdAPE | Calibrated 80% coverage |
|---|---:|---:|---:|
| Normal-segment oracle experiment | 193.72 t | 34.83% | 77.05% |

## Known limitations (read this before you trust it too much)

- The `normal`, `weak`, and `high` catch segments were assigned after the actual outcome was known. The normal-only result is therefore an oracle experiment. A real deployment would need a pre-forecast gating classifier.
- The current ocean inputs are historical Copernicus reanalysis, not guaranteed future conditions. The dashboard's forward scenario carries selected recent conditions forward.
- `quota_pct_used` is a catch-based proxy, not official port-level quota data.
- The model cannot predict sudden events that do not appear in the historical data, such as an unexpected regulation change or a fleet disruption.
- MAPE can look very high when actual catch is close to zero. Use MAE and MdAPE alongside it.

For the full evaluation and discussion, see [`reports/final_evaluation.md`](reports/final_evaluation.md).

## Project structure

```text
FCF/
├── app.py                 # Streamlit dashboard
├── config.py              # Shared ports, species, years, and model constants
├── requirements.txt       # Python dependencies
├── Dockerfile             # Optional local dashboard image
├── docker-compose.yml     # Optional Docker Compose setup
├── run_pipeline.py        # Pipeline runner
├── scripts/               # Small project utilities
├── reports/               # Evaluation reports
├── data/                  # Local and generated data
├── models/                # Local model artifacts
└── src/                   # Ingestion, features, selection, and training
```

The main path is `src/` -> `models/` -> `app.py`. The `data/` and `models/` folders are deliberately local and are not part of the public Git workflow by default.

## Data & model artifacts

Expected local data locations include:

- `data/raw/fiskeridir/`: manually obtained Fisheries Directorate files.
- `data/ocean_features.csv`: prepared Copernicus features.
- `data/raw_catch_data.csv`: merged data used by feature engineering and training.

Important model files in `models/` include `xgb_quantile_model.json`, `feature_columns.json`, `metrics.json`, and `conformal_Q.json`. The `.gitignore` excludes raw data, generated CSV files, model JSON/CSV artifacts, experiment archives, and secrets. Generate them locally rather than expecting them to appear after cloning.

## What's next? (if someone feels like hacking on this)

- Add a pre-forecast anomaly or regime-gating classifier.
- Try a mixture-of-experts approach for weak, normal, and high-catch regimes.
- Add better fishing-ground or catch-location features when reliable source data is available.
- Replace historical reanalysis with operational forecast products.

There is plenty of room for improvement, but at least the first version knows when it does not know. That is already a respectable start.

## License & credits

FCF uses data from the Norwegian Fisheries Directorate and Copernicus Marine. Users must obtain the data themselves and check the applicable licenses, attribution requirements, and terms of use before using or redistributing it. This repository does not include raw licensed data or generated model artifacts by default.
