"""
app.py
======
 Streamlit dashboard for Fjord Catch Forecast.

Shows:
1. Actual catch versus the median forecast with a calibrated 80% interval.
2. Key performance metrics and the CQR correction.
3. Top features by gain importance.
4. A 1-3 week recursive forecast based on the latest available observations.

Local usage:
    pip install -r requirements.txt
    streamlit run app.py

 Requires these steps to be run once beforehand:
    python src/ingest_fiskeridir.py
    python src/ingest_copernicus.py
    python src/merge_sources.py
    python src/train.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from features import build_features  # noqa: E402
from train import predict_quantiles, apply_conformal  # noqa: E402
from config import TEST_YEARS, CALIB_YEARS  # noqa: E402

st.set_page_config(
    page_title="Fjord Catch Forecast",
    page_icon="🎣",
    layout="wide",
    menu_items={
        "About": "FCF v1.0 – Probabilistic weekly catch forecasts for Norwegian ports and species."
    },
)

DATA_PATH = Path("data/raw_catch_data.csv")
MODEL_PATH = Path("models/xgb_quantile_model.json")
FEATURE_COLS_PATH = Path("models/feature_columns.json")
METRICS_PATH = Path("models/metrics.json")
CONFORMAL_PATH = Path("models/conformal_Q.json")
IMPORTANCE_PATH = Path("models/feature_importance.csv")


@st.cache_data
def load_raw():
    return pd.read_csv(DATA_PATH, parse_dates=["date"])


@st.cache_data
def load_features(raw):
    feat = build_features(raw)
    return feat.dropna(subset=["lag_tons_4", "seasonal_avg_hist"]).reset_index(drop=True)


@st.cache_resource
def load_model():
    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))
    return booster


def forecast_next_weeks(raw: pd.DataFrame, booster: xgb.Booster, feature_cols: list[str],
                         port: str, species: str, n_weeks: int, conformal_by_species: dict) -> pd.DataFrame:
    """Recursively forecast n_weeks ahead for one port and species pair.

    Simplifying assumption, also shown in the UI: ocean features (SST,
    chlorophyll, and wave height), quota_pct_used, and storm are carried
    forward unchanged from the latest known week. This is a scenario, not a
    live weather forecast. Operational forecasting would require Copernicus
    "_anfc_" products; see the README.
    """
    subset = raw[(raw["port"] == port) & (raw["species"] == species)].sort_values(["year", "week"]).reset_index(drop=True)
    if subset.empty:
        return pd.DataFrame()

    conformal_Q = conformal_by_species.get(species, conformal_by_species["_global"])

    extended = subset.copy()
    rows_out = []

    for _ in range(n_weeks):
        last = extended.iloc[-1].copy()
        next_week = int(last["week"]) + 1
        next_year = int(last["year"])
        if next_week > 52:
            next_week = 1
            next_year += 1

        new_row = last.copy()
        new_row["year"] = next_year
        new_row["week"] = next_week
        new_row["date"] = pd.Timestamp.fromisocalendar(next_year, next_week, 1)
        # Copy the latest tons value temporarily so build_features can compute
        # lagged features; replace it with the predicted median below.
        extended = pd.concat([extended, pd.DataFrame([new_row])], ignore_index=True)

        feat_ext = build_features(extended)
        feat_row = feat_ext.iloc[[-1]].reindex(columns=feature_cols, fill_value=0)

        preds = predict_quantiles(booster, feat_row)
        median = float(preds[0.5][0])
        lower = float(preds[0.1][0]) - conformal_Q
        upper = float(preds[0.9][0]) + conformal_Q

        extended.loc[extended.index[-1], "tons"] = median
        rows_out.append({
            "year": next_year, "week": next_week,
            "label": f"{next_year}-W{next_week:02d}",
            "median": median, "lower": max(0.0, lower), "upper": max(0.0, upper),
        })

    return pd.DataFrame(rows_out)


def main():
    # Render the dashboard shell and the onboarding content.
    st.markdown(
        """
        <style>
        [data-testid="stMetric"] { background: #f5f9fc; border: 1px solid #dbe8f0;
                                    padding: .8rem; border-radius: 10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Fjord Catch Forecast")
    st.markdown("Probabilistic weekly landed-catch forecasts for Norwegian ports and species.")

    with st.expander("About this app", expanded=True):
        st.markdown(
            "Fjord Catch Forecast explores probabilistic weekly catch forecasts for Norwegian "
            "ports and species. It is intended for exploring historical forecasts and uncertainty, "
            "not for making operational decisions on its own. The data combines Fisheries Directorate "
            "landing records with Copernicus Marine features. On the chart, the blue line is the median "
            "forecast, the shaded band is the calibrated 80% interval, and gray dots are actual landings."
        )

    with st.expander("How to use this dashboard"):
        st.markdown(
            "- Use the sidebar to choose a port, species, test year, and visible week range.\n"
            "- Compare the median forecast with actual landings in the interactive chart.\n"
            "- Treat the shaded 80% interval as a plausible range, not a guarantee.\n"
            "- Read MAE, MdAPE, and coverage together; each describes a different aspect of quality.\n"
            "- Open the README and `reports/final_evaluation.md` for methodology and limitations."
        )

    if not (DATA_PATH.exists() and MODEL_PATH.exists()):
        st.error(
            "Data or model artifacts are missing. Run these commands in the terminal first:\n\n"
            "`python src/ingest_fiskeridir.py`\n\n"
            "`python src/ingest_copernicus.py`\n\n"
            "`python src/merge_sources.py`\n\n"
            "`python src/train.py`"
        )
        st.stop()

    raw = load_raw()
    feat = load_features(raw)
    booster = load_model()
    feature_cols = json.loads(FEATURE_COLS_PATH.read_text())
    metrics = json.loads(METRICS_PATH.read_text())
    conformal_by_species = json.loads(CONFORMAL_PATH.read_text())["conformal_by_species"]

    ports = sorted(raw["port"].unique())
    species_list = sorted(raw["species"].unique())
    test_years = sorted(TEST_YEARS)

    # Keep navigation controls together in the sidebar.
    with st.sidebar:
        st.header("Forecast controls")
        port = st.selectbox("Port", ports, index=ports.index("VÅGAN") if "VÅGAN" in ports else 0)
        species = st.selectbox("Species", species_list, index=species_list.index("torsk") if "torsk" in species_list else 0)
        selected_year = st.selectbox("Test year", ["All test years", *test_years])
        week_range = st.slider("Visible ISO weeks", 1, 52, (1, 52))
        show_residuals = st.checkbox("Show residuals", value=False)
        forecast_horizon = st.slider("Forecast horizon (weeks ahead)", 1, 3, 3)
        st.divider()
        st.caption("Filters affect the forecast chart. Saved model metrics remain global test-set metrics.")

    st.markdown("### Forecast")
    subset = feat[(feat["port_" + port] == True) & (feat["species_" + species] == True)].copy()
    subset = subset.sort_values(["year", "week"])
    plot_df = subset[subset["year"].isin(TEST_YEARS)].copy()
    if selected_year != "All test years":
        plot_df = plot_df[plot_df["year"] == selected_year]
    plot_df = plot_df[plot_df["week"].between(week_range[0], week_range[1])].copy()

    if plot_df.empty:
        st.warning("No test-period data is available for the selected filters.")
        return

    X_plot = plot_df[feature_cols]
    preds = predict_quantiles(booster, X_plot)
    pred_median = preds[0.5]
    pred_lower, pred_upper = apply_conformal(plot_df, preds, conformal_by_species)
    plot_df["residual"] = plot_df["tons"].to_numpy() - pred_median

    chart_year = selected_year if selected_year != "All test years" else f"{test_years[0]}-{test_years[-1]}"
    visible_week_ticks = [week for week in [1, 10, 20, 30, 40, 50] if week_range[0] <= week <= week_range[1]]

    forecast_fig = go.Figure()
    forecast_fig.add_trace(go.Scatter(
        x=plot_df["week"], y=pred_upper, mode="lines", line=dict(width=0),
        name="80% upper bound", showlegend=False,
    ))
    forecast_fig.add_trace(go.Scatter(
        x=plot_df["week"], y=pred_lower, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(87, 180, 210, 0.20)", name="Calibrated 80% interval",
    ))
    forecast_fig.add_trace(go.Scatter(
        x=plot_df["week"], y=pred_median, mode="lines", name="Median forecast",
        line=dict(color="#1565c0", width=2),
    ))
    forecast_fig.add_trace(go.Scatter(
        x=plot_df["week"], y=plot_df["tons"], mode="markers", name="Actual landings",
        marker=dict(color="#6b7280", size=7),
    ))
    forecast_fig.update_layout(
        title=f"Forecast vs actual landings – {port} / {species} / {chart_year}",
        xaxis=dict(
            title="Week (ISO)", tickmode="array", tickvals=visible_week_ticks,
            ticktext=[str(week) for week in visible_week_ticks], dtick=1,
        ),
        yaxis_title="Landed catch (tonnes)",
        hovermode="x unified", template="plotly_white", height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=80, b=20),
    )
    st.plotly_chart(forecast_fig, use_container_width=True)
    st.caption(
        f"Test years are held out from fitting. Calibration year {CALIB_YEARS[0]} was used only "
        "to calculate the CQR interval correction."
    )

    if show_residuals:
        residual_fig = go.Figure(go.Bar(
            x=plot_df["week"], y=plot_df["residual"], marker_color="#8ecae6",
            name="Residual (actual - forecast)",
        ))
        residual_fig.add_hline(y=0, line_color="#d9e2ec", line_width=1)
        residual_fig.update_layout(
            title="Forecast residuals", xaxis_title="Week (ISO)",
            xaxis=dict(tickmode="array", tickvals=visible_week_ticks, ticktext=[str(week) for week in visible_week_ticks]),
            yaxis_title="Residual (tonnes)", template="plotly_dark", height=360,
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(residual_fig, use_container_width=True)

    # Render the recursive forecast scenario without changing its calculations.
    st.markdown("### Forecast scenario")
    conformal_q = conformal_by_species.get(species, conformal_by_species["_global"])
    future = forecast_next_weeks(raw, booster, feature_cols, port, species, forecast_horizon, conformal_by_species)
    if not future.empty:
        cols = st.columns(len(future))
        for column, (_, row) in zip(cols, future.iterrows()):
            column.metric(
                row["label"], f"{row['median']:.1f} t",
                help=f"80% interval: {row['lower']:.1f} - {row['upper']:.1f} t",
            )
        st.caption(
            "Scenario assumption: ocean features, storm, and quota_pct_used are carried forward "
            "from the latest known week. This is not a live weather forecast."
        )

    st.divider()
    st.markdown("### Model performance")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("MAE", f"{metrics['mae_median_tons']:.2f} t", help="Mean absolute error of the median forecast.")
    k2.metric("MdAPE", f"{metrics.get('mdape_median_pct', float('nan')):.2f}%", help="Median absolute percentage error.")
    k3.metric("Calibrated 80% coverage", f"{metrics['coverage_80pct_interval_calibrated'] * 100:.1f}%", help="Share of actual values inside the calibrated interval.")
    k4.metric("Mean interval width", f"{metrics['mean_interval_width_calibrated_tons']:.1f} t", help="Average width of the calibrated interval.")

    st.caption("MAE measures absolute error, MdAPE measures typical relative error, and coverage measures interval reliability.")
    performance_rows = [
        ("MAE (median forecast)", "mae_median_tons", "tonnes"),
        ("MdAPE", "mdape_median_pct", "%"),
        ("Pinball loss q10", "pinball_loss_log_q10", "log scale"),
        ("Pinball loss q50", "pinball_loss_log_q50", "log scale"),
        ("Pinball loss q90", "pinball_loss_log_q90", "log scale"),
        ("Raw 80% coverage", "coverage_80pct_interval_raw", "%"),
        ("Calibrated 80% coverage", "coverage_80pct_interval_calibrated", "%"),
        ("Mean calibrated interval width", "mean_interval_width_calibrated_tons", "tonnes"),
    ]
    performance_data = []
    for label, key, unit in performance_rows:
        value = metrics.get(key)
        if value is None:
            continue
        display_value = value * 100 if "coverage" in key else value
        performance_data.append({"Metric": label, "Value": f"{display_value:.3f}", "Unit": unit})
    st.dataframe(pd.DataFrame(performance_data), hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("### Data & notes")
    note_col, importance_col = st.columns([1, 1])
    with note_col:
        st.markdown(
            f"**Selected series:** `{port}` / `{species}`  \n"
            f"**Visible observations:** {len(plot_df)}  \n"
            f"**CQR correction:** +/- {conformal_q:.2f} tonnes  \n"
            "Raw data and model artifacts are local files excluded from Git by default."
        )
        with st.expander("Model limitations"):
            st.markdown(
                "- Historical Copernicus `_my_` reanalysis is not an operational weather forecast.\n"
                "- `quota_pct_used` is a catch-based proxy, not official port-level quota data.\n"
                "- The future forecast scenario carries selected conditions forward unchanged.\n"
                "- The model cannot anticipate one-off events absent from historical data."
            )
    with importance_col:
        st.markdown("#### What drives the forecast")
        if IMPORTANCE_PATH.exists():
            importance = pd.read_csv(IMPORTANCE_PATH, index_col=0).iloc[:, 0].sort_values().tail(8)
            importance_fig = go.Figure(go.Bar(
                x=importance.values, y=importance.index, orientation="h",
                marker_color="#2e86ab", name="Gain importance",
            ))
            importance_fig.update_layout(
                xaxis_title="Gain importance", yaxis_title="Feature", template="plotly_white",
                height=360, margin=dict(l=20, r=20, t=20, b=20), showlegend=False,
            )
            st.plotly_chart(importance_fig, use_container_width=True)
        else:
            st.info("feature_importance.csv is missing. Run src/train.py first.")


if __name__ == "__main__":
    main()
