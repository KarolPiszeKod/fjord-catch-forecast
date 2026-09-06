# Final Evaluation

## Evaluation Goal

The evaluation compares the full-data production baseline with a diagnostic experiment restricted to the post-outcome `normal` segment. The goal is to assess predictive accuracy and calibrated interval coverage under the project's temporal validation design.

## Results

| Variant | MAE | MdAPE | Calibrated 80% coverage |
|---|---:|---:|---:|
| Full-data baseline | 249.57 t | 50.66% | 80.55% |
| Normal-segment oracle experiment | 193.72 t | 34.83% | 77.05% |

## Interpretation

The full-data baseline is the candidate v1 production model. It uses all available records and includes `n_vessels_lag_1` as a lagged fishing-effort proxy.

The normal-segment result is an oracle or diagnostic experiment. The `normal`, `weak`, and `high` catch segments were assigned after observing actual catch volume. Therefore, the normal-segment model cannot be deployed alone: its segment label is not known before the forecast.

## Limitations

- The future segment label is unavailable at forecast time. A deployable regime-specific system would require a pre-forecast gating classifier or another model that predicts the regime without using the outcome.
- MAPE is unstable for observations close to zero. MAE and MdAPE should be considered alongside MAPE.
- Historical Copernicus reanalysis is not the same as an operational future forecast source.
- The dashboard's multi-week scenario carries forward selected recent conditions and should not be interpreted as a guaranteed weather forecast.

## What the Model Is Useful For

- Probabilistic weekly catch-volume forecasts by port and species.
- Comparing expected catch volume with calibrated uncertainty intervals.
- Supporting planning and prioritization when uncertainty is considered explicitly.

## What It Should Not Be Used For

- A guarantee of exact future landings.
- A forecast that assumes a post-outcome catch segment is known in advance.
- Operational weather forecasting or causal conclusions about fishing behavior.
