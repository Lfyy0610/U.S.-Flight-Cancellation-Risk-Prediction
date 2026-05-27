# Part 4 - Evidently Monitoring and Drift Demo

This section follows the same method as the Assignment 4 Evidently workflow: build reference and current monitoring datasets, apply a controlled data shift, score both datasets with the deployed model, export scenario metrics, and generate a local Evidently HTML report.

## Files

- `drift_demo.py`: builds monitoring frames, creates the modified current scenario, runs model predictions, and generates the Evidently report.
- `drift_report.html`: generated Evidently `DataDriftPreset` and `DataSummaryPreset` report.
- `scenario_metrics.csv`: summary metrics for the original 2023 test sample and the modified current sample.
- `test_drift_demo.py`: unit tests for feature engineering, scenario construction, and metrics generation.
- `requirements_monitoring.txt`: dependencies for running the monitoring workflow.

## Monitoring Design

Reference data:

- A sample from the original 2023 test rows in `data/flights_sample_3m.csv`.

Current data:

- A modified copy of the reference data that simulates production drift.
- The shift changes airline carrier mix, origin airport mix, route distribution, departure period, departure hour, and holiday-season traffic.

Model scoring:

- The script loads `artifacts/model_pipeline.pkl`.
- It uses `artifacts/model_config.json` for feature order and the tuned classification threshold.
- It adds `cancellation_probability` and `will_cancel` to both reference and current monitoring frames.

Evidently report:

- Uses `DataDriftPreset()`.
- Uses `DataSummaryPreset()`.
- Saves the report locally as `monitoring/drift_report.html`.

## Setup

From the project root:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r monitoring/requirements_monitoring.txt
```

If you already installed the project dependencies, make sure `evidently` is installed in the same environment.

## Run the Demo

```bash
.venv/bin/python monitoring/drift_demo.py --sample-size 5000
```

Generated outputs:

```text
monitoring/drift_report.html
monitoring/scenario_metrics.csv
```

## Run Tests

```bash
.venv/bin/python -m unittest monitoring/test_drift_demo.py
```

## Current Metrics

The generated `scenario_metrics.csv` shows:

```text
Original_2023_Test: mean cancellation probability around 0.0206
Modified_Current: mean cancellation probability around 0.0246
```

The modified current data has a higher average predicted cancellation probability, which makes sense because the simulated current traffic has shifted toward different carrier, airport, route, and departure-time patterns.

## Presentation Notes

This completes the monitoring part of the MLOps pipeline:

1. Part 2 trains and packages the model.
2. Part 3 serves predictions through FastAPI and Docker.
3. Part 4 uses Evidently to compare reference data against current production-like data.
4. The report shows whether input features have drifted.
5. Drift results can trigger investigation, retraining, or feature/data pipeline checks.

The key point is that deployment is not the final step. After the API is live, the team still needs monitoring to detect when production data no longer looks like the data used to validate the model.
