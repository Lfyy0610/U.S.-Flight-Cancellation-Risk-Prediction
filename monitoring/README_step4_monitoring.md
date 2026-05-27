# Part 4 - Monitoring and Drift Demo

This section demonstrates input data drift for the deployed flight cancellation model. It compares the original 2023 test distribution against a modified test distribution that simulates production changes in airline carrier mix, origin airport mix, route mix, departure time, and holiday-season traffic.

## Files

- `drift_demo.py`: monitoring script that builds reference/current datasets, computes drift scores, and generates an HTML report.
- `drift_report.html`: generated drift report for presentation.
- `test_drift_demo.py`: lightweight unit tests for the monitoring feature engineering and PSI drift scoring logic.

## Why Drift Monitoring Matters

The deployed model uses airline, airport, route, departure-time, and calendar features. If production traffic starts looking different from the training/test data, model predictions can become less reliable even if the API is still running correctly.

This demo intentionally modifies the current dataset to show how a monitoring report can detect those changes.

## Drift Scenario

Reference data:

- Original 2023 test rows from `data/flights_sample_3m.csv`.

Modified current data:

- Shifts part of the airline distribution toward `WN`.
- Shifts part of the origin airport distribution toward `LAX`.
- Shifts part of the departure-time distribution toward red-eye flights.
- Increases holiday-season examples.
- Recomputes route after changing origin values.

## Run the Demo

From the project root:

```bash
python3 monitoring/drift_demo.py --sample-size 5000
```

The script writes:

```text
monitoring/drift_report.html
```

Open the report in a browser to view the feature-level drift summary.

## Run Tests

```bash
python3 -m unittest monitoring/test_drift_demo.py
```

## Report Interpretation

The script uses Population Stability Index (PSI) for both categorical and numeric features.

```text
PSI < 0.10      Low drift
0.10 - 0.25     Moderate drift
PSI >= 0.25     High drift
```

In the generated report, the strongest drift appears in:

- `ROUTE`
- `ORIGIN`
- `DEP_PERIOD`
- `AIRLINE_CODE`
- `DEP_HOUR`

These are expected because the modified current dataset intentionally changes carrier, airport, route, and scheduled departure-time distributions.

## Presentation Notes

This monitoring demo completes the MLOps loop after deployment:

1. Train and package the model.
2. Serve predictions through FastAPI and Docker.
3. Compare incoming production-like data with reference test data.
4. Detect drift in model input features.
5. Use drift results as a signal to investigate model performance and decide whether retraining is needed.

The key message is that a deployed model is not finished after the API works. The input distribution must be monitored because real airline traffic patterns can change over time.
