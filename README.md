# U.S. Flight Cancellation Risk Prediction with MLOps

This project builds an MLOps pipeline for predicting U.S. domestic flight cancellation risk. The workflow uses historical flight records, creates pre-flight features, trains an imbalanced binary classification model, packages the preprocessing and model artifacts, and deploys the model through a FastAPI service with Docker support.

## Project Goal

Predict whether a scheduled U.S. domestic flight is likely to be cancelled using airline, airport, route, schedule, distance, and time-based information. The project is designed as a practical MLOps workflow: data preparation, EDA, model training, model packaging, API deployment, containerization, and monitoring/drift demonstration.

## Current Repository Structure

```text
.
├── Flight Cancellation Analysis.ipynb
├── data/
│   └── flights_sample_3m.csv
├── artifacts/
│   ├── model_config.json
│   └── model_pipeline.pkl
├── app.py
├── Dockerfile
├── requirements.txt
├── sample_request.json
├── README_step3_deployment.md
├── best_model.pkl
└── README.md
```

## End-to-End Architecture

```text
Raw flight dataset
        |
        v
Part 1: Data cleaning, EDA, and feature engineering
        |
        v
Temporal train/test split
        |
        v
Part 2: AutoML model training with preprocessing pipeline
        |
        v
artifacts/model_pipeline.pkl + artifacts/model_config.json
        |
        v
Part 3: FastAPI inference service
        |
        v
Dockerized API deployment
        |
        v
Part 4: Monitoring and drift demo
```

## Part 1: Data, EDA, and Feature Engineering

Main file:

- `Flight Cancellation Analysis.ipynb`

Dataset:

- `data/flights_sample_3m.csv`

What this part does:

- Loads the Kaggle flight delay and cancellation sample dataset.
- Parses `FL_DATE` into calendar features such as `YEAR`, `MONTH`, and `DAY_OF_WEEK`.
- Uses `CANCELLED` as the binary target label.
- Removes diverted flights.
- Drops or excludes post-flight leakage columns such as actual departure/arrival times, delays, taxi times, air time, and cancellation code.
- Standardizes key categorical fields such as `AIRLINE_CODE`, `ORIGIN`, and `DEST`.
- Creates deployment-friendly pre-flight features:
  - `DEP_HOUR`
  - `DEP_MIN`
  - `DEP_PERIOD`
  - `IS_WEEKEND`
  - `IS_HOLIDAY_SEASON`
  - `IS_SUMMER_PEAK`
  - `IS_COVID_YEAR`
  - `ROUTE`
- Splits the data temporally:
  - Training data: 2019-2022
  - Test data: 2023
- Produces EDA charts for class imbalance, monthly cancellation rate, airline cancellation rate, departure-hour patterns, numeric distributions, and heatmaps.

Important note:

- The notebook currently references `Data/flights_sample_3m.csv`, while the repository folder is `data/flights_sample_3m.csv`. If running locally on a case-sensitive path, update the notebook path to `data/flights_sample_3m.csv`.

## Part 2: Modeling, AutoML, and MLflow

Main file:

- `Flight Cancellation Analysis.ipynb`

Generated artifacts:

- `artifacts/model_pipeline.pkl`
- `artifacts/model_config.json`

What this part does:

- Prepares `X_train`, `X_test`, `y_train`, and `y_test` from the Part 1 temporal split.
- Keeps only deployment-safe input features:
  - Categorical: `AIRLINE_CODE`, `ORIGIN`, `DEST`, `ROUTE`, `DEP_PERIOD`
  - Numeric: `DEP_HOUR`, `DEP_MIN`, `IS_WEEKEND`, `IS_HOLIDAY_SEASON`, `IS_SUMMER_PEAK`, `IS_COVID_YEAR`
- Builds a scikit-learn preprocessing pipeline with `OrdinalEncoder`.
- Handles unknown categories during inference with:

```python
OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
```

- Uses FLAML AutoML for model selection.
- Optimizes for Average Precision / PR-AUC because flight cancellations are highly imbalanced.
- Tunes the prediction threshold using the precision-recall curve instead of relying on the default `0.5` threshold.
- Logs metrics and artifacts with MLflow.
- Saves a complete deployment pipeline that includes both preprocessing and the trained classifier.

Current model summary from the notebook:

- Dataset cancellation rate: about 2.64%.
- Default `0.5` threshold produced F1 = 0 for the cancelled class.
- Tuned threshold: about `0.0523`.
- Tuned F1 for the cancelled class: about `0.0847`.
- The model identifies some cancellation risk, but the cancelled class remains difficult because the dataset is extremely imbalanced.

Artifact responsibilities:

- `artifacts/model_pipeline.pkl`: complete preprocessing plus model pipeline used by the API.
- `artifacts/model_config.json`: feature order, categorical columns, numeric columns, and tuned classification threshold.
- `best_model.pkl`: older standalone model artifact. The FastAPI deployment uses `artifacts/model_pipeline.pkl`, not this file.

## Part 3: Deployment, API, and Docker

Main files:

- `app.py`
- `requirements.txt`
- `Dockerfile`
- `sample_request.json`
- `README_step3_deployment.md`

What this part does:

- Loads `artifacts/model_pipeline.pkl` and `artifacts/model_config.json` at API startup.
- Defines a FastAPI application for inference.
- Provides health and prediction endpoints:
  - `GET /`
  - `GET /health`
  - `POST /predict`
- Validates input fields with Pydantic.
- Normalizes airline and airport codes before prediction.
- Automatically fills `ROUTE` as `ORIGIN_DEST` if it is not supplied.
- Returns:
  - cancellation probability
  - tuned decision threshold
  - binary prediction
  - readable risk label
- Packages the service into a Docker container.

Run locally:

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

Run with Docker:

```bash
docker build -t flight-cancel-api .
docker run --rm -p 8000:8000 flight-cancel-api
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## Part 4: Monitoring and Drift Demo

This part demonstrates data drift monitoring for the deployed cancellation-risk model. It compares the original 2023 test data against a modified current dataset that simulates distribution shifts in carrier, origin airport, route, departure period, departure hour, and holiday-season traffic.

Main files:

```text
monitoring/
├── drift_demo.py
├── drift_report.html
├── scenario_metrics.csv
├── requirements_monitoring.txt
├── test_drift_demo.py
└── README_step4_monitoring.md
```

What this part does:

- Uses the 2023 test slice as reference data.
- Creates a deterministic modified current dataset by changing selected feature distributions.
- Recomputes deployment features such as `DEP_HOUR`, `DEP_PERIOD`, `IS_HOLIDAY_SEASON`, and `ROUTE`.
- Loads the trained `artifacts/model_pipeline.pkl` and scores both reference and current data.
- Saves scenario-level metrics in `monitoring/scenario_metrics.csv`.
- Uses Evidently `DataDriftPreset` and `DataSummaryPreset`.
- Exports a browser-readable Evidently HTML report.
- Includes lightweight unit tests for feature engineering and scenario construction.

Install monitoring dependencies:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r monitoring/requirements_monitoring.txt
```

Run the monitoring demo:

```bash
.venv/bin/python monitoring/drift_demo.py --sample-size 5000
```

Run Part 4 tests:

```bash
.venv/bin/python -m unittest monitoring/test_drift_demo.py
```

Generated report:

```text
monitoring/drift_report.html
```

## API Input Schema

The deployed model expects the following fields:

```json
{
  "AIRLINE_CODE": "UA",
  "ORIGIN": "FLL",
  "DEST": "EWR",
  "ROUTE": "FLL_EWR",
  "DEP_PERIOD": "morning",
  "DEP_HOUR": 11,
  "DEP_MIN": 55,
  "IS_WEEKEND": 0,
  "IS_HOLIDAY_SEASON": 1,
  "IS_SUMMER_PEAK": 0,
  "IS_COVID_YEAR": 0
}
```

Example output:

```json
{
  "cancellation_probability": 0.0237,
  "threshold": 0.0523,
  "will_cancel": 0,
  "risk_label": "low"
}
```

## Team Workflow

Recommended Git workflow for team members:

```bash
git clone https://github.com/Lfyy0610/U.S.-Flight-Cancellation-Risk-Prediction.git
cd U.S.-Flight-Cancellation-Risk-Prediction
git checkout -b your-name-your-part
```

After finishing changes:

```bash
git status
git add .
git commit -m "Add your part description"
git push origin your-name-your-part
```

Then open a pull request on GitHub and merge into `main` after review.

## Presentation Talking Points

- This is a real business problem because airlines, airports, and passengers benefit from early cancellation-risk signals.
- The task is binary classification with strong class imbalance.
- Accuracy alone is misleading because most flights are not cancelled.
- PR-AUC, recall, F1, and threshold tuning are more meaningful than accuracy for the cancelled class.
- The deployment artifact is a full preprocessing-plus-model pipeline, which reduces training-serving skew.
- FastAPI provides a clean inference interface, and Docker makes the deployment reproducible.
- The monitoring section should show how data drift can affect model reliability after deployment.
