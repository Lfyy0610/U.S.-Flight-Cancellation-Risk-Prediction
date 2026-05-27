"""Part 4 monitoring and drift demo with Evidently.

This follows the same workflow as the Assignment 4 Evidently notebook:
build reference/current monitoring frames, evaluate the shifted scenario,
run Evidently DataDrift/DataSummary reports, and save local artifacts.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "flights_sample_3m.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model_pipeline.pkl"
CONFIG_PATH = ARTIFACT_DIR / "model_config.json"
REPORT_DIR = Path(__file__).resolve().parent

RAW_COLUMNS = [
    "FL_DATE",
    "AIRLINE_CODE",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "CANCELLED",
]

CATEGORICAL_COLUMNS = [
    "AIRLINE_CODE",
    "ORIGIN",
    "DEST",
    "ROUTE",
    "DEP_PERIOD",
]

NUMERICAL_COLUMNS = [
    "DEP_HOUR",
    "DEP_MIN",
    "IS_WEEKEND",
    "IS_HOLIDAY_SEASON",
    "IS_SUMMER_PEAK",
    "IS_COVID_YEAR",
    "cancellation_probability",
    "will_cancel",
]

TARGET = "CANCELLED"
PREDICTION = "will_cancel"


def hour_to_period(hour: int) -> str:
    if hour < 6:
        return "red_eye"
    if hour < 12:
        return "morning"
    if hour < 17:
        return "afternoon"
    if hour < 21:
        return "evening"
    return "night"


def add_deployment_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    features["FL_DATE"] = pd.to_datetime(features["FL_DATE"], errors="coerce")
    features = features.dropna(subset=["FL_DATE", "AIRLINE_CODE", "ORIGIN", "DEST", "CRS_DEP_TIME"])

    features["AIRLINE_CODE"] = features["AIRLINE_CODE"].astype(str).str.strip().str.upper()
    features["ORIGIN"] = features["ORIGIN"].astype(str).str.strip().str.upper()
    features["DEST"] = features["DEST"].astype(str).str.strip().str.upper()
    features["CANCELLED"] = pd.to_numeric(features["CANCELLED"], errors="coerce").fillna(0).astype(int)

    crs_dep_time = pd.to_numeric(features["CRS_DEP_TIME"], errors="coerce").fillna(0).astype(int)
    features["DEP_HOUR"] = (crs_dep_time // 100).clip(0, 23).astype(int)
    features["DEP_MIN"] = (crs_dep_time % 100).clip(0, 59).astype(int)
    features["DEP_PERIOD"] = features["DEP_HOUR"].apply(hour_to_period)
    features["IS_WEEKEND"] = features["FL_DATE"].dt.dayofweek.isin([5, 6]).astype(int)
    features["IS_HOLIDAY_SEASON"] = features["FL_DATE"].dt.month.isin([11, 12, 1]).astype(int)
    features["IS_SUMMER_PEAK"] = features["FL_DATE"].dt.month.isin([6, 7, 8]).astype(int)
    features["IS_COVID_YEAR"] = (features["FL_DATE"].dt.year == 2020).astype(int)
    features["ROUTE"] = features["ORIGIN"] + "_" + features["DEST"]

    return features


def load_2023_reference(data_path: Path, sample_size: int, random_state: int) -> pd.DataFrame:
    raw = pd.read_csv(data_path, usecols=RAW_COLUMNS, low_memory=False)
    raw["FL_DATE"] = pd.to_datetime(raw["FL_DATE"], errors="coerce")
    raw = raw[raw["FL_DATE"].dt.year == 2023].copy()
    if raw.empty:
        raise RuntimeError(f"No 2023 rows found in {data_path}")
    if len(raw) > sample_size:
        raw = raw.sample(n=sample_size, random_state=random_state)
    return add_deployment_features(raw)


def make_modified_current(reference: pd.DataFrame) -> pd.DataFrame:
    current = reference.copy().reset_index(drop=True)
    n_rows = len(current)

    current.loc[: int(n_rows * 0.40), "AIRLINE_CODE"] = "WN"
    current.loc[: int(n_rows * 0.30), "ORIGIN"] = "LAX"
    current.loc[: int(n_rows * 0.25), "DEP_HOUR"] = 23
    current.loc[: int(n_rows * 0.25), "DEP_MIN"] = 45
    current.loc[: int(n_rows * 0.25), "DEP_PERIOD"] = "red_eye"
    current.loc[: int(n_rows * 0.20), "IS_HOLIDAY_SEASON"] = 1
    current["ROUTE"] = current["ORIGIN"] + "_" + current["DEST"]

    return current


def load_model_and_config() -> tuple[object, dict[str, object]]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = json.load(f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = joblib.load(MODEL_PATH)
    return model, config


def add_predictions(df: pd.DataFrame, model: object, config: dict[str, object]) -> pd.DataFrame:
    monitoring_df = df.copy()
    feature_columns = config["feature_columns"]
    probabilities = model.predict_proba(monitoring_df[feature_columns])[:, 1]
    threshold = float(config["optimal_threshold"])
    monitoring_df["cancellation_probability"] = probabilities
    monitoring_df["will_cancel"] = (probabilities >= threshold).astype(int)
    return monitoring_df[
        CATEGORICAL_COLUMNS
        + NUMERICAL_COLUMNS
        + [TARGET]
    ]


def build_monitoring_frames(sample_size: int, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    model, config = load_model_and_config()
    reference_features = load_2023_reference(DATA_PATH, sample_size, random_state)
    current_features = make_modified_current(reference_features)
    reference = add_predictions(reference_features, model, config)
    current = add_predictions(current_features, model, config)
    return reference, current


def evaluate_scenario(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Scenario": "Original_2023_Test",
                "Rows": len(reference),
                "Cancellation Rate": reference[TARGET].mean(),
                "Mean Prediction Probability": reference["cancellation_probability"].mean(),
                "Predicted Cancellation Rate": reference["will_cancel"].mean(),
            },
            {
                "Scenario": "Modified_Current",
                "Rows": len(current),
                "Cancellation Rate": current[TARGET].mean(),
                "Mean Prediction Probability": current["cancellation_probability"].mean(),
                "Predicted Cancellation Rate": current["will_cancel"].mean(),
            },
        ]
    )


def run_evidently_report(reference: pd.DataFrame, current: pd.DataFrame, output_path: Path) -> None:
    data_definition = DataDefinition(
        categorical_columns=CATEGORICAL_COLUMNS,
        numerical_columns=NUMERICAL_COLUMNS + [TARGET],
    )
    reference_dataset = Dataset.from_pandas(reference, data_definition=data_definition)
    current_dataset = Dataset.from_pandas(current, data_definition=data_definition)

    report = Report([
        DataDriftPreset(),
        DataSummaryPreset(),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        snapshot = report.run(
            current_data=current_dataset,
            reference_data=reference_dataset,
            metadata={"scenario": "Modified_Current"},
            name="Flight Cancellation Drift Report",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Evidently monitoring report for Part 4.")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    REPORT_DIR.mkdir(exist_ok=True)
    report_path = REPORT_DIR / "drift_report.html"
    metrics_path = REPORT_DIR / "scenario_metrics.csv"

    reference, current = build_monitoring_frames(args.sample_size, args.random_state)
    metrics = evaluate_scenario(reference, current)
    metrics.to_csv(metrics_path, index=False)
    run_evidently_report(reference, current, report_path)

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved Evidently report: {report_path}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
