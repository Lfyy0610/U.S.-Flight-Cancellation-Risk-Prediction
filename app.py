import json
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model_pipeline.pkl"
CONFIG_PATH = ARTIFACT_DIR / "model_config.json"


class FlightFeatures(BaseModel):
    AIRLINE_CODE: str = Field(..., examples=["UA"])
    ORIGIN: str = Field(..., examples=["FLL"])
    DEST: str = Field(..., examples=["EWR"])
    ROUTE: Optional[str] = Field(None, examples=["FLL_EWR"])
    DEP_PERIOD: str = Field(..., examples=["morning"])
    DEP_HOUR: int = Field(..., ge=0, le=23, examples=[11])
    DEP_MIN: int = Field(..., ge=0, le=59, examples=[55])
    IS_WEEKEND: int = Field(..., ge=0, le=1, examples=[0])
    IS_HOLIDAY_SEASON: int = Field(..., ge=0, le=1, examples=[1])
    IS_SUMMER_PEAK: int = Field(..., ge=0, le=1, examples=[0])
    IS_COVID_YEAR: int = Field(..., ge=0, le=1, examples=[0])

    model_config = {
        "json_schema_extra": {
            "example": {
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
                "IS_COVID_YEAR": 0,
            }
        }
    }


class PredictionResponse(BaseModel):
    cancellation_probability: float
    threshold: float
    will_cancel: int
    risk_label: str


def load_artifacts():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model artifact not found: {MODEL_PATH}")
    if not CONFIG_PATH.exists():
        raise RuntimeError(f"Model config not found: {CONFIG_PATH}")

    model_pipeline = joblib.load(MODEL_PATH)
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        model_config = json.load(f)

    return model_pipeline, model_config


pipeline, config = load_artifacts()

app = FastAPI(
    title="Flight Cancellation Risk API",
    description="FastAPI service for predicting U.S. flight cancellation risk.",
    version="1.0.0",
)


def prepare_features(input_data: FlightFeatures) -> pd.DataFrame:
    row = input_data.model_dump()

    row["AIRLINE_CODE"] = row["AIRLINE_CODE"].strip().upper()
    row["ORIGIN"] = row["ORIGIN"].strip().upper()
    row["DEST"] = row["DEST"].strip().upper()
    row["DEP_PERIOD"] = row["DEP_PERIOD"].strip().lower()
    row["ROUTE"] = (row.get("ROUTE") or f"{row['ORIGIN']}_{row['DEST']}").strip().upper()

    try:
        return pd.DataFrame([row])[config["feature_columns"]]
    except KeyError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Input is missing required model feature: {exc}",
        ) from exc


def risk_label(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "high"
    if probability >= threshold * 0.5:
        return "medium"
    return "low"


@app.get("/")
def root():
    return {
        "service": "Flight Cancellation Risk API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
        "feature_count": len(config["feature_columns"]),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: FlightFeatures):
    features = prepare_features(input_data)

    try:
        probability = float(pipeline.predict_proba(features)[0, 1])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {exc}",
        ) from exc

    threshold = float(config["optimal_threshold"])
    will_cancel = int(probability >= threshold)

    return PredictionResponse(
        cancellation_probability=probability,
        threshold=threshold,
        will_cancel=will_cancel,
        risk_label=risk_label(probability, threshold),
    )
