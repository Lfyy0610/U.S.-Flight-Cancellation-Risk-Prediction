# Part 3 - FastAPI Deployment and Docker

This project deploys the trained flight cancellation model as a FastAPI inference service.

## Files

- `app.py`: FastAPI application with `/predict` endpoint.
- `artifacts/model_pipeline.pkl`: trained preprocessing and model pipeline from Part 2.
- `artifacts/model_config.json`: model feature order and decision threshold.
- `requirements.txt`: Python dependencies for local and Docker execution.
- `Dockerfile`: container image definition.
- `sample_request.json`: example prediction payload.

## Run Locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Example API request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

## Run With Docker

Build the Docker image:

```bash
docker build -t flight-cancel-api .
```

Run the container:

```bash
docker run --rm -p 8000:8000 flight-cancel-api
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## Endpoint

`POST /predict`

Input:

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

Output:

```json
{
  "cancellation_probability": 0.0237,
  "threshold": 0.0523,
  "will_cancel": 0,
  "risk_label": "low"
}
```

On macOS, XGBoost may require OpenMP before local execution:

```bash
brew install libomp
```

## Presentation Notes

FastAPI exposes the trained model through HTTP endpoints. Docker packages the API code, model artifacts, and dependencies into a reproducible container, so the same service can run on different machines.
