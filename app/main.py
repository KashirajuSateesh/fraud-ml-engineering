from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import json
from datetime import datetime, timezone

app = FastAPI()
model = None
threshold = None

MODEL_PATH = "artifacts/xgb_model.joblib"
THRESHOLD_PATH = "artifacts/threshold.joblib"
LOG_PATH = "logs/predictions.jsonl"

@app.on_event("startup")
def load_artifacts():
    global model, threshold
    model = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)


@app.get("/")
def root():
    return {"message" : "Fraud Detection API is Running"}

@app.get("/health")
def health():
    return {"status" : "ok", "model_loaded": model is not None}


class PredictRequest(BaseModel):
    features: dict[str, float]

@app.post("/predict")
def predict(req: PredictRequest):
    # Build a 1 row data frame from incoming features
    X = pd.DataFrame([req.features])

    # Get fraud probability (class 1)
    proba = float(model.predict_proba(X)[0][1])
    label = int(proba >= float(threshold))

    log_record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "fraud_probability": proba,
        "threshold": float(threshold),
        "label": label
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(log_record)+"\n")
    print("PREDICTION_LOG:", json.dumps(log_record))

    return {
        "fraud_probability": proba,
        "threshold": float(threshold),
        "label": label

    }