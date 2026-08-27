from pathlib import Path
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.h5"
SCALER_PATH = BASE_DIR / "scaler.pkl"
SEQUENCE_LENGTH = 50
N_FEATURES = 18
RISK_THRESHOLD = 0.5

app = FastAPI(
    title="Aircraft Engine Predictive Maintenance API",
    version="1.0.0",
    description="Serves LSTM-based critical-risk predictions from the latest 50 telemetry cycles.",
)

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


class PredictionRequest(BaseModel):
    sequence: List[List[float]] = Field(
        ...,
        description="Exactly 50 chronological telemetry rows, each containing 18 model features.",
    )

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, value: List[List[float]]) -> List[List[float]]:
        if len(value) != SEQUENCE_LENGTH:
            raise ValueError(f"sequence must contain exactly {SEQUENCE_LENGTH} rows")
        if any(len(row) != N_FEATURES for row in value):
            raise ValueError(f"each row must contain exactly {N_FEATURES} features")
        if not np.isfinite(np.asarray(value, dtype=np.float32)).all():
            raise ValueError("sequence must contain only finite numeric values")
        return value


class PredictionResponse(BaseModel):
    risk_probability: float
    risk_label: int
    risk_status: str
    threshold: float


@app.get("/")
def root():
    return {
        "service": "aircraft-engine-predictive-maintenance",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        sequence = np.asarray(request.sequence, dtype=np.float32)
        scaled = scaler.transform(sequence)
        model_input = scaled.reshape(1, SEQUENCE_LENGTH, N_FEATURES)
        probability = float(model.predict(model_input, verbose=0)[0][0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    risk_label = int(probability >= RISK_THRESHOLD)
    risk_status = "CRITICAL_RISK" if risk_label else "NORMAL"

    return PredictionResponse(
        risk_probability=round(probability, 6),
        risk_label=risk_label,
        risk_status=risk_status,
        threshold=RISK_THRESHOLD,
    )
