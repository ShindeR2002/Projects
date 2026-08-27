from collections import defaultdict, deque
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
    description="Serves LSTM-based critical-risk predictions from aircraft engine telemetry.",
)

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# In-memory rolling windows for the local single-process service.
engine_windows = defaultdict(lambda: deque(maxlen=SEQUENCE_LENGTH))


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


class StreamPredictionRequest(BaseModel):
    engine_id: int = Field(..., description="Aircraft engine identifier")
    telemetry: List[float] = Field(
        ...,
        description="One chronological telemetry row containing the 18 model features.",
    )

    @field_validator("telemetry")
    @classmethod
    def validate_telemetry(cls, value: List[float]) -> List[float]:
        if len(value) != N_FEATURES:
            raise ValueError(f"telemetry must contain exactly {N_FEATURES} features")
        if not np.isfinite(np.asarray(value, dtype=np.float32)).all():
            raise ValueError("telemetry must contain only finite numeric values")
        return value


class PredictionResponse(BaseModel):
    risk_probability: float
    risk_label: int
    risk_status: str
    threshold: float


class StreamPredictionResponse(PredictionResponse):
    engine_id: int
    window_size: int
    ready_for_prediction: bool


def predict_sequence(sequence: np.ndarray) -> float:
    scaled = scaler.transform(sequence)
    model_input = scaled.reshape(1, SEQUENCE_LENGTH, N_FEATURES)
    return float(model.predict(model_input, verbose=0)[0][0])


def build_response(probability: float) -> PredictionResponse:
    risk_label = int(probability >= RISK_THRESHOLD)
    return PredictionResponse(
        risk_probability=round(probability, 6),
        risk_label=risk_label,
        risk_status="CRITICAL_RISK" if risk_label else "NORMAL",
        threshold=RISK_THRESHOLD,
    )


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
        probability = predict_sequence(sequence)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return build_response(probability)


@app.post("/predict/stream", response_model=StreamPredictionResponse)
def predict_stream(request: StreamPredictionRequest):
    window = engine_windows[request.engine_id]
    window.append(request.telemetry)

    if len(window) < SEQUENCE_LENGTH:
        return StreamPredictionResponse(
            risk_probability=0.0,
            risk_label=0,
            risk_status="INSUFFICIENT_HISTORY",
            threshold=RISK_THRESHOLD,
            engine_id=request.engine_id,
            window_size=len(window),
            ready_for_prediction=False,
        )

    try:
        sequence = np.asarray(window, dtype=np.float32)
        probability = predict_sequence(sequence)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    response = build_response(probability)
    return StreamPredictionResponse(
        **response.model_dump(),
        engine_id=request.engine_id,
        window_size=len(window),
        ready_for_prediction=True,
    )
