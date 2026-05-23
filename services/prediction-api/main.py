"""
MLOps Pulse - Prediction API
Trains model on startup for demo purposes.
In production, load from S3-backed MLflow.
"""
import os
import time
import logging
import numpy as np
import pandas as pd
import joblib
from contextlib import asynccontextmanager
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PREDICTION_COUNTER = Counter("prediction_requests_total", "Total prediction requests", ["status"])
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency", buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0])
PREDICTION_CONFIDENCE = Histogram("prediction_confidence_score", "Model confidence", buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
HIGH_RISK_COUNTER = Counter("high_risk_predictions_total", "High risk predictions")
MODEL_LOAD_STATUS = Gauge("model_loaded", "Model loaded status")

model = None
model_version = "v1.0-embedded"

def train_model():
    np.random.seed(42)
    n = 1000
    data = pd.DataFrame({
        "age": np.random.randint(18, 75, n),
        "credit_amount": np.random.randint(250, 18000, n),
        "duration_months": np.random.randint(4, 72, n),
        "employment_years": np.random.randint(0, 10, n),
        "installment_rate": np.random.randint(1, 5, n),
        "residence_years": np.random.randint(1, 5, n),
        "existing_credits": np.random.randint(1, 4, n),
        "num_dependents": np.random.randint(1, 3, n),
        "has_telephone": np.random.randint(0, 2, n),
        "is_foreign_worker": np.random.randint(0, 2, n),
    })
    risk_score = (
        -0.3 * (data["credit_amount"] / data["credit_amount"].max())
        + 0.2 * (data["duration_months"] / data["duration_months"].max())
        + 0.2 * (data["employment_years"] / data["employment_years"].max())
        + 0.1 * (data["age"] / data["age"].max())
        + np.random.normal(0, 0.1, n)
    )
    data["target"] = (risk_score > risk_score.median()).astype(int)
    X = data.drop("target", axis=1)
    y = data["target"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def load_model():
    global model
    try:
        logger.info("Training model on startup...")
        model = train_model()
        MODEL_LOAD_STATUS.set(1)
        logger.info("✅ Model trained and ready")
    except Exception as e:
        MODEL_LOAD_STATUS.set(0)
        logger.error(f"❌ Failed to train model: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(title="MLOps Pulse - Prediction API", version="1.0.0", lifespan=lifespan)

class PredictionRequest(BaseModel):
    age: int = Field(..., ge=18, le=100, example=35)
    credit_amount: float = Field(..., gt=0, example=5000)
    duration_months: int = Field(..., gt=0, example=24)
    employment_years: int = Field(..., ge=0, example=3)
    installment_rate: int = Field(..., ge=1, le=4, example=2)
    residence_years: int = Field(..., ge=1, le=4, example=3)
    existing_credits: int = Field(..., ge=1, example=1)
    num_dependents: int = Field(..., ge=1, le=2, example=1)
    has_telephone: int = Field(..., ge=0, le=1, example=1)
    is_foreign_worker: int = Field(..., ge=0, le=1, example=0)

class PredictionResponse(BaseModel):
    prediction: int
    label: str
    confidence: float
    model_version: str
    latency_ms: float

@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": "credit-risk-classifier", "version": model_version}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        PREDICTION_COUNTER.labels(status="error").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    start_time = time.time()
    try:
        input_df = pd.DataFrame([request.model_dump()])
        prediction = int(model.predict(input_df)[0])
        confidence = float(model.predict_proba(input_df)[0][prediction])
        latency = time.time() - start_time
        PREDICTION_COUNTER.labels(status="success").inc()
        PREDICTION_LATENCY.observe(latency)
        PREDICTION_CONFIDENCE.observe(confidence)
        if prediction == 0:
            HIGH_RISK_COUNTER.inc()
        label = "good_credit" if prediction == 1 else "high_risk"
        return PredictionResponse(prediction=prediction, label=label, confidence=round(confidence, 4), model_version=model_version, latency_ms=round(latency * 1000, 2))
    except Exception as e:
        PREDICTION_COUNTER.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/")
def root():
    return {"service": "MLOps Pulse", "health": "/health", "metrics": "/metrics", "docs": "/docs"}
