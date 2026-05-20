"""
MLOps Pulse - Prediction API
Serves credit risk predictions from a registered MLflow model.
Exposes Prometheus metrics for observability.
"""

import os
import time
import logging
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5002")
MODEL_NAME = os.getenv("MODEL_NAME", "credit-risk-classifier")
MODEL_STAGE = os.getenv("MODEL_STAGE", "latest")

# ── Prometheus Metrics ────────────────────────────────────────────────────────
# These are the metrics Prometheus will scrape from /metrics
# Each one tells a different story about the health of the service

PREDICTION_COUNTER = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
    ["status", "model_version"],  # labels let you filter in Grafana
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time spent processing a prediction request",
    ["model_version"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence_score",
    "Distribution of model confidence scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

HIGH_RISK_COUNTER = Counter(
    "high_risk_predictions_total",
    "Total predictions classified as high credit risk",
)

MODEL_LOAD_STATUS = Gauge(
    "model_loaded",
    "Whether the ML model is currently loaded (1=yes, 0=no)",
)

# ── Model State ───────────────────────────────────────────────────────────────
model = None
model_version = "unknown"


def load_model():
    """Load the registered model from MLflow Model Registry."""
    global model, model_version
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        logger.info(f"Loading model from: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        model_version = MODEL_STAGE
        MODEL_LOAD_STATUS.set(1)
        logger.info(f"✅ Model loaded successfully: {MODEL_NAME}/{MODEL_STAGE}")
    except Exception as e:
        MODEL_LOAD_STATUS.set(0)
        logger.error(f"❌ Failed to load model: {e}")
        # Don't crash on startup — health check will surface this


# ── App Lifecycle ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    logger.info("Starting MLOps Pulse Prediction API...")
    load_model()
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="MLOps Pulse - Prediction API",
    description="Credit risk prediction service with full observability",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────
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


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """
    Kubernetes liveness + readiness probe endpoint.
    Returns 503 if model isn't loaded so K8s won't route traffic to a broken pod.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": MODEL_NAME, "version": model_version}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Run a credit risk prediction.
    Returns prediction (0=high risk, 1=good credit), confidence score, and latency.
    """
    if model is None:
        PREDICTION_COUNTER.labels(status="error", model_version=model_version).inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        # Build input dataframe — column order must match training
        input_df = pd.DataFrame([request.model_dump()])

        # Run inference
        prediction = int(model.predict(input_df)[0])
        confidence = float(model.predict_proba(input_df)[0][prediction])
        latency = time.time() - start_time

        # ── Record metrics
        PREDICTION_COUNTER.labels(status="success", model_version=model_version).inc()
        PREDICTION_LATENCY.labels(model_version=model_version).observe(latency)
        PREDICTION_CONFIDENCE.observe(confidence)

        if prediction == 0:
            HIGH_RISK_COUNTER.inc()

        label = "good_credit" if prediction == 1 else "high_risk"
        logger.info(f"Prediction: {label} | Confidence: {confidence:.3f} | Latency: {latency*1000:.1f}ms")

        return PredictionResponse(
            prediction=prediction,
            label=label,
            confidence=round(confidence, 4),
            model_version=model_version,
            latency_ms=round(latency * 1000, 2),
        )

    except Exception as e:
        PREDICTION_COUNTER.labels(status="error", model_version=model_version).inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload-model")
def reload_model():
    """Hot-reload the model from MLflow without restarting the pod."""
    load_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Model reload failed")
    return {"status": "reloaded", "model": MODEL_NAME, "version": model_version}


@app.get("/metrics")
def metrics():
    """
    Prometheus scrape endpoint.
    Prometheus hits this every 15s and stores the time-series data.
    Grafana then queries Prometheus to build dashboards.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
def root():
    return {
        "service": "MLOps Pulse - Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "metrics": "/metrics",
        "health": "/health",
    }