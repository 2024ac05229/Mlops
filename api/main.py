import os
import time
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from api.schemas import PredictRequest, PredictResponse

# Paths
API_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(API_DIR)
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "best_model.joblib")

# Global variables to hold the model
model = None
model_name = "Unknown"

# Define Prometheus metrics
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["endpoint", "method", "http_status"]
)
REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "API request latency in seconds",
    ["endpoint"]
)
PREDICTIONS_COUNT = Counter(
    "api_predictions_total",
    "Total predictions made",
    ["prediction"]
)
CONFIDENCE_DIST = Histogram(
    "api_prediction_confidence",
    "Distribution of prediction confidence scores",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
)

def load_model():
    global model, model_name
    print(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"WARNING: Model file not found at {MODEL_PATH}. Prediction endpoint will fail until a model is trained.")
        return
    try:
        model = joblib.load(MODEL_PATH)
        # Determine model name from classifier step
        clf_step = model.named_steps.get("classifier")
        model_name = type(clf_step).__name__ if clf_step else "Scikit-learn Pipeline"
        print(f"Successfully loaded {model_name}")
    except Exception as e:
        print(f"Error loading model: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    load_model()
    yield

# Initialize FastAPI with lifespan
app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="FastAPI serving a machine learning model to predict heart disease risk.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health():
    global model
    status = "healthy"
    model_loaded = model is not None
    if not model_loaded:
        status = "degraded"
    
    return {
        "status": status,
        "model_loaded": model_loaded,
        "model_name": model_name
    }

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    global model
    if model is None:
        REQUEST_COUNT.labels(endpoint="/predict", method="POST", http_status="503").inc()
        raise HTTPException(status_code=503, detail="Model is not loaded on server.")

    start_time = time.time()
    try:
        # Convert request to pandas DataFrame with single row (Pydantic v2 style)
        input_data = payload.model_dump()
        df = pd.DataFrame([input_data])

        # Generate prediction and probabilities
        pred = int(model.predict(df)[0])
        probabilities = model.predict_proba(df)[0]
        confidence = float(probabilities[pred])

        # Record metrics
        PREDICTIONS_COUNT.labels(prediction=str(pred)).inc()
        CONFIDENCE_DIST.observe(confidence)
        REQUEST_COUNT.labels(endpoint="/predict", method="POST", http_status="200").inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

        return PredictResponse(
            prediction=pred,
            confidence=confidence,
            model_name=model_name
        )

    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict", method="POST", http_status="500").inc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
