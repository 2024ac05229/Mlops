import os
import pytest
from fastapi.testclient import TestClient
from api.main import app

# Initialize test client
client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data

def test_predict_endpoint_validation():
    # Sending incomplete request body (missing fields)
    incomplete_payload = {
        "age": 52.0,
        "sex": 1.0
    }
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422 # Unprocessable Entity

def test_predict_endpoint_success():
    # If the model is not loaded, we expect a 503 error, which is a handled case.
    # If the model is loaded, we expect a 200 success response.
    # This allows tests to run in different states.
    payload = {
        "age": 52.0,
        "sex": 1.0,
        "cp": 3.0,
        "trestbps": 125.0,
        "chol": 212.0,
        "fbs": 0.0,
        "restecg": 1.0,
        "thalach": 168.0,
        "exang": 0.0,
        "oldpeak": 1.0,
        "slope": 1.0,
        "ca": 2.0,
        "thal": 3.0
    }
    
    response = client.post("/predict", json=payload)
    
    health_response = client.get("/health")
    model_loaded = health_response.json().get("model_loaded", False)
    
    if model_loaded:
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "model_name" in data
        assert data["prediction"] in [0, 1]
        assert 0.0 <= data["confidence"] <= 1.0
    else:
        assert response.status_code == 503
        assert "Model is not loaded" in response.json()["detail"]

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "api_requests_total" in response.text
