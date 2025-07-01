import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.pydantic_models import PredictionRequest

# Test client
client = TestClient(app)

# Sample valid request
SAMPLE_REQUEST = {
    "CustomerId": "CUST12345",
    "Total_Amount": 1500.0,
    "Avg_Amount": 500.0,
    "Transaction_Count": 3,
    "Recency": 15,
    "Frequency": 5,
    "Monetary": 2500.0,
    "ProductCategory": "Electronics",
    "ChannelId": "Web",
    "PricingStrategy": "Standard",
    "TransactionStartTime": "2023-06-15T14:30:00Z"
}

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_valid_prediction():
    response = client.post("/predict", json=SAMPLE_REQUEST)
    assert response.status_code == 200
    data = response.json()
    assert "risk_probability" in data
    assert "credit_score" in data
    assert 300 <= data["credit_score"] <= 850

def test_invalid_input():
    invalid_request = SAMPLE_REQUEST.copy()
    invalid_request["Total_Amount"] = "invalid"  # Wrong type
    
    response = client.post("/predict", json=invalid_request)
    assert response.status_code == 422
    assert "detail" in response.json()

def test_missing_field():
    incomplete_request = SAMPLE_REQUEST.copy()
    del incomplete_request["CustomerId"]
    
    response = client.post("/predict", json=incomplete_request)
    assert response.status_code == 422
    assert "CustomerId" in response.json()["detail"][0]["loc"]

def test_model_version():
    response = client.post("/predict", json=SAMPLE_REQUEST)
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert data["model_version"] in ["local", "production", "staging"]