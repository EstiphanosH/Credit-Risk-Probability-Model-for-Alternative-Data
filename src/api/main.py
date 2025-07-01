import os
import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic_models import PredictionRequest, PredictionResponse
from datetime import datetime

# Initialize FastAPI
app = FastAPI(
    title="Bati Bank Credit Risk API",
    description="API for credit risk predictions",
    version="1.0.0"
)

# Load model from MLflow
def load_model():
    model_uri = f"models:/CreditRiskModel/Production"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        app.state.model = model
        app.state.model_version = model._model_meta.run_id
        print(f"Loaded model version: {app.state.model_version}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        # Fallback to local model
        try:
            model = mlflow.pyfunc.load_model("models/credit_risk_model")
            app.state.model = model
            app.state.model_version = "local"
            print("Loaded local model")
        except:
            raise RuntimeError("Could not load any model")

# Startup event
@app.on_event("startup")
async def startup_event():
    load_model()

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Create input DataFrame
        input_data = pd.DataFrame([request.dict()])
        
        # Predict risk probability
        risk_prob = app.state.model.predict(input_data)[0]
        
        # Convert to credit score (300-850 range)
        credit_score = int(300 + (1 - risk_prob) * 550)
        
        # Simple loan recommendation (business logic)
        loan_amount = None
        loan_duration = None
        if credit_score > 700:
            loan_amount = min(5000, request.Monetary * 0.5)
            loan_duration = 12
        elif credit_score > 600:
            loan_amount = min(2000, request.Monetary * 0.3)
            loan_duration = 6
        
        return PredictionResponse(
            customer_id=request.CustomerId,
            risk_probability=risk_prob,
            credit_score=credit_score,
            loan_amount=loan_amount,
            loan_duration=loan_duration,
            model_version=app.state.model_version
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": hasattr(app.state, "model"),
        "timestamp": datetime.utcnow().isoformat()
    }