from pydantic import BaseModel
from typing import List, Dict, Optional

class PredictionRequest(BaseModel):
    CustomerId: str
    Total_Amount: float
    Avg_Amount: float
    Transaction_Count: int
    Recency: int
    Frequency: int
    Monetary: float
    ProductCategory: str
    ChannelId: str
    PricingStrategy: str
    TransactionStartTime: str

class PredictionResponse(BaseModel):
    customer_id: str
    risk_probability: float
    credit_score: int
    loan_amount: Optional[float] = None
    loan_duration: Optional[int] = None
    model_version: str