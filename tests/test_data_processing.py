import pytest
from src.data_processing import RFMCalculator
import pandas as pd

# Test data
@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'CustomerId': [1, 1, 2, 2],
        'Amount': [100, 200, 50, 75],
        'TransactionStartTime': ['2025-06-20', '2025-06-22', '2025-06-18', '2025-06-15']
    })

def test_rfm_calculator(sample_data):
    """Test RFM feature calculation"""
    transformer = RFMCalculator(snapshot_date='2025-06-25')
    transformed = transformer.transform(sample_data)
    
    # Add your assertions here
    assert 'Recency' in transformed.columns
    assert 'Frequency' in transformed.columns
    assert 'Monetary' in transformed.columns
