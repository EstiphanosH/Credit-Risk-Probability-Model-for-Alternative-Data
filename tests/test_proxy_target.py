import os
# MUST BE SET BEFORE IMPORTING ANY SCIKIT-LEARN COMPONENTS
os.environ["OMP_NUM_THREADS"] = "1"

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.proxy_target import RiskLabelGenerator
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RiskLabelTest")

# Suppress remaining warnings
warnings.filterwarnings("ignore", category=UserWarning)

@pytest.fixture
def sample_data():
    """Generate test transaction data with clear risk patterns"""
    np.random.seed(42)
    customers = [f"CUST-{i}" for i in range(1, 101)]
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(100)]
    
    # Create clear risk segments
    data = []
    for i, cust in enumerate(customers):
        # High-risk customers (20%)
        if i < 20:
            n_transactions = np.random.randint(1, 5)
            last_date = datetime(2023, 4, 1)  # Inactive for 30+ days
            amounts = np.random.uniform(10, 50, n_transactions)
        
        # Medium-risk (30%)
        elif i < 50:
            n_transactions = np.random.randint(5, 15)
            last_date = datetime(2023, 4, 20)  # Recently active
            amounts = np.random.uniform(50, 200, n_transactions)
        
        # Low-risk (50%)
        else:
            n_transactions = np.random.randint(20, 40)
            last_date = datetime(2023, 4, 25)  # Very active
            amounts = np.random.uniform(100, 500, n_transactions)
        
        # Generate transactions
        transaction_dates = sorted([last_date - timedelta(days=int(x)) 
                                  for x in np.linspace(0, 30, n_transactions)])
        
        for amt, date in zip(amounts, transaction_dates):
            data.append({
                "TransactionId": f"TXN-{len(data)}",
                "CustomerId": cust,
                "TransactionStartTime": date,
                "Amount": amt,
                "FraudResult": 1 if i < 20 and np.random.rand() < 0.3 else 0
            })
    
    return pd.DataFrame(data)

def test_rfm_calculation(sample_data):
    """Test RFM feature calculation"""
    labeler = RiskLabelGenerator()
    rfm = labeler._calculate_rfm(sample_data)
    
    # Validate structure
    assert set(rfm.columns) == {"CustomerId", "recency", "frequency", "monetary"}
    assert len(rfm) == len(sample_data["CustomerId"].unique())
    
    # Test specific customer
    test_cust = sample_data["CustomerId"].iloc[0]
    cust_data = sample_data[sample_data["CustomerId"] == test_cust]
    
    # Frequency check
    assert rfm[rfm["CustomerId"] == test_cust]["frequency"].values[0] == len(cust_data)
    
    # Monetary check
    assert np.isclose(
        rfm[rfm["CustomerId"] == test_cust]["monetary"].values[0],
        cust_data["Amount"].sum(),
        atol=0.01
    )

def test_risk_labeling(sample_data):
    """Test end-to-end risk labeling"""
    labeler = RiskLabelGenerator(
        recency_weight=0.6,
        frequency_weight=0.2,
        monetary_weight=0.2
    )
    labels = labeler.fit_transform(sample_data)
    
    # Validate output
    assert "is_high_risk" in labels.columns
    assert set(labels["is_high_risk"].unique()) == {0, 1}
    
    # Check high-risk count (should be ~20%)
    high_risk_pct = labels["is_high_risk"].mean()
    assert 0.18 <= high_risk_pct <= 0.22
    
    # Validate risk characteristics
    rfm = labeler._calculate_rfm(sample_data)
    labeled_rfm = rfm.merge(labels, on="CustomerId")
    
    high_risk = labeled_rfm[labeled_rfm["is_high_risk"] == 1]
    low_risk = labeled_rfm[labeled_rfm["is_high_risk"] == 0]
    
    # High-risk should have higher recency
    assert high_risk["recency"].mean() > low_risk["recency"].mean() + 10
    
    # High-risk should have lower frequency
    assert high_risk["frequency"].mean() < low_risk["frequency"].mean() / 2
    
    # High-risk should have lower monetary
    assert high_risk["monetary"].mean() < low_risk["monetary"].mean() / 2

def test_reproducibility(sample_data):
    """Test results are reproducible with same seed"""
    labeler1 = RiskLabelGenerator(random_state=42)
    labels1 = labeler1.fit_transform(sample_data)

    labeler2 = RiskLabelGenerator(random_state=42)
    labels2 = labeler2.fit_transform(sample_data)

    # Should be identical
    pd.testing.assert_frame_equal(labels1, labels2)

    # For different seeds - accept either outcome
    labeler3 = RiskLabelGenerator(random_state=99)
    labels3 = labeler3.fit_transform(sample_data)
    if labels1.equals(labels3):
        logger.info("Different random_state produced same result (data is well-separated)")
    else:
        logger.info("Different random_state produced different result")

def test_new_data_handling(sample_data):
    """Test handling of new unseen customers"""
    # Train on first 80 customers
    train_data = sample_data[sample_data["CustomerId"].isin(sample_data["CustomerId"].unique()[:80])]
    
    # Test on last 20 customers
    test_data = sample_data[sample_data["CustomerId"].isin(sample_data["CustomerId"].unique()[80:])]
    
    labeler = RiskLabelGenerator()
    labeler.fit(train_data)
    test_labels = labeler.transform(test_data)
    
    # Should process new customers
    assert len(test_labels) == 20
    assert test_labels["is_high_risk"].notnull().all()
    
    # Should maintain risk characteristics
    test_rfm = labeler._calculate_rfm(test_data)
    test_full = test_rfm.merge(test_labels, on="CustomerId")
    
    # New high-risk customers should have similar profiles
    high_risk = test_full[test_full["is_high_risk"] == 1]
    if not high_risk.empty:
        assert high_risk["recency"].mean() > 25  # Should be inactive
        assert high_risk["frequency"].mean() < 5  # Low transaction count

def test_cluster_report(sample_data):
    """Test cluster reporting functionality"""
    labeler = RiskLabelGenerator()
    labeler.fit(sample_data)
    report = labeler.generate_cluster_report(sample_data)
    
    # Validate report structure
    expected_cols = [
        "cluster", "customer_count", "recency_mean", "frequency_mean",
        "monetary_mean", "recency_std", "frequency_std", "monetary_std",
        "is_high_risk", "customer_pct"
    ]
    assert list(report.columns) == expected_cols
    
    # Validate high-risk identification
    high_risk_cluster = report[report["is_high_risk"] > 0.5]["cluster"].iloc[0]
    assert high_risk_cluster == labeler.high_risk_cluster_
    
    # Should have correct risk proportions
    assert report["customer_pct"].sum() == 1.0
    assert report.loc[high_risk_cluster, "customer_pct"] == pytest.approx(0.2, abs=0.05)

if __name__ == "__main__":
    pytest.main([__file__])