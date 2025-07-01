import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.train import load_data, split_data, get_preprocessor
from sklearn.pipeline import Pipeline

# Sample test data
@pytest.fixture
def sample_data():
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    return pd.DataFrame({
        'CustomerId': range(100),
        'is_high_risk': np.random.randint(0, 2, 100),
        'TransactionStartTime': dates,
        'feature1': np.random.randn(100),
        'feature2': np.random.choice(['A', 'B', 'C'], 100)
    })

def test_load_data_valid(tmp_path, sample_data):
    """Test loading valid data file"""
    data_path = tmp_path / "data.csv"
    sample_data.to_csv(data_path, index=False)
    data = load_data(str(data_path))
    assert data.shape == (100, 5)

def test_load_data_missing_columns(tmp_path, sample_data):
    """Test handling of missing columns"""
    data_path = tmp_path / "data.csv"
    sample_data.drop(columns=['is_high_risk']).to_csv(data_path, index=False)
    with pytest.raises(ValueError):
        load_data(str(data_path))

def test_time_based_split(sample_data):
    """Test time-based data splitting"""
    X = sample_data.drop(columns=['is_high_risk'])
    y = sample_data['is_high_risk']
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Verify sizes
    assert len(X_train) == 70
    assert len(X_val) == 15
    assert len(X_test) == 15
    
    # Verify chronological order
    assert X_train['TransactionStartTime'].max() < X_val['TransactionStartTime'].min()
    assert X_val['TransactionStartTime'].max() < X_test['TransactionStartTime'].min()

def test_preprocessor(sample_data):
    """Test preprocessing pipeline creation"""
    X = sample_data.drop(columns=['is_high_risk', 'TransactionStartTime'])
    preprocessor = get_preprocessor(X)
    
    # Verify structure
    assert isinstance(preprocessor, ColumnTransformer)
    assert 'num' in preprocessor.named_transformers_
    assert 'cat' in preprocessor.named_transformers_

@patch("src.train.mlflow")
@patch("src.train.GridSearchCV")
def test_training_pipeline(mock_grid, mock_mlflow, sample_data, tmp_path):
    """Test full training pipeline"""
    from src.train import train_model
    
    # Mock dependencies
    mock_grid.return_value.fit.return_value = None
    mock_grid.return_value.best_estimator_ = MagicMock(spec=Pipeline)
    mock_grid.return_value.best_estimator_.predict_proba.return_value = np.array([[0.2, 0.8]] * 15)
    mock_grid.return_value.best_estimator_.predict.return_value = [1] * 15
    
    # Create temp data file
    data_path = tmp_path / "processed_data.csv"
    sample_data.to_csv(data_path, index=False)
    
    # Mock load_data to use our temp file
    with patch("src.train.load_data") as mock_load:
        mock_load.return_value = sample_data
        
        # Run training
        model = train_model()
        
        # Verify pipeline executed
        assert model is not None
        mock_mlflow.start_run.assert_called()
        mock_mlflow.log_metrics.assert_called()
        
        # Verify model saving
        assert (tmp_path / "models/credit_risk_model.pkl").exists()

def test_split_data_no_time_column(sample_data):
    """Test error handling when time column is missing"""
    X = sample_data.drop(columns=['TransactionStartTime', 'is_high_risk'])
    y = sample_data['is_high_risk']
    
    with pytest.raises(KeyError):
        split_data(X, y)