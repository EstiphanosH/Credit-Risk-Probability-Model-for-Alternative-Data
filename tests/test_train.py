# tests/test_train.py
import pytest
import pandas as pd
import numpy as np
import joblib
import mlflow
from unittest.mock import patch, MagicMock
from src.train import load_data, train_model
from sklearn.exceptions import NotFittedError

# Sample test data structure matching processed_data.csv
TEST_DATA = pd.DataFrame({
    "CustomerId": [101, 102, 103, 104, 105],
    "is_high_risk": [0, 1, 0, 1, 0],
    "Total_Amount": [1500, 200, 3000, 500, 4500],
    "Avg_Amount": [500, 200, 1000, 500, 1500],
    "Transaction_Count": [3, 1, 3, 1, 3],
    "ProductCategory": ["Electronics", "Clothing", "Electronics", "Home", "Electronics"],
    "Recency": [15, 120, 30, 90, 10],
    "Frequency": [5, 1, 8, 2, 10],
    "Monetary": [2500, 200, 4000, 500, 6000]
})

@pytest.fixture
def mock_data(tmp_path):
    """Create a temporary processed data file"""
    data_path = tmp_path / "processed_data.csv"
    TEST_DATA.to_csv(data_path, index=False)
    return str(data_path)

def test_load_data_valid(mock_data):
    """Test loading valid data file"""
    df = load_data(mock_data)
    assert df.shape == (5, 9)
    assert "CustomerId" in df.columns
    assert "is_high_risk" in df.columns

def test_load_data_missing_file():
    """Test handling of missing data file"""
    with pytest.raises(FileNotFoundError):
        load_data("non_existent_file.csv")

def test_load_data_invalid_columns(tmp_path):
    """Test data with missing required columns"""
    invalid_data = TEST_DATA.drop(columns=["is_high_risk"])
    data_path = tmp_path / "invalid_data.csv"
    invalid_data.to_csv(data_path, index=False)
    
    with pytest.raises(ValueError) as excinfo:
        load_data(str(data_path))
    assert "Missing required columns" in str(excinfo.value)

@patch("src.train.mlflow.start_run")
@patch("src.train.GridSearchCV")
@patch("src.train.train_test_split")
def test_train_model_success(
    mock_split, mock_grid, mock_start_run, mock_data, tmp_path, caplog
):
    """Test successful model training workflow"""
    # Mock dependencies
    mock_model = MagicMock()
    mock_model.predict.return_value = [0, 1, 0, 1, 0]
    mock_model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7], 
                                                     [0.9, 0.1], [0.4, 0.6], 
                                                     [0.85, 0.15]])
    
    mock_grid.return_value.best_estimator_ = mock_model
    mock_grid.return_value.best_params_ = {"C": 1.0}
    mock_grid.return_value.fit.return_value = None
    
    mock_split.return_value = (
        TEST_DATA.drop(columns=["is_high_risk", "CustomerId"]),  # X_train
        TEST_DATA.drop(columns=["is_high_risk", "CustomerId"]),  # X_test
        TEST_DATA["is_high_risk"],  # y_train
        TEST_DATA["is_high_risk"]   # y_test
    )
    
    # Mock MLflow
    mock_active_run = MagicMock()
    mock_active_run.info.run_id = "test_run_id"
    mlflow.active_run = MagicMock(return_value=mock_active_run)
    
    # Run training
    train_model()
    
    # Verify logging
    assert "Starting model training" in caplog.text
    assert "Data split" in caplog.text
    assert "Best model is" in caplog.text
    assert "Best model saved locally" in caplog.text
    
    # Verify model saving
    assert (tmp_path / "models/best_model.pkl").exists()

@patch("src.train.mlflow.start_run")
def test_train_model_data_loading_error(mock_start_run, caplog):
    """Test error handling during data loading"""
    with pytest.raises(FileNotFoundError):
        train_model()
    assert "File not found" in caplog.text

@patch("src.train.mlflow.start_run")
@patch("src.train.train_test_split")
def test_train_model_training_error(mock_split, mock_start_run, mock_data, caplog):
    """Test error handling during model training"""
    # Force an error during training
    mock_split.side_effect = ValueError("Test error")
    
    with pytest.raises(ValueError):
        train_model()
    
    assert "Test error" in caplog.text
    assert "An error occurred during model training" in caplog.text

def test_model_serialization(tmp_path):
    """Test model can be serialized and deserialized"""
    # Create a simple model
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    
    # Train on sample data
    X = TEST_DATA[["Total_Amount", "Recency", "Frequency"]]
    y = TEST_DATA["is_high_risk"]
    model.fit(X, y)
    
    # Save model
    model_path = tmp_path / "test_model.pkl"
    joblib.dump(model, model_path)
    
    # Load and validate
    loaded_model = joblib.load(model_path)
    assert loaded_model.predict(X)[0] == model.predict(X)[0]

@patch("src.train.mlflow.sklearn.log_model")
def test_mlflow_model_logging(mock_log_model, mock_data):
    """Test MLflow model logging functionality"""
    # Mock training process
    with patch("src.train.GridSearchCV") as mock_grid:
        mock_model = MagicMock()
        mock_grid.return_value.best_estimator_ = mock_model
        mock_grid.return_value.fit.return_value = None
        
        train_model()
        
        # Verify MLflow logging was called
        mock_log_model.assert_called()
        args, kwargs = mock_log_model.call_args
        assert kwargs["artifact_path"] == "LogisticRegression"
        assert "signature" in kwargs

@patch("src.train.mlflow.set_tracking_uri")
@patch("src.train.mlflow.set_experiment")
def test_mlflow_setup(mock_set_experiment, mock_set_tracking_uri):
    """Test MLflow initialization"""
    from src.train import mlflow
    mlflow.set_tracking_uri.assert_called_with("sqlite:///mlflow.db")
    mlflow.set_experiment.assert_called_with("Credit_Risk_Model")

def test_feature_importance_logging(tmp_path, mock_data):
    """Test feature importance logging"""
    # Mock a model with feature_importances_
    with patch("src.train.GridSearchCV") as mock_grid, \
         patch("src.train.mlflow.log_artifact") as mock_log_artifact:
        
        class MockModel:
            def feature_importances_(self):
                return np.array([0.4, 0.3, 0.2, 0.1])
        
        mock_model = MagicMock()
        mock_model.named_steps = {"classifier": MockModel()}
        mock_grid.return_value.best_estimator_ = mock_model
        
        train_model()
        
        # Verify artifact logging
        mock_log_artifact.assert_called_with("feature_importance.csv")

def test_model_metrics_calculation():
    """Test model metrics are calculated correctly"""
    from src.train import train_model  # Import to access inner functions
    
    # Test data
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_proba = np.array([0.1, 0.9, 0.6, 0.8])
    
    # Calculate expected metrics
    expected_accuracy = 0.75
    expected_precision = 0.6667  # 2/3
    expected_recall = 1.0        # 2/2
    expected_f1 = 0.8            # 2*(0.6667*1)/(0.6667+1)
    expected_roc_auc = 0.875      # (1+0.75)/2
    
    # Calculate actual metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    # Verify calculations
    assert accuracy == expected_accuracy
    assert round(precision, 4) == expected_precision
    assert recall == expected_recall
    assert round(f1, 4) == expected_f1
    assert roc_auc == expected_roc_auc