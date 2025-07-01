import logging
import os
import joblib
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_data(filepath: str) -> pd.DataFrame:
    """Load and validate processed data"""
    logger.info(f"Loading data from {filepath}")
    if not os.path.exists(filepath):
        logger.error("Data file not found")
        raise FileNotFoundError(f"Data file {filepath} does not exist")
    
    data = pd.read_csv(filepath)
    
    # Validate required columns
    required_columns = {"is_high_risk", "CustomerId", "TransactionStartTime"}
    missing = required_columns - set(data.columns)
    if missing:
        logger.error(f"Missing required columns: {missing}")
        raise ValueError(f"Dataset missing required columns: {missing}")
    
    return data

def split_data(X, y, time_col='TransactionStartTime'):
    """Split data into train/validation/test sets with time ordering"""
    logger.info("Splitting data with time-based ordering")
    
    # Sort by transaction time
    sorted_idx = X[time_col].argsort()
    X_sorted = X.iloc[sorted_idx].reset_index(drop=True)
    y_sorted = y.iloc[sorted_idx].reset_index(drop=True)
    
    # Calculate split indices
    train_end = int(0.7 * len(X))
    val_end = train_end + int(0.15 * len(X))
    
    # Split datasets
    X_train = X_sorted.iloc[:train_end]
    X_val = X_sorted.iloc[train_end:val_end]
    X_test = X_sorted.iloc[val_end:]
    
    y_train = y_sorted.iloc[:train_end]
    y_val = y_sorted.iloc[train_end:val_end]
    y_test = y_sorted.iloc[val_end:]
    
    logger.info(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def get_preprocessor(X):
    """Create preprocessing pipeline"""
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

def train_model():
    try:
        logger.info("Starting model training pipeline")
        start_time = datetime.now()
        
        # MLflow setup
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("Credit_Risk_Model")
        
        # Load data
        data = load_data("../data/processed/processed_data.csv")
        X = data.drop(columns=["is_high_risk"])
        y = data["is_high_risk"]
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        
        # Get preprocessor
        preprocessor = get_preprocessor(X_train)
        
        # Models to evaluate
        models = {
            "LogisticRegression": {
                "model": LogisticRegression(solver='liblinear', random_state=42, max_iter=1000),
                "params": {'classifier__C': [0.01, 0.1, 1], 'classifier__penalty': ['l1', 'l2']}
            },
            "RandomForest": {
                "model": RandomForestClassifier(random_state=42),
                "params": {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [10, None]}
            },
            "XGBoost": {
                "model": XGBClassifier(random_state=42, eval_metric='logloss'),
                "params": {'classifier__n_estimators': [100], 
                          'classifier__learning_rate': [0.01, 0.1],
                          'classifier__max_depth': [3, 5]}
            }
        }
        
        best_model = None
        best_score = -1
        
        for name, config in models.items():
            with mlflow.start_run(run_name=f"{name}_Experiment", nested=True):
                # Create pipeline
                pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('classifier', config["model"])
                ])
                
                # Hyperparameter tuning with time-series cross-validation
                ts_cv = TimeSeriesSplit(n_splits=5)
                grid_search = GridSearchCV(
                    pipeline,
                    param_grid=config["params"],
                    cv=ts_cv,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=1
                )
                
                logger.info(f"Training {name} with time-series CV...")
                grid_search.fit(X_train, y_train)
                
                # Evaluate on validation set
                val_score = roc_auc_score(y_val, grid_search.predict_proba(X_val)[:, 1])
                mlflow.log_metric("val_roc_auc", val_score)
                
                # Track best model
                if val_score > best_score:
                    best_score = val_score
                    best_model = grid_search.best_estimator_
                    best_model_name = name
                
                logger.info(f"{name} validation AUC: {val_score:.4f}")
        
        # Final evaluation on test set
        with mlflow.start_run(run_name="Final_Evaluation"):
            test_probs = best_model.predict_proba(X_test)[:, 1]
            test_auc = roc_auc_score(y_test, test_probs)
            
            # Log metrics
            mlflow.log_metrics({
                "test_roc_auc": test_auc,
                "test_accuracy": accuracy_score(y_test, best_model.predict(X_test)),
                "test_precision": precision_score(y_test, best_model.predict(X_test)),
                "test_recall": recall_score(y_test, best_model.predict(X_test))
            })
            
            # Save and register model
            mlflow.sklearn.log_model(
                best_model, 
                "model", 
                registered_model_name="CreditRiskModel"
            )
            logger.info(f"Registered best model ({best_model_name}) with test AUC: {test_auc:.4f}")
        
        # Save model locally
        os.makedirs("models", exist_ok=True)
        joblib.dump(best_model, "models/credit_risk_model.pkl")
        logger.info("Model saved to models/credit_risk_model.pkl")
        
        # Log pipeline duration
        duration = (datetime.now() - start_time).total_seconds() / 60
        logger.info(f"Training completed in {duration:.2f} minutes")
        
        return best_model
    
    except Exception as e:
        logger.exception("Model training failed")
        raise

if __name__ == "__main__":
    train_model()