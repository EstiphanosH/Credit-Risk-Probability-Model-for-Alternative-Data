import os
import logging
from datetime import datetime
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(filepath):
    data = pd.read_csv(filepath)
    # If TransactionStartTime is present, parse as datetime
    if 'TransactionStartTime' in data.columns:
        data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'], errors='coerce')
    required = {'is_high_risk', 'TransactionStartTime'}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    return data

def split_data(X, y, test_size=0.2, val_size=0.2):
    # Time-based split
    df = X.copy()
    df['is_high_risk'] = y
    df = df.sort_values('TransactionStartTime')
    n = len(df)
    test_n = int(n * test_size)
    val_n = int(n * val_size)
    train_n = n - test_n - val_n
    train = df.iloc[:train_n]
    val = df.iloc[train_n:train_n+val_n]
    test = df.iloc[train_n+val_n:]
    X_train, y_train = train.drop(columns=['is_high_risk']), train['is_high_risk']
    X_val, y_val = val.drop(columns=['is_high_risk']), val['is_high_risk']
    X_test, y_test = test.drop(columns=['is_high_risk']), test['is_high_risk']
    return X_train, X_val, X_test, y_train, y_val, y_test

def get_preprocessor(X):
    numeric_features = X.select_dtypes(include=['number']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    # Remove TransactionStartTime and CustomerId from features
    for col in ['TransactionStartTime', 'CustomerId']:
        if col in numeric_features:
            numeric_features.remove(col)
        if col in categorical_features:
            categorical_features.remove(col)
    numeric_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    return ColumnTransformer([
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
        data = load_data("../data/processed/processed_data_train.csv")
        X = data.drop(columns=["is_high_risk"])
        y = data["is_high_risk"]

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

        # Check label distribution
        logger.info(f"Train label distribution: {y_train.value_counts().to_dict()}")
        logger.info(f"Validation label distribution: {y_val.value_counts().to_dict()}")
        logger.info(f"Test label distribution: {y_test.value_counts().to_dict()}")
        if y_train.nunique() < 2:
            raise ValueError("Training data contains only one class. Check your label generation pipeline.")

        # Preprocessing
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
                "params": {'classifier__n_estimators': [100], 'classifier__learning_rate': [0.01, 0.1], 'classifier__max_depth': [3, 5]}
            }
        }

        best_score = -1
        best_model = None
        best_name = None

        for name, cfg in models.items():
            logger.info(f"Training {name}...")
            pipe = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', cfg["model"])
            ])
            grid_search = GridSearchCV(pipe, cfg["params"], cv=3, scoring='roc_auc', n_jobs=-1, error_score='raise')
            grid_search.fit(X_train, y_train)
            score = grid_search.best_score_
            logger.info(f"{name} best CV ROC AUC: {score:.4f}")
            if score > best_score:
                best_score = score
                best_model = grid_search.best_estimator_
                best_name = name

        logger.info(f"Best model: {best_name} with ROC AUC: {best_score:.4f}")

        # Evaluate on validation and test sets
        val_score = best_model.score(X_val, y_val)
        test_score = best_model.score(X_test, y_test)
        logger.info(f"Validation set accuracy: {val_score:.4f}")
        logger.info(f"Test set accuracy: {test_score:.4f}")

        # Log with MLflow
        with mlflow.start_run():
            mlflow.log_param("model", best_name)
            mlflow.log_metric("val_accuracy", val_score)
            mlflow.log_metric("test_accuracy", test_score)
            mlflow.sklearn.log_model(best_model, "model")

        logger.info(f"Training completed in {datetime.now() - start_time}")

    except Exception as e:
        logger.error("Model training failed")
        logger.error(str(e))
        raise

if __name__ == "__main__":
    train_model()