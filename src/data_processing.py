import os
import logging
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Feature Engineering: Five Cs ---
class FiveCsFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering pipeline for credit risk using the Five Cs framework.
    """
    def __init__(self, snapshot_date=None):
        self.snapshot_date = snapshot_date

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['TransactionStartTime']):
            df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
        snapshot = self.snapshot_date or df['TransactionStartTime'].max() + timedelta(days=30)
        agg = df.groupby('CustomerId').agg(
            recency=('TransactionStartTime', lambda x: (snapshot - x.max()).days),
            frequency=('TransactionStartTime', 'count'),
            monetary=('Amount', lambda x: x[x > 0].sum()),
            avg_spend=('Amount', lambda x: x[x > 0].mean() if (x > 0).any() else 0),
            amount_std=('Amount', 'std'),
            lifetime_value=('Amount', lambda x: x[x > 0].sum()),
            net_cash_flow=('Amount', 'sum'),
            fraud_count=('FraudResult', 'sum'),
            category_diversity=('ProductCategory', 'nunique'),
        )
        # Add TransactionStartTime for downstream splitting
        last_txn = df.groupby('CustomerId')['TransactionStartTime'].max()
        agg['TransactionStartTime'] = last_txn
        # Restore ProductCategory for WoE encoding
        first_cat = df.groupby('CustomerId')['ProductCategory'].first()
        agg['ProductCategory'] = first_cat
        return agg.reset_index()

# --- Outlier Capping ---
class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Caps outliers in numeric columns using the IQR method.
    """
    def __init__(self, factor=1.5):
        self.factor = factor
        self.bounds = {}

    def fit(self, X, y=None):
        for col in X.select_dtypes(include=np.number).columns:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            self.bounds[col] = (q1 - self.factor * iqr, q3 + self.factor * iqr)
        return self

    def transform(self, X):
        Xc = X.copy()
        for col, (lb, ub) in self.bounds.items():
            Xc[col] = Xc[col].clip(lower=lb, upper=ub)
        return Xc

# --- WoE Encoder (simple version) ---
class WoETransformer(BaseEstimator, TransformerMixin):
    """
    Weight of Evidence encoder for categorical variables.
    """
    def __init__(self, target='is_high_risk', min_samples=1):
        self.target = target
        self.min_samples = min_samples
        self.woe_dict = {}

    def fit(self, X, y):
        for col in X.select_dtypes(include=['object', 'category']).columns:
            df = pd.DataFrame({col: X[col], self.target: y})
            stats = df.groupby(col)[self.target].agg(['sum', 'count'])
            stats = stats[stats['count'] >= self.min_samples]
            stats['non_event'] = stats['count'] - stats['sum']
            event_total = stats['sum'].sum()
            non_event_total = stats['non_event'].sum()
            stats['woe'] = np.log(
                (stats['sum'] / event_total + 1e-6) /
                (stats['non_event'] / non_event_total + 1e-6)
            )
            self.woe_dict[col] = stats['woe'].to_dict()
        return self

    def transform(self, X):
        Xc = X.copy()
        for col, mapping in self.woe_dict.items():
            Xc[col] = Xc[col].map(mapping).fillna(0)
        return Xc

# --- Proxy Risk Label Generator ---
class RiskLabelGenerator(BaseEstimator, TransformerMixin):
    """
    Generates proxy risk labels using KMeans clustering on engineered features.
    """
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.scaler = RobustScaler()
        self.high_risk_cluster_ = None

    def fit(self, X, y=None):
        features = ['recency', 'amount_std', 'lifetime_value']
        X_scaled = self.scaler.fit_transform(X[features].fillna(0))
        self.kmeans.fit(X_scaled)
        centers = self.kmeans.cluster_centers_
        # High risk: highest recency, highest std, lowest lifetime_value
        risk_scores = centers[:, 0] + centers[:, 1] - centers[:, 2]
        self.high_risk_cluster_ = np.argmax(risk_scores)
        return self

    def transform(self, X):
        features = ['recency', 'amount_std', 'lifetime_value']
        X_scaled = self.scaler.transform(X[features].fillna(0))
        clusters = self.kmeans.predict(X_scaled)
        Xc = X.copy()
        Xc['is_high_risk'] = (clusters == self.high_risk_cluster_).astype(int)
        return Xc

# --- Main Data Processing Pipeline ---
def process_data(input_csv, output_prefix):
    """
    Full data processing pipeline: feature engineering, risk labeling, outlier capping, WoE encoding, and train/test split.
    """
    logging.info(f"Loading data from {input_csv}")
    df = pd.read_csv(input_csv, parse_dates=['TransactionStartTime'])

    # 1. Feature engineering
    logging.info("Engineering features")
    feature_engineer = FiveCsFeatureEngineer()
    features = feature_engineer.fit_transform(df)

    # 2. Risk label generation
    logging.info("Generating proxy risk labels")
    labeler = RiskLabelGenerator()
    features = labeler.fit_transform(features)

    # 3. Outlier capping
    logging.info("Capping outliers")
    capper = OutlierCapper()
    features_capped = capper.fit_transform(features)

    # 4. WoE encoding for ProductCategory
    logging.info("Applying WoE encoding")
    if 'ProductCategory' in features_capped.columns:
        woe_encoder = WoETransformer(target='is_high_risk')
        features_capped = woe_encoder.fit_transform(features_capped, features_capped['is_high_risk'])

    # 5. Handle missing values
    logging.info("Imputing missing values")
    num_cols = features_capped.select_dtypes(include=np.number).columns
    imputer = SimpleImputer(strategy='median')
    features_capped[num_cols] = imputer.fit_transform(features_capped[num_cols])

    # 6. Split into train/test and save both
    logging.info("Splitting into train/test sets")
    features_capped = features_capped.sort_values("TransactionStartTime")
    n = len(features_capped)
    test_size = int(n * 0.2)
    train = features_capped.iloc[:-test_size]
    test = features_capped.iloc[-test_size:]

    # --- Ensure TransactionStartTime is present in both train and test ---
    if 'CustomerId' in df.columns and 'TransactionStartTime' in df.columns:
        custid_to_time = df.groupby('CustomerId')['TransactionStartTime'].max()
        train = train.copy()
        test = test.copy()
        train.loc[:, 'TransactionStartTime'] = train['CustomerId'].map(custid_to_time)
        test.loc[:, 'TransactionStartTime'] = test['CustomerId'].map(custid_to_time)

    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    train_path = output_prefix.replace('.csv', '_train.csv')
    test_path = output_prefix.replace('.csv', '_test.csv')
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    logging.info(f"Processed train data saved to {train_path}")
    logging.info(f"Processed test data saved to {test_path}")
    return train, test

if __name__ == "__main__":
    input_csv = "../data/raw/data.csv"
    output_prefix = "../data/processed/processed_data.csv"
    process_data(input_csv, output_prefix)
    import pandas as pd
    df = pd.read_csv("../data/processed/processed_data_train.csv")
    print(df['is_high_risk'].value_counts())