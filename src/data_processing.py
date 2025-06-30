# src/data_processing.py 
import os
import logging
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, RobustScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === Basel III Compliance Metrics ===
def calculate_regulatory_capital(PD, LGD=0.45, EAD=1000):
    """Calculate Basel III regulatory capital requirements"""
    # Simplified Basel III formula for demonstration
    R = 0.12 * (1 - np.exp(-50 * PD)) / (1 - np.exp(-50)) + \
         0.24 * (1 - (1 - np.exp(-50 * PD)) / (1 - np.exp(-50)))
    K = LGD * stats.norm.cdf((1 - R)**-0.5 * stats.norm.ppf(PD) + 
                          (R / (1 - R))**0.5 * stats.norm.ppf(0.999)) - PD * LGD
    return K * EAD

# === Feature Engineering Aligned with Five Cs ===
class FiveCsFeatureEngineer(BaseEstimator, TransformerMixin):
    """Creates features aligned with the Five Cs of Credit framework"""
    def __init__(self, snapshot_date=None):
        self.snapshot_date = snapshot_date
        
    def fit(self, X, y=None):
        # Set snapshot date as max transaction date + 1 day
        if self.snapshot_date is None:
            self.snapshot_date = X['TransactionStartTime'].max() + timedelta(days=1)
        return self
        
    def transform(self, X):
        X = X.copy()
        # Convert to datetime
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
        
        # Character Features (Payment reliability)
        char_features = X.groupby('CustomerId').agg(
            payment_consistency=('Amount', lambda x: (x < 0).mean()),
            fraud_count=('FraudResult', 'sum'),
            txn_variability=('Amount', lambda x: np.std(x[x > 0]) if len(x[x > 0]) > 1 else 0)
        ).reset_index()
        
        # Capacity Features (Spending ability)
        cap_features = X[X['Amount'] > 0].groupby('CustomerId').agg(
            avg_spend=('Amount', 'mean'),
            max_spend=('Amount', 'max'),
            spend_velocity=('TransactionStartTime', 
                lambda x: len(x) / ((x.max() - x.min()).days) if len(x) > 1 and (x.max() - x.min()).days > 0 else 0
            )).reset_index()
        
        # Capital Features (Cumulative value)
        cap_val_features = X.groupby('CustomerId').agg(
            lifetime_value=('Amount', lambda x: x[x > 0].sum()),
            net_cash_flow=('Amount', 'sum')
        ).reset_index()
        
        # Collateral Proxy (Product value concentration)
        collateral_features = X[X['Amount'] > 0].groupby(['CustomerId', 'ProductCategory']).agg(
            category_value=('Amount', 'sum')
        ).reset_index().groupby('CustomerId').agg(
            top_category_share=('category_value', lambda x: x.max()/x.sum())
        ).reset_index()
        
        # Conditions Features (Temporal patterns)
        time_features = X.groupby('CustomerId').agg(
            last_txn_date=('TransactionStartTime', 'max'),
            spending_slope=('TransactionStartTime', lambda x: self._calculate_spending_slope(X.loc[x.index])),
            holiday_spending_ratio=('TransactionStartTime', lambda x: self._holiday_spending_ratio(X.loc[x.index]))
        ).reset_index()
        time_features['recency'] = (self.snapshot_date - time_features['last_txn_date']).dt.days
        time_features.drop(columns=['last_txn_date'], inplace=True)
        
        # Merge all features
        features = char_features.merge(cap_features, on='CustomerId', how='left')\
                                .merge(cap_val_features, on='CustomerId', how='left')\
                                .merge(collateral_features, on='CustomerId', how='left')\
                                .merge(time_features, on='CustomerId', how='left')
        
        # Add Basel III metrics
        features['expected_loss'] = features.apply(
            lambda row: calculate_regulatory_capital(0.05, 0.45, row['lifetime_value'] * 0.1), 
            axis=1
        )
        
        return features

    def _calculate_spending_slope(self, customer_df):
        """Calculate linear trend of spending over time"""
        # Only use positive-amount transactions
        df = customer_df[customer_df['Amount'] > 0].sort_values('TransactionStartTime')
        if len(df) < 3:
            return 0
        x = np.arange(len(df))
        y = df['Amount'].values
        return np.polyfit(x, y, 1)[0]  # Return slope

    def _holiday_spending_ratio(self, customer_df):
        """Calculate holiday vs non-holiday spending ratio"""
        customer_df['is_holiday'] = customer_df['TransactionStartTime'].dt.month.isin([11,12])
        holiday_spend = customer_df[customer_df['is_holiday'] & (customer_df['Amount'] > 0)]['Amount'].sum()
        non_holiday_spend = customer_df[~customer_df['is_holiday'] & (customer_df['Amount'] > 0)]['Amount'].sum()
        return holiday_spend / (non_holiday_spend + 1e-5) if non_holiday_spend > 0 else 0

# === Weight of Evidence (WoE) Encoder ===
class WoETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target='is_high_risk', min_samples=100):
        self.target = target
        self.min_samples = min_samples
        self.woe_dict = {}

    def fit(self, X, y=None):
        df = X.copy()
        if y is not None:
            df[self.target] = y

        epsilon = 1e-6  # Small value to avoid log(0)
        for col in X.select_dtypes(include=['object', 'category']).columns:
            # Calculate WoE per category
            agg_df = df.groupby(col)[self.target].agg(['count', 'mean'])
            agg_df = agg_df[agg_df['count'] > self.min_samples]
            agg_df['non_event'] = 1 - agg_df['mean']
            agg_df['woe'] = np.log((agg_df['mean'] + epsilon) / (agg_df['non_event'] + epsilon))
            self.woe_dict[col] = agg_df['woe'].to_dict()
        return self

    def transform(self, X):
        X_trans = X.copy()
        for col, mapping in self.woe_dict.items():
            X_trans[col] = X_trans[col].map(mapping).fillna(0)
        return X_trans

# === Outlier Treatment ===
class OutlierCapper(BaseEstimator, TransformerMixin):
    """Caps outliers using IQR method with dynamic threshold"""
    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_bounds = {}
        self.upper_bounds = {}
        
    def fit(self, X, y=None):
        for col in X.select_dtypes(include=np.number).columns:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            self.lower_bounds[col] = q1 - self.factor * iqr
            self.upper_bounds[col] = q3 + self.factor * iqr
        return self
        
    def transform(self, X):
        X_out = X.copy()
        for col in X_out.select_dtypes(include=np.number).columns:
            if col in self.lower_bounds:
                lb = self.lower_bounds[col]
                ub = self.upper_bounds[col]
                X_out[col] = X_out[col].clip(lower=lb, upper=ub)
        return X_out

# === Proxy Risk Label Creator ===
class RiskLabelGenerator(BaseEstimator, TransformerMixin):
    """Creates proxy risk labels using RFM clustering"""
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.scaler = RobustScaler()
        
    def fit(self, X, y=None):
        # Use Five Cs features for clustering
        cluster_features = ['recency', 'txn_variability', 'net_cash_flow', 'spending_slope']
        self.cluster_scaled_ = self.scaler.fit_transform(X[cluster_features])
        
        # Cluster customers
        self.kmeans.fit(self.cluster_scaled_)
        
        # Identify high-risk cluster (business rules)
        cluster_profiles = X[cluster_features].groupby(self.kmeans.labels_).mean()
        cluster_profiles['risk_score'] = (
            cluster_profiles['recency'] * 0.4 + 
            cluster_profiles['txn_variability'] * 0.3 +
            cluster_profiles['net_cash_flow'] * (-0.2) +
            cluster_profiles['spending_slope'] * (-0.1)
        )
        self.high_risk_cluster_ = cluster_profiles['risk_score'].idxmax()
        
        return self
        
    def transform(self, X):
        # Add cluster labels
        cluster_features = ['recency', 'txn_variability', 'net_cash_flow', 'spending_slope']
        X_scaled = self.scaler.transform(X[cluster_features])
        X['risk_cluster'] = self.kmeans.predict(X_scaled)
        
        # Create binary risk label
        X['is_high_risk'] = (X['risk_cluster'] == self.high_risk_cluster_).astype(int)
        X.drop(columns=['risk_cluster'], inplace=True)
        
        logging.info(f"High-risk customers: {X['is_high_risk'].mean():.2%}")
        return X

# === Data Processing Pipeline ===
def process_data(input_csv, output_csv):
    """End-to-end data processing pipeline"""
    try:
        # 1. Load data
        logging.info(f"Loading data from {input_csv}")
        df = pd.read_csv(input_csv, parse_dates=['TransactionStartTime'])
        
        # 2. Feature engineering with Five Cs framework
        logging.info("Engineering features with Five Cs framework")
        feature_engineer = FiveCsFeatureEngineer()
        customer_features = feature_engineer.fit_transform(df)
        
        # 3. Create proxy risk labels
        logging.info("Creating proxy risk labels")
        label_generator = RiskLabelGenerator()
        customer_features = label_generator.fit_transform(customer_features)
        
        # 4. Handle missing values with iterative imputation
        logging.info("Handling missing values")
        num_cols = customer_features.select_dtypes(include=np.number).columns.tolist()
        num_cols.remove('is_high_risk') if 'is_high_risk' in num_cols else None
        
        imputer = IterativeImputer(random_state=42, max_iter=10)
        customer_features_imputed = customer_features.copy()
        customer_features_imputed[num_cols] = imputer.fit_transform(customer_features_imputed[num_cols])
        
        # 5. Outlier treatment
        capper = OutlierCapper(factor=1.5)
        customer_features_capped = capper.fit_transform(customer_features_imputed)
        
        # 6. WoE encoding for categorical features
        # Note: Requires target variable - only for model training data
        woe_encoder = WoETransformer()
        if 'ProductCategory' in customer_features_capped.columns:
            customer_features_encoded = woe_encoder.fit_transform(
                customer_features_capped, 
                customer_features_capped['is_high_risk']
            )
        else:
            customer_features_encoded = customer_features_capped
            
        # 7. Save processed data
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        customer_features_encoded.to_csv(output_csv, index=False)
        logging.info(f"Processed data saved to {output_csv}")
        logging.info(f"Final shape: {customer_features_encoded.shape}")
        
        return customer_features_encoded
        
    except Exception as e:
        logging.error(f"Data processing failed: {str(e)}")
        raise

# === For CLI usage ===
if __name__ == '__main__':
    raw_csv_path = '../data/raw/data.csv'
    processed_csv_path = 'data/processed/customer_features.csv'
    
    # Execute processing pipeline
    processed_data = process_data(raw_csv_path, processed_csv_path)
    
    # Print sample output
    print("\nSample processed data:")
    print(processed_data[['CustomerId', 'recency', 'payment_consistency', 'spending_slope', 'is_high_risk']].head())