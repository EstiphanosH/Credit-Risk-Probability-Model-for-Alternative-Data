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
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, FunctionTransformer, RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === Basel III Compliance Metrics ===
def calculate_regulatory_capital(PD, LGD=0.45, EAD=1000):
    """
    Calculate Basel III regulatory capital requirements
    
    Args:
        PD: Probability of Default
        LGD: Loss Given Default (default 0.45)
        EAD: Exposure at Default (default 1000)
    
    Returns:
        Regulatory capital requirement
    """
    R = 0.12 * (1 - np.exp(-50 * PD)) / (1 - np.exp(-50)) + \
         0.24 * (1 - (1 - np.exp(-50 * PD)) / (1 - np.exp(-50)))
    K = LGD * stats.norm.cdf((1 - R)**-0.5 * stats.norm.ppf(PD) + 
                          (R / (1 - R))**0.5 * stats.norm.ppf(0.999)) - PD * LGD
    return K * EAD

# === Feature Engineering Aligned with Five Cs ===
class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts temporal features from transaction timestamps"""
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        X['TransactionHour'] = X['TransactionStartTime'].dt.hour
        X['TransactionDay'] = X['TransactionStartTime'].dt.day
        X['TransactionDayOfWeek'] = X['TransactionStartTime'].dt.dayofweek
        X['TransactionMonth'] = X['TransactionStartTime'].dt.month
        X['TransactionYear'] = X['TransactionStartTime'].dt.year
        return X

class CustomerAggregator(BaseEstimator, TransformerMixin):
    """
    Creates customer-level aggregated features aligned with Five Cs framework:
    Character, Capacity, Capital, Collateral, Conditions
    """
    def __init__(self, snapshot_date=None):
        self.snapshot_date = snapshot_date or pd.Timestamp.now()
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # Define aggregations using named aggregation
        agg_features = X.groupby('CustomerId').agg(
            lifetime_value=('Amount', lambda x: x[x > 0].sum()),
            avg_amount=('Amount', lambda x: x[x > 0].mean()),
            max_amount=('Amount', 'max'),
            min_amount=('Amount', 'min'),
            amount_std=('Amount', lambda x: x.std() if len(x) > 1 else 0),
            transaction_count=('Amount', 'count'),
            positive_transaction_ratio=('Amount', lambda x: (x > 0).mean()),
            fraud_count=('FraudResult', 'sum'),
            fraud_rate=('FraudResult', 'mean'),
            top_category=('ProductCategory', lambda x: x.mode()[0] if not x.mode().empty else 'other'),
            category_diversity=('ProductCategory', 'nunique'),
            peak_hour=('TransactionHour', lambda x: x.mode()[0] if not x.mode().empty else -1)
        )
        
        # Calculate time-based features
        time_diff = X.groupby('CustomerId')['TransactionStartTime'].agg(
            time_diff=lambda x: (x.max() - x.min()).days if len(x) > 1 else 0
        )
        agg_features['txn_velocity'] = agg_features['transaction_count'] / (time_diff + 1)
        
        # Calculate recency (days since last transaction relative to snapshot date)
        last_transaction = X.groupby('CustomerId')['TransactionStartTime'].max()
        agg_features['recency'] = (self.snapshot_date - last_transaction).dt.days
        
        # Calculate holiday spending ratio
        holiday_months = [11, 12]  # Nov and Dec
        holiday_spend = X[X['TransactionStartTime'].dt.month.isin(holiday_months)].groupby('CustomerId')['Amount'].sum()
        non_holiday_spend = X[~X['TransactionStartTime'].dt.month.isin(holiday_months)].groupby('CustomerId')['Amount'].sum()
        agg_features['holiday_spending_ratio'] = holiday_spend / (non_holiday_spend + 1e-6)
        
        # Calculate spending slope (linear trend)
        def spending_slope(group):
            if len(group) < 3:
                return 0
            x = np.arange(len(group))
            y = group['Amount'].values
            return np.polyfit(x, y, 1)[0]
        
        spending_slopes = X[X['Amount'] > 0].sort_values('TransactionStartTime').groupby('CustomerId').apply(spending_slope)
        agg_features['spending_slope'] = spending_slopes
        
        # Calculate payment consistency
        agg_features['payment_consistency'] = X.groupby('CustomerId')['Amount'].apply(
            lambda x: (x < 0).mean() if len(x) > 0 else 0
        )
        
        # Add Basel III metrics
        agg_features['expected_loss'] = calculate_regulatory_capital(
            0.05, 0.45, agg_features['lifetime_value'] * 0.1
        )
        
        return agg_features.reset_index()

# === Feature Engineering Pipeline ===
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
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, FunctionTransformer, RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === Basel III Compliance Metrics ===
def calculate_regulatory_capital(PD, LGD=0.45, EAD=1000):
    """
    Calculate Basel III regulatory capital requirements
    
    Args:
        PD: Probability of Default
        LGD: Loss Given Default (default 0.45)
        EAD: Exposure at Default (default 1000)
    
    Returns:
        Regulatory capital requirement
    """
    R = 0.12 * (1 - np.exp(-50 * PD)) / (1 - np.exp(-50)) + \
         0.24 * (1 - (1 - np.exp(-50 * PD)) / (1 - np.exp(-50)))
    K = LGD * stats.norm.cdf((1 - R)**-0.5 * stats.norm.ppf(PD) + 
                          (R / (1 - R))**0.5 * stats.norm.ppf(0.999)) - PD * LGD
    return K * EAD

# === Feature Engineering Aligned with Five Cs ===
class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts temporal features from transaction timestamps"""
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        X['TransactionHour'] = X['TransactionStartTime'].dt.hour
        X['TransactionDay'] = X['TransactionStartTime'].dt.day
        X['TransactionDayOfWeek'] = X['TransactionStartTime'].dt.dayofweek
        X['TransactionMonth'] = X['TransactionStartTime'].dt.month
        X['TransactionYear'] = X['TransactionStartTime'].dt.year
        return X

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
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, FunctionTransformer, RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === Basel III Compliance Metrics ===
def calculate_regulatory_capital(PD, LGD=0.45, EAD=1000):
    """
    Calculate Basel III regulatory capital requirements
    
    Args:
        PD: Probability of Default
        LGD: Loss Given Default (default 0.45)
        EAD: Exposure at Default (default 1000)
    
    Returns:
        Regulatory capital requirement
    """
    R = 0.12 * (1 - np.exp(-50 * PD)) / (1 - np.exp(-50)) + \
         0.24 * (1 - (1 - np.exp(-50 * PD)) / (1 - np.exp(-50)))
    K = LGD * stats.norm.cdf((1 - R)**-0.5 * stats.norm.ppf(PD) + 
                          (R / (1 - R))**0.5 * stats.norm.ppf(0.999)) - PD * LGD
    return K * EAD

# === Feature Engineering Aligned with Five Cs ===
class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts temporal features from transaction timestamps"""
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        X['TransactionHour'] = X['TransactionStartTime'].dt.hour
        X['TransactionDay'] = X['TransactionStartTime'].dt.day
        X['TransactionDayOfWeek'] = X['TransactionStartTime'].dt.dayofweek
        X['TransactionMonth'] = X['TransactionStartTime'].dt.month
        X['TransactionYear'] = X['TransactionStartTime'].dt.year
        return X

class CustomerAggregator(BaseEstimator, TransformerMixin):
    """
    Creates customer-level aggregated features aligned with Five Cs framework:
    Character, Capacity, Capital, Collateral, Conditions
    """
    def __init__(self, snapshot_date=None):
        self.snapshot_date = snapshot_date or pd.Timestamp.now()
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # Define aggregations using named aggregation
        agg_features = X.groupby('CustomerId').agg(
            lifetime_value=('Amount', lambda x: x[x > 0].sum()),
            avg_amount=('Amount', lambda x: x[x > 0].mean()),
            max_amount=('Amount', 'max'),
            min_amount=('Amount', 'min'),
            amount_std=('Amount', lambda x: x.std() if len(x) > 1 else 0),
            transaction_count=('Amount', 'count'),
            positive_transaction_ratio=('Amount', lambda x: (x > 0).mean()),
            fraud_count=('FraudResult', 'sum'),
            fraud_rate=('FraudResult', 'mean'),
            top_category=('ProductCategory', lambda x: x.mode()[0] if not x.mode().empty else 'other'),
            category_diversity=('ProductCategory', 'nunique'),
            peak_hour=('TransactionHour', lambda x: x.mode()[0] if not x.mode().empty else -1)
        )
        
        # Calculate time-based features - FIXED VERSION
        time_diff = X.groupby('CustomerId')['TransactionStartTime'].apply(
            lambda x: (x.max() - x.min()).days if len(x) > 1 else 0
        ).rename('time_diff')
        
        # Now safely calculate txn_velocity
        agg_features = agg_features.merge(time_diff, left_index=True, right_index=True)
        agg_features['txn_velocity'] = agg_features['transaction_count'] / (agg_features['time_diff'] + 1)
        
        # Calculate recency (days since last transaction relative to snapshot date)
        last_transaction = X.groupby('CustomerId')['TransactionStartTime'].max()
        agg_features['recency'] = (self.snapshot_date - last_transaction).dt.days
        
        # Calculate holiday spending ratio
        holiday_months = [11, 12]  # Nov and Dec
        holiday_spend = X[X['TransactionStartTime'].dt.month.isin(holiday_months)].groupby('CustomerId')['Amount'].sum()
        non_holiday_spend = X[~X['TransactionStartTime'].dt.month.isin(holiday_months)].groupby('CustomerId')['Amount'].sum()
        agg_features['holiday_spending_ratio'] = holiday_spend / (non_holiday_spend + 1e-6)
        
        # Calculate spending slope (linear trend)
        def spending_slope(group):
            if len(group) < 3:
                return 0
            x = np.arange(len(group))
            y = group['Amount'].values
            return np.polyfit(x, y, 1)[0]
        
        spending_slopes = X[X['Amount'] > 0].sort_values('TransactionStartTime').groupby('CustomerId').apply(spending_slope)
        agg_features['spending_slope'] = spending_slopes
        
        # Calculate payment consistency
        payment_consistency = X.groupby('CustomerId')['Amount'].apply(
            lambda x: (x < 0).mean() if len(x) > 0 else 0
        )
        agg_features['payment_consistency'] = payment_consistency
        
        # Add Basel III metrics
        agg_features['expected_loss'] = calculate_regulatory_capital(
            0.05, 0.45, agg_features['lifetime_value'] * 0.1
        )
        
        # Clean up temporary column
        agg_features = agg_features.drop(columns=['time_diff'])
        
        return agg_features.reset_index()


# === Feature Engineering Pipeline ===
class FiveCsFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Orchestrates the feature engineering process using Five Cs framework:
    Character, Capacity, Capital, Collateral, Conditions
    
    Args:
        snapshot_date: Reference date for recency calculations
    """
    def __init__(self, snapshot_date=None):
        self.snapshot_date = snapshot_date
        self.temporal_extractor = TemporalFeatureExtractor()
        self.customer_aggregator = CustomerAggregator(snapshot_date=snapshot_date)
        
    def fit(self, X, y=None):
        self.temporal_extractor.fit(X)
        return self
        
    def transform(self, X):
        X_temp = self.temporal_extractor.transform(X)
        return self.customer_aggregator.transform(X_temp)

# === Weight of Evidence (WoE) Encoder ===
class WoETransformer(BaseEstimator, TransformerMixin):
    """
    Weight of Evidence encoder with fallback to simple implementation
    
    Args:
        target: Target variable name
        min_samples: Minimum samples required for binning
        n_bins: Number of bins for numerical features
    """
    def __init__(self, target='is_high_risk', min_samples=100, n_bins=5):
        self.target = target
        self.min_samples = min_samples
        self.n_bins = n_bins
        self.woe_dict = {}
        self.use_xverse = True
        self.xverse_available = True
        
        # Check if xverse is available
        try:
            from xverse.transformer import MonotonicBinning, WOE
        except ImportError:
            self.xverse_available = False

    def fit(self, X, y):
        if not self.xverse_available:
            return self._fit_simple(X, y)
            
        # Try using xverse if available
        try:
            from xverse.transformer import MonotonicBinning, WOE
            for col in X.select_dtypes(include=['number']).columns:
                try:
                    # Monotonic binning for numerical features
                    binner = MonotonicBinning(n_bins=self.n_bins)
                    binned_col = binner.fit_transform(X[[col]], y)
                    
                    # Calculate WoE for bins
                    woe_calc = WOE()
                    woe_calc.fit(binned_col, y)
                    self.woe_dict[col] = woe_calc.woe_df
                except Exception:
                    # Fallback to simple WoE if binning fails
                    self._fit_simple_for_col(X, y, col)
            
            # For categorical features
            for col in X.select_dtypes(include=['object', 'category']).columns:
                woe_calc = WOE()
                woe_calc.fit(X[[col]], y)
                self.woe_dict[col] = woe_calc.woe_df
                
        except Exception as e:
            logging.warning(f"xverse failed, using simple WoE: {str(e)}")
            self.use_xverse = False
            self._fit_simple(X, y)
            
        return self

    def _fit_simple(self, X, y):
        """Simple WoE implementation without xverse"""
        epsilon = 1e-6  # Small value to avoid log(0)
        for col in X.columns:
            self._fit_simple_for_col(X, y, col, epsilon)
        return self

    def _fit_simple_for_col(self, X, y, col, epsilon=1e-6):
        """Simple WoE implementation for a single column"""
        # Group by column values
        agg_df = X[[col]].copy()
        agg_df[self.target] = y
        group = agg_df.groupby(col)[self.target].agg(['count', 'mean'])
        group = group[group['count'] > self.min_samples]
        
        if len(group) == 0:
            self.woe_dict[col] = {np.nan: 0}
            return
            
        # Calculate WoE
        group['non_event'] = 1 - group['mean']
        group['woe'] = np.log((group['mean'] + epsilon) / (group['non_event'] + epsilon))
        self.woe_dict[col] = group['woe'].to_dict()

    def transform(self, X):
        X_trans = X.copy()
        for col in X_trans.columns:
            if col in self.woe_dict:
                mapping = self.woe_dict[col]
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
    """
    Creates proxy risk labels using RFM clustering
    
    Args:
        n_clusters: Number of clusters for segmentation
        random_state: Random seed for reproducibility
    """
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.scaler = RobustScaler()
        
    def fit(self, X, y=None):
        # Use Five Cs features for clustering
        cluster_features = ['recency', 'amount_std', 'lifetime_value', 'spending_slope']
        self.cluster_scaled_ = self.scaler.fit_transform(X[cluster_features].fillna(0))
        
        # Cluster customers
        self.kmeans.fit(self.cluster_scaled_)
        
        # Identify high-risk cluster (business rules)
        cluster_profiles = X[cluster_features].fillna(0).groupby(self.kmeans.labels_).mean()
        cluster_profiles['risk_score'] = (
            cluster_profiles['recency'] * 0.4 + 
            cluster_profiles['amount_std'] * 0.3 +
            cluster_profiles['lifetime_value'] * (-0.2) +
            cluster_profiles['spending_slope'] * (-0.1)
        )
        self.high_risk_cluster_ = cluster_profiles['risk_score'].idxmax()
        
        return self
        
    def transform(self, X):
        # Add cluster labels
        cluster_features = ['recency', 'amount_std', 'lifetime_value', 'spending_slope']
        X_scaled = self.scaler.transform(X[cluster_features].fillna(0))
        X['risk_cluster'] = self.kmeans.predict(X_scaled)
        
        # Create binary risk label
        X['is_high_risk'] = (X['risk_cluster'] == self.high_risk_cluster_).astype(int)
        X.drop(columns=['risk_cluster'], inplace=True)
        
        logging.info(f"High-risk customers: {X['is_high_risk'].mean():.2%}")
        return X

# === Data Processing Pipeline ===
def process_data(input_csv, output_csv):
    """
    End-to-end data processing pipeline
    
    Args:
        input_csv: Path to raw input CSV file
        output_csv: Path for processed output CSV file
    
    Returns:
        Processed training and test datasets
    """
    try:
        # 1. Load data
        logging.info(f"Loading data from {input_csv}")
        df = pd.read_csv(input_csv, parse_dates=['TransactionStartTime'])
        
        # 2. Feature engineering with Five Cs framework
        logging.info("Engineering features with Five Cs framework")
        snapshot_date = df['TransactionStartTime'].max() + timedelta(days=30)
        feature_engineer = FiveCsFeatureEngineer(snapshot_date=snapshot_date)
        customer_features = feature_engineer.fit_transform(df)
        
        # 3. Create proxy risk labels
        logging.info("Creating proxy risk labels")
        label_generator = RiskLabelGenerator()
        customer_features = label_generator.fit_transform(customer_features)
        
        # 4. Split data for WoE encoding to prevent leakage
        X = customer_features.drop(columns=['is_high_risk', 'CustomerId'], errors='ignore')
        y = customer_features['is_high_risk']
        
        # Split before any target-dependent transformations
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 5. Handle missing values
        logging.info("Handling missing values")
        num_imputer = IterativeImputer(random_state=42, max_iter=10)
        cat_imputer = SimpleImputer(strategy='most_frequent')
        
        # Identify columns
        num_cols = X.select_dtypes(include=np.number).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Impute missing values
        X_train_imputed = X_train.copy()
        X_test_imputed = X_test.copy()
        
        if num_cols:
            X_train_imputed[num_cols] = num_imputer.fit_transform(X_train[num_cols])
            X_test_imputed[num_cols] = num_imputer.transform(X_test[num_cols])
            
        if cat_cols:
            X_train_imputed[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
            X_test_imputed[cat_cols] = cat_imputer.transform(X_test[cat_cols])
        
        # 6. Outlier treatment
        logging.info("Capping outliers")
        capper = OutlierCapper(factor=1.5)
        X_train_capped = capper.fit_transform(X_train_imputed)
        X_test_capped = capper.transform(X_test_imputed)
        
        # 7. WoE encoding for categorical features
        logging.info("Applying WoE encoding")
        woe_encoder = WoETransformer()
        X_train_woe = woe_encoder.fit_transform(X_train_capped, y_train)
        X_test_woe = woe_encoder.transform(X_test_capped)
        
        # 8. Normalize numerical features
        logging.info("Normalizing numerical features")
        scaler = StandardScaler()
        X_train_scaled = X_train_woe.copy()
        X_test_scaled = X_test_woe.copy()
        
        if num_cols:
            X_train_scaled[num_cols] = scaler.fit_transform(X_train_woe[num_cols])
            X_test_scaled[num_cols] = scaler.transform(X_test_woe[num_cols])
        
        # 9. One-hot encode remaining categorical features
        logging.info("One-hot encoding categorical features")
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        if cat_cols:
            # One-hot encode non-WoE categorical features
            non_woed_cat_cols = [col for col in cat_cols if col in X_train_scaled.columns]
            
            if non_woed_cat_cols:
                train_ohe = ohe.fit_transform(X_train_scaled[non_woed_cat_cols])
                test_ohe = ohe.transform(X_test_scaled[non_woed_cat_cols])
                
                # Create feature names
                ohe_cols = ohe.get_feature_names_out(non_woed_cat_cols)
                
                # Create DataFrames
                train_ohe_df = pd.DataFrame(train_ohe, columns=ohe_cols, index=X_train_scaled.index)
                test_ohe_df = pd.DataFrame(test_ohe, columns=ohe_cols, index=X_test_scaled.index)
                
                # Combine with main data
                X_train_final = pd.concat([
                    X_train_scaled.drop(columns=non_woed_cat_cols), 
                    train_ohe_df
                ], axis=1)
                
                X_test_final = pd.concat([
                    X_test_scaled.drop(columns=non_woed_cat_cols), 
                    test_ohe_df
                ], axis=1)
            else:
                X_train_final = X_train_scaled
                X_test_final = X_test_scaled
        else:
            X_train_final = X_train_scaled
            X_test_final = X_test_scaled
        
        # Add identifiers and target
        X_train_final['CustomerId'] = customer_features.loc[X_train_final.index, 'CustomerId'].values
        X_test_final['CustomerId'] = customer_features.loc[X_test_final.index, 'CustomerId'].values
        X_train_final['is_high_risk'] = y_train.values
        X_test_final['is_high_risk'] = y_test.values
        
        # 10. Save processed data
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        train_output = output_csv.replace('.csv', '_train.csv')
        test_output = output_csv.replace('.csv', '_test.csv')
        
        X_train_final.to_csv(train_output, index=False)
        X_test_final.to_csv(test_output, index=False)
        
        logging.info(f"Processed data saved to {train_output} and {test_output}")
        logging.info(f"Train shape: {X_train_final.shape}, Test shape: {X_test_final.shape}")
        
        return X_train_final, X_test_final
        
    except Exception as e:
        logging.error(f"Data processing failed: {str(e)}")
        raise

# === For CLI usage ===
if __name__ == '__main__':
    raw_csv_path = '../data/raw/data.csv'
    processed_csv_path = '../data/processed/customer_features.csv'
    
    # Execute processing pipeline
    train_data, test_data = process_data(raw_csv_path, processed_csv_path)
    
    # Print sample output
    print("\nSample processed data:")
    print(train_data[['CustomerId', 'recency', 'payment_consistency', 'spending_slope', 'is_high_risk']].head())