"""
RFM-based Credit Risk Label Generator
-------------------------------------
Creates binary "is_high_risk" labels using K-Means clustering on RFM features
(Recency, Frequency, Monetary) to identify high-risk customers.

Key Features:
1. RobustScaler for outlier-resistant feature scaling
2. Elbow method for optimal cluster determination
3. Business-rule based risk identification
4. Comprehensive input validation
5. Detailed logging for audit trails
6. Cluster visualization utilities

Usage:
1. Initialize: labeler = RiskLabelGenerator()
2. Fit: labeler.fit(transactions_df)
3. Transform: risk_labels = labeler.transform(transactions_df)
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import logging
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RiskLabelGenerator")

class RiskLabelGenerator:
    """
    Creates proxy credit risk labels using RFM-based clustering
    
    Parameters:
    n_clusters (int): Number of clusters (default=3)
    random_state (int): Random seed (default=42)
    recency_weight (float): Weight for recency in risk score (default=0.4)
    frequency_weight (float): Weight for frequency in risk score (default=0.3)
    monetary_weight (float): Weight for monetary in risk score (default=0.3)
    """
    
    def __init__(self, n_clusters=3, recency_weight=0.5, frequency_weight=0.3, monetary_weight=0.2, random_state=None):
        self.n_clusters = n_clusters
        self.recency_weight = recency_weight
        self.frequency_weight = frequency_weight
        self.monetary_weight = monetary_weight
        self.random_state = random_state
        self.weights = {
            'recency': recency_weight,
            'frequency': frequency_weight,
            'monetary': monetary_weight
        }
        self.scaler = RobustScaler()
        self.kmeans = None
        self.high_risk_cluster_ = None
        self.snapshot_date = None
        self.feature_columns = ['recency', 'frequency', 'monetary']
        self._validate_parameters()

    def _validate_parameters(self):
        """Validate input parameters"""
        if not isinstance(self.n_clusters, int) or self.n_clusters < 2:
            raise ValueError("n_clusters must be integer >= 2")
        if not 0 <= self.weights['recency'] <= 1:
            raise ValueError("recency_weight must be between 0-1")
        if not 0 <= self.weights['frequency'] <= 1:
            raise ValueError("frequency_weight must be between 0-1")
        if not 0 <= self.weights['monetary'] <= 1:
            raise ValueError("monetary_weight must be between 0-1")
        total_weight = sum(self.weights.values())
        if not 0.99 <= total_weight <= 1.01:
            raise ValueError("Weights must sum to 1.0")

    def fit(self, transactions_df):
        """
        Fit the clustering model and identify high-risk cluster

        Parameters:
        transactions_df (pd.DataFrame): Transaction data with columns:
            - CustomerId (str or int)
            - TransactionStartTime (datetime)
            - Amount (float)
        """
        logger.info("Fitting risk label generator")
        self._validate_input_data(transactions_df)

        # Calculate RFM features
        rfm = self._calculate_rfm(transactions_df)

        # Scale features
        rfm_scaled = self.scaler.fit_transform(rfm[self.feature_columns])

        # Fit K-Means
        self.kmeans = KMeans(n_clusters=self.n_clusters,
                             random_state=self.random_state,
                             n_init=10)
        self.kmeans.fit(rfm_scaled)

        # Identify high-risk cluster
        self._identify_high_risk_cluster(rfm)
        return self
    
    def transform(self, transactions_df):
        """
        Assign risk labels to customers
        
        Parameters:
        transactions_df (pd.DataFrame): Transaction data
        
        Returns:
        pd.DataFrame: CustomerId and is_high_risk label
        """
        logger.info("Transforming data to risk labels")
        self._validate_input_data(transactions_df)
        
        if not self.kmeans:
            raise RuntimeError("Model not fitted. Call fit() first")
            
        # Calculate RFM features
        rfm = self._calculate_rfm(transactions_df)
        
        # Scale features
        rfm_scaled = self.scaler.transform(rfm[self.feature_columns])
        
        # Predict clusters
        clusters = self.kmeans.predict(rfm_scaled)
        rfm['is_high_risk'] = (clusters == self.high_risk_cluster_).astype(int)
        
        # Return risk labels
        return rfm[['CustomerId', 'is_high_risk']]
    
    def fit_transform(self, transactions_df):
        """Fit and transform in one step"""
        self.fit(transactions_df)
        return self.transform(transactions_df)
    
    def _validate_input_data(self, df):
        """Validate input data structure"""
        required_columns = {'CustomerId', 'TransactionStartTime', 'Amount'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")
            
        if df['TransactionStartTime'].dtype != 'datetime64[ns]':
            try:
                df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
            except Exception as e:
                raise ValueError("TransactionStartTime must be convertible to datetime") from e
                
        if df['CustomerId'].nunique() < 10:
            logger.warning(f"Only {df['CustomerId'].nunique()} customers detected")
    
    def _calculate_rfm(self, transactions_df):
        """Calculate RFM metrics for each customer"""
        # Set snapshot date (max transaction date + 1 day)
        if self.snapshot_date is None:
            self.snapshot_date = transactions_df['TransactionStartTime'].max() + timedelta(days=1)
            logger.info(f"Snapshot date set to: {self.snapshot_date}")
        
        # RFM Calculation
        rfm = transactions_df.groupby('CustomerId').agg(
            last_txn_date=('TransactionStartTime', 'max'),
            frequency=('TransactionStartTime', 'count'),
            monetary=('Amount', lambda x: x[x > 0].sum())  # Only positive amounts
        ).reset_index()
        
        rfm['recency'] = (self.snapshot_date - rfm['last_txn_date']).dt.days
        return rfm.drop(columns=['last_txn_date'])
    
    def find_optimal_clusters(self, transactions_df, max_k=10):
        """
        Determine optimal cluster count using elbow method
        
        Returns:
        dict: Distortion scores for each k
        """
        rfm = self._calculate_rfm(transactions_df)
        rfm_scaled = self.scaler.fit_transform(rfm[self.feature_columns])
        
        distortions = {}
        for k in range(2, max_k+1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(rfm_scaled)
            distortions[k] = sum(np.min(cdist(rfm_scaled, kmeans.cluster_centers_, 'euclidean'), 
                                 axis=1)) / rfm_scaled.shape[0]
        
        return distortions
    
    def _identify_high_risk_cluster(self, rfm):
        """Identify high-risk cluster based on business rules"""
        # Get cluster centers in original scale
        cluster_centers = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        
        # Create risk score for each cluster
        risk_scores = (
            cluster_centers[:, 0] * self.weights['recency'] -  # Higher recency = riskier
            cluster_centers[:, 1] * self.weights['frequency'] -  # Higher frequency = less risky
            cluster_centers[:, 2] * self.weights['monetary']  # Higher monetary = less risky
        )
        
        # Identify highest risk cluster
        self.high_risk_cluster_ = np.argmax(risk_scores)
        
        # Log cluster profiles
        cluster_profiles = pd.DataFrame(
            cluster_centers,
            columns=self.feature_columns
        )
        cluster_profiles['risk_score'] = risk_scores
        cluster_profiles['is_high_risk'] = [i == self.high_risk_cluster_ for i in range(len(cluster_profiles))]
        
        logger.info("Cluster Profiles:\n" + cluster_profiles.to_string())
        logger.info(f"Identified Cluster {self.high_risk_cluster_} as high-risk")
    
    def generate_cluster_report(self, transactions_df):
        """Generate cluster characteristics report"""
        if not self.kmeans:
            raise RuntimeError("Model not fitted. Call fit() first")
            
        rfm = self._calculate_rfm(transactions_df)
        rfm_scaled = self.scaler.transform(rfm[self.feature_columns])
        rfm['cluster'] = self.kmeans.predict(rfm_scaled)
        
        report = rfm.groupby('cluster').agg(
            customer_count=('CustomerId', 'count'),
            recency_mean=('recency', 'mean'),
            frequency_mean=('frequency', 'mean'),
            monetary_mean=('monetary', 'mean'),
            recency_std=('recency', 'std'),
            frequency_std=('frequency', 'std'),
            monetary_std=('monetary', 'std'),
            is_high_risk=('cluster', lambda x: (x == self.high_risk_cluster_).mean())
        ).reset_index()
        
        report['customer_pct'] = report['customer_count'] / report['customer_count'].sum()
        return report