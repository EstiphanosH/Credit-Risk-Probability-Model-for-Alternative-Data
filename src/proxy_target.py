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
import argparse
import os
import sys
import matplotlib.pyplot as plt

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
    
def main():
    # Set up command-line interface
    parser = argparse.ArgumentParser(
        description='Generate RFM-based Credit Risk Labels',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input', help='Path to transaction data CSV file')
    parser.add_argument('--output', default='risk_labels.csv',
                        help='Output filename for risk labels')
    parser.add_argument('--clusters', type=int, default=3,
                        help='Number of clusters for K-Means')
    parser.add_argument('--recency_weight', type=float, default=0.5,
                        help='Weight for recency in risk score')
    parser.add_argument('--frequency_weight', type=float, default=0.3,
                        help='Weight for frequency in risk score')
    parser.add_argument('--monetary_weight', type=float, default=0.2,
                        help='Weight for monetary in risk score')
    parser.add_argument('--optimal_clusters', action='store_true',
                        help='Find optimal cluster count using elbow method')
    parser.add_argument('--max_k', type=int, default=10,
                        help='Max clusters for elbow method')
    parser.add_argument('--report', help='Generate cluster report to specified file')
    parser.add_argument('--plot_elbow', action='store_true',
                        help='Plot elbow curve when finding optimal clusters')
    parser.add_argument('--random_state', type=int, 
                        help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable detailed logging')
    
    args = parser.parse_args()

    # Configure logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("RiskLabelGeneratorCLI")

    try:
        # Load transaction data
        logger.info(f"Loading data from {args.input}")
        transactions = pd.read_csv(args.input, parse_dates=['TransactionStartTime'])
        
        # Initialize label generator
        labeler = RiskLabelGenerator(
            n_clusters=args.clusters,
            recency_weight=args.recency_weight,
            frequency_weight=args.frequency_weight,
            monetary_weight=args.monetary_weight,
            random_state=args.random_state
        )

        # Handle optimal cluster analysis
        if args.optimal_clusters:
            logger.info(f"Finding optimal clusters (max_k={args.max_k})")
            distortions = labeler.find_optimal_clusters(transactions, max_k=args.max_k)
            
            print("\nElbow Method Results:")
            print(f"{'Clusters':<10} {'Distortion':<15}")
            for k, dist in distortions.items():
                print(f"{k:<10} {dist:.6f}")
            
            if args.plot_elbow:
                plt.figure(figsize=(10, 6))
                plt.plot(list(distortions.keys()), list(distortions.values()), 'bx-')
                plt.xlabel('Number of clusters')
                plt.ylabel('Distortion')
                plt.title('Elbow Method for Optimal K')
                plt.savefig('elbow_plot.png', bbox_inches='tight')
                logger.info("Saved elbow plot to elbow_plot.png")
            return

        # Generate risk labels
        logger.info("Generating risk labels")
        labels = labeler.fit_transform(transactions)
        labels.to_csv(args.output, index=False)
        logger.info(f"Saved risk labels to {args.output}")
        
        # Calculate risk distribution
        high_risk_count = labels['is_high_risk'].sum()
        total_customers = len(labels)
        risk_percentage = (high_risk_count / total_customers) * 100
        
        print("\nRisk Label Summary:")
        print(f"Total customers: {total_customers}")
        print(f"High-risk customers: {high_risk_count} ({risk_percentage:.1f}%)")
        print(f"Output saved to: {os.path.abspath(args.output)}")

        # Generate cluster report if requested
        if args.report:
            logger.info("Generating cluster report")
            report = labeler.generate_cluster_report(transactions)
            report.to_csv(args.report, index=False)
            logger.info(f"Saved cluster report to {args.report}")
            
            print("\nCluster Report Summary:")
            print(report[['cluster', 'customer_count', 'customer_pct', 'is_high_risk']])
            print(f"Full report saved to: {os.path.abspath(args.report)}")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()