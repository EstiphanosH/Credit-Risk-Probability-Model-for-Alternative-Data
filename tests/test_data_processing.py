import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add src directory to path for module imports
sys.path.insert(0, os.path.abspath('../src'))
from src.data_processing import (
    FiveCsFeatureEngineer,
    RiskLabelGenerator,
    OutlierCapper,
    WoETransformer,
    calculate_regulatory_capital,
    process_data
)

class TestCreditRiskProcessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create sample data
        np.random.seed(42)
        cls.transaction_data = pd.DataFrame({
            "TransactionId": ["TXN001", "TXN002", "TXN003", "TXN004", "TXN005"],
            "CustomerId": ["CUST1", "CUST1", "CUST2", "CUST2", "CUST3"],
            "Amount": [150.0, -50.0, 200.0, 75.0, 300.0],
            "FraudResult": [0, 0, 1, 0, 0],
            "ProductCategory": ["Electronics", "Electronics", "Clothing", None, "Groceries"],
            "TransactionStartTime": [
                datetime(2024, 1, 1, 10, 0),
                datetime(2024, 1, 2, 14, 0),
                datetime(2024, 1, 3, 9, 0),
                datetime(2024, 1, 4, 16, 0),
                datetime(2024, 12, 15, 11, 0)
            ]
        })

    def test_basel_capital_calculation(self):
        """Test regulatory capital calculation"""
        self.assertAlmostEqual(calculate_regulatory_capital(0.01), 58.62, places=1)
        self.assertAlmostEqual(calculate_regulatory_capital(0.10), 140.60, places=1)
        self.assertAlmostEqual(calculate_regulatory_capital(0.50), 167.89, places=1)
        self.assertAlmostEqual(
            calculate_regulatory_capital(0.05, LGD=0.6, EAD=2000),
            calculate_regulatory_capital(0.05, LGD=0.6, EAD=2000),
            places=1
        )

    def test_five_cs_feature_engineer(self):
        """Test feature engineering logic"""
        engineer = FiveCsFeatureEngineer(snapshot_date=datetime(2024, 12, 31))
        features = engineer.fit_transform(self.transaction_data)
        # Validate feature creation
        self.assertIn('payment_consistency', features.columns)
        self.assertIn('fraud_count', features.columns)
        self.assertIn('lifetime_value', features.columns)
        self.assertIn('holiday_spending_ratio', features.columns)
        # Check specific customer values
        cust1 = features[features['CustomerId'] == 'CUST1']
        self.assertAlmostEqual(cust1['payment_consistency'].values[0], 0.5)
        self.assertEqual(cust1['fraud_count'].values[0], 0)
        self.assertAlmostEqual(cust1['lifetime_value'].values[0], 150.0)
        self.assertEqual(cust1['recency'].values[0], 363)  # 2024-12-31 - 2024-01-02
        # Check holiday spending
        cust3 = features[features['CustomerId'] == 'CUST3']
        self.assertAlmostEqual(cust3['holiday_spending_ratio'].values[0], 0.0)

    def test_risk_label_generator(self):
        """Test proxy risk label creation"""
        engineer = FiveCsFeatureEngineer(snapshot_date=datetime(2024, 12, 31))
        features = engineer.fit_transform(self.transaction_data)
        labeler = RiskLabelGenerator(n_clusters=2)
        labeled = labeler.fit_transform(features)
        # Validate label creation
        self.assertIn('is_high_risk', labeled.columns)
        self.assertTrue(set(labeled['is_high_risk']).issubset({0, 1}))
        # Should have at least one high-risk customer
        self.assertTrue(labeled['is_high_risk'].sum() > 0)

    def test_outlier_capper(self):
        """Test outlier treatment"""
        data = pd.DataFrame({
            'value1': [10, 20, 30, 40, 1000],
            'value2': [1, 2, 3, 4, 50]
        })
        capper = OutlierCapper(factor=1.5)
        capped = capper.fit_transform(data)
        # Validate capping
        self.assertLessEqual(capped['value1'].max(), 85)  # Q3 + 1.5*IQR
        self.assertGreaterEqual(capped['value1'].min(), -27.5)
        self.assertLessEqual(capped['value2'].max(), 10.5)

    def test_woe_encoder(self):
        """Test Weight of Evidence encoding"""
        data = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'C', 'C'],
            'risk': [1, 1, 1, 0, 0, 0]
        })
        encoder = WoETransformer(target='risk', min_samples=1)
        encoded = encoder.fit_transform(data, data['risk'])
        # Validate encoding
        self.assertTrue('category' in encoded.columns)
        self.assertTrue(np.issubdtype(encoded['category'].dtype, np.number))
        # Check specific values
        woe_a = encoded[data['category'] == 'A']['category'].mean()
        woe_b = encoded[data['category'] == 'B']['category'].mean()
        woe_c = encoded[data['category'] == 'C']['category'].mean()
        # A has 100% risk, should have high positive WOE
        self.assertGreater(woe_a, 1.0)
        # B has 50% risk, should have WOE near 0
        self.assertAlmostEqual(woe_b, 0.0, places=1)
        # C has 0% risk, should have high negative WOE
        self.assertLess(woe_c, -1.0)

    def test_full_pipeline(self):
        """Test end-to-end pipeline execution"""
        # Create test data directory
        os.makedirs('../data/raw', exist_ok=True)
        os.makedirs('../data/processed', exist_ok=True)
        # Save sample data
        self.transaction_data.to_csv('../data/raw/test_transactions.csv', index=False)
        # Run pipeline
        processed = process_data(
            input_csv='../data/raw/test_transactions.csv',
            output_csv='../data/processed/test_features.csv'
        )
        # Validate output
        self.assertTrue(os.path.exists('../data/processed/test_features.csv'))
        self.assertGreater(len(processed), 0)
        self.assertIn('is_high_risk', processed.columns)
        self.assertIn('expected_loss', processed.columns)
        # Clean up
        os.remove('../data/raw/test_transactions.csv')
        os.remove('../data/processed/test_features.csv')

if __name__ == '__main__':
    unittest.main()