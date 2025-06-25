import os
import json

# Define the folder structure
folders = [
    ".github/workflows",
    "data/raw",
    "data/processed",
    "notebooks",
    "src/api",
    "tests",
]

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Created directory: {folder}")

# Define files with their content
files = {
    # CI/CD Pipeline
    ".github/workflows/ci.yml": """name: CI Pipeline
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run tests
      run: pytest tests/
    - name: Lint code
      run: flake8 src/ --count --show-source --statistics
""",
    # Git ignore rules
    ".gitignore": """# Python artifacts
__pycache__/
*.pyc
*.pyo
*.pyd

# Virtual environments
.env
.venv
venv/
env/

# Data files
data/

# Notebook checkpoints
*.ipynb_checkpoints/

# OS files
.DS_Store

# Logs
*.log

# IDE files
.idea/
.vscode/
""",
    # Requirements with full stack
    "requirements.txt": """# Core data analysis
numpy==1.26.4
pandas==2.2.1
scikit-learn==1.4.1.post1
joblib==1.3.2

# Visualization
matplotlib==3.8.3
seaborn==0.13.2

# ML tools
mlflow==2.10.1
xgboost==2.0.3

# API dependencies
fastapi==0.110.0
uvicorn==0.29.0
pydantic==2.6.0

# Dev tools
pytest==8.0.0
flake8==7.0.0
jupyter==1.0.0
python-dotenv==1.0.1
""",
    # Main README with Task 1 placeholder
    "README.md": """# Credit Risk Modeling Project

## Project Structure
- `src/` - Main source code
  - `data_processing.py` - Feature engineering pipeline
  - `train.py` - Model training script
  - `predict.py` - Inference script
  - `api/` - API implementation
- `notebooks/` - Exploratory analysis notebooks
- `tests/` - Unit tests
- `data/` - Raw and processed data (ignored by git)

## Setup
1. Clone the repository
2. Create virtual environment: `python -m venv venv`
3. Activate environment:
   - Windows: `venv\\Scripts\\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run tests: `pytest tests/`

## Credit Scoring Business Understanding
*To be completed for Task 1*

### 1. Basel II Accord's Influence
*Your response here*

### 2. Proxy Variable Necessity and Risks
*Your response here*

### 3. Model Selection Trade-offs
*Your response here*
""",
    # Initial EDA notebook
    "notebooks/1.0-eda.ipynb": json.dumps({
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Exploratory Data Analysis\n",
                    "## Credit Risk Modeling Project"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "# Load data\n",
                    "df = pd.read_csv('../data/raw/transactions.csv')\n",
                    "\n",
                    "# Initial exploration\n",
                    "print(f'Dataset shape: {df.shape}')\n",
                    "print(f'Missing values:\\n{df.isnull().sum()}')\n",
                    "df.describe()"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }),
    # Core scripts
    "src/__init__.py": "",
    "src/data_processing.py": """import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class RFMCalculator(BaseEstimator, TransformerMixin):
    \"\"\"Calculates RFM features from transaction data\"\"\"
    def __init__(self, snapshot_date='2025-06-25'):
        self.snapshot_date = pd.to_datetime(snapshot_date)
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # RFM calculation logic
        return X

# Main processing pipeline
def create_pipeline():
    return Pipeline([
        ('rfm_features', RFMCalculator()),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
""",
    "src/train.py": """import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from .data_processing import create_pipeline

# Placeholder for training logic
def train_model():
    print("Training workflow will be implemented here")
""",
    "tests/test_data_processing.py": """import pytest
from src.data_processing import RFMCalculator
import pandas as pd

# Test data
@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'CustomerId': [1, 1, 2, 2],
        'Amount': [100, 200, 50, 75],
        'TransactionStartTime': ['2025-06-20', '2025-06-22', '2025-06-18', '2025-06-15']
    })

def test_rfm_calculator(sample_data):
    \"\"\"Test RFM feature calculation\"\"\"
    transformer = RFMCalculator(snapshot_date='2025-06-25')
    transformed = transformer.transform(sample_data)
    
    # Add your assertions here
    assert 'Recency' in transformed.columns
    assert 'Frequency' in transformed.columns
    assert 'Monetary' in transformed.columns
"""
}

# Create files
for file_path, content in files.items():
    with open(file_path, 'w') as f:
        if file_path.endswith('.ipynb'):
            f.write(content)  # Jupyter notebook is already in string format
        else:
            f.write(content)
    print(f"Created file: {file_path}")

print("Repository setup complete!")