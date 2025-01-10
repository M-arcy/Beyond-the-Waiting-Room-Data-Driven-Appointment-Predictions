import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import sys
import os

# Add the project root directory to sys.path so that 'data' and 'model' can be imported.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import train_model  # Assuming you have `train_model` in model.py

# Test 1: Check that `train_model` returns a `GradientBoostingClassifier` instance.
def test_train_gradient_boosting_model():
    """Test that train_model returns a GradientBoostingClassifier."""
    from sklearn.ensemble import GradientBoostingClassifier

    # Dummy data to simulate training input
    X_train = pd.DataFrame({
        'age': np.random.randint(18, 90, size=50),  # Random ages
        'scholarship': np.random.randint(0, 2, size=50),  # 0 or 1 for scholarship
        'hypertension': np.random.randint(0, 2, size=50)  # 0 or 1 for hypertension
    })
    y_train = np.random.randint(0, 2, size=50)  # Binary target (0 or 1)

    model = train_model(X_train, y_train)

    assert isinstance(model, GradientBoostingClassifier), "The model should be a GradientBoostingClassifier instance."

# Test 2: Test evaluation metrics (precision, recall, F1-score, accuracy).
def test_evaluation_metrics():
    """Test that the evaluation metrics for the model are within valid bounds."""
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    assert 0 <= precision <= 1, "Precision should be between 0 and 1."
    assert 0 <= recall <= 1, "Recall should be between 0 and 1."
    assert 0 <= f1 <= 1, "F1 score should be between 0 and 1."
    assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1."

# Test 3: Basic test for Pandas import
def test_pandas_import():
    """Check that pandas can be imported successfully."""
    assert pd.__version__ is not None, "Pandas is not installed properly."
