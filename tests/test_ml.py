import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from data import clean_data, load_data  # Import from root-level data.py
from model import train_model  # Import from root-level model.py

#test 0 - check that pandas can be imported successfully

def test_pandas_import():
    """Check that pandas can be imported successfully."""
    assert pd.__version__ is not None, "Pandas is not installed properly."

# Test 1: Check that `clean_data` returns a DataFrame without NaN values.
def test_clean_data():
    """
    Test that clean_data removes missing values and returns the correct columns.
    """
    data = pd.DataFrame({
        'age': [25, 40, None, 30],
        'scheduled_day': ['2023-01-01', '2023-01-05', '2023-01-10', '2023-01-15'],
        'appointment_day': ['2023-01-03', '2023-01-07', '2023-01-12', '2023-01-20'],
        'no_show': ['No', 'Yes', 'No', 'Yes']
    })
    data['scheduled_day'] = pd.to_datetime(data['scheduled_day'])
    data['appointment_day'] = pd.to_datetime(data['appointment_day'])

    cleaned_data = clean_data(data)
    
    # Check for no missing values
    assert cleaned_data.isnull().sum().sum() == 0, "There should be no NaN values after cleaning"
    assert 'waiting_time' in cleaned_data.columns, "The cleaned data should have a 'waiting_time' column"

# Test 2: Check that `train_model` returns a LogisticRegression instance.
def test_train_model():
    """
    Test that train_model returns a LogisticRegression instance.
    """
    # Dummy data to simulate training input
    X_train = np.random.rand(50, 5)  # 50 samples, 5 features
    y_train = np.random.randint(0, 2, 50)  # Binary target (0 or 1)
    
    model = train_model(X_train, y_train)
    
    assert isinstance(model, LogisticRegression), "The model should be a LogisticRegression instance"

# Test 3: Check precision, recall, and F1-score calculation.
def test_metrics():
    """
    Test that precision, recall, and F1-score are between 0 and 1.
    """
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    assert 0 <= precision <= 1, "Precision should be between 0 and 1"
    assert 0 <= recall <= 1, "Recall should be between 0 and 1"
    assert 0 <= f1 <= 1, "F1 score should be between 0 and 1"
