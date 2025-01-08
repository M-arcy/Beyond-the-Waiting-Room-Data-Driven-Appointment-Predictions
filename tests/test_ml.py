import pytest
from data import clean_data
from model import train_model 
import pandas as pd

def test_clean_data_runs():
    """Test that clean_data function runs without throwing errors."""
    df = pd.DataFrame({'age': [25, 30, 40], 'no_show': ['No', 'Yes', 'No']})
    try:
        cleaned_df = clean_data(df)
        assert True  # If no exceptions, pass the test
    except Exception:
        assert False  # Fail the test if an exception is raised

def test_waiting_time_added():
    """Test that the waiting_time column is added."""
    df = pd.DataFrame({
        'scheduled_day': ['2023-01-01'],
        'appointment_day': ['2023-01-03'],
        'age': [25],
        'no_show': ['No']
    })
    df['scheduled_day'] = pd.to_datetime(df['scheduled_day'])
    df['appointment_day'] = pd.to_datetime(df['appointment_day'])
    
    cleaned_df = clean_data(df)
    assert 'waiting_time' in cleaned_df.columns  # Ensure waiting_time exists

def test_no_missing_values():
    """Test that there are no NaN values in the cleaned dataframe."""
    df = pd.DataFrame({'age': [25, 30, 40], 'no_show': ['No', 'Yes', 'No']})
    cleaned_df = clean_data(df)
    assert cleaned_df.isnull().sum().sum() == 0  # Ensure no missing values
