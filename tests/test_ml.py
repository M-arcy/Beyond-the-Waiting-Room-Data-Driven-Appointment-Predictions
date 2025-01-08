import pytest
from ml.data import clean_data
import pandas as pd

def test_clean_data_no_nulls():
    df = pd.DataFrame({'age': [25, 30, 40], 'no_show': ['No', 'Yes', 'No']})
    cleaned_df = clean_data(df)
    assert cleaned_df.isnull().sum().sum() == 0

def test_clean_data_no_negatives():
    df = pd.DataFrame({'age': [25, -5, 40], 'no_show': ['No', 'Yes', 'No']})
    cleaned_df = clean_data(df)
    assert (cleaned_df['age'] >= 0).all()

def test_waiting_time_column_exists():
    df = pd.DataFrame({'scheduled_day': ['2023-01-01'], 'appointment_day': ['2023-01-03']})
    df['scheduled_day'] = pd.to_datetime(df['scheduled_day'])
    df['appointment_day'] = pd.to_datetime(df['appointment_day'])
    cleaned_df = clean_data(df)
    assert 'waiting_time' in cleaned_df.columns
