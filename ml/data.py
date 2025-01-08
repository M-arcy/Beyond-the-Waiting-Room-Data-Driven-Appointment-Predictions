import pandas as pd

def load_data(file_path):
    """Loads the health dataset from the given file path."""
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(df):
    """Cleans and prepares the dataset."""
    # Drop any duplicates if present
    df = df.drop_duplicates()

    # Convert datetime columns
    df['scheduled_day'] = pd.to_datetime(df['scheduled_day'])
    df['appointment_day'] = pd.to_datetime(df['appointment_day'])

    # Convert boolean columns
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Convert "no_show" to numeric (0 for No, 1 for Yes)
    df['no_show'] = df['no_show'].map({'No': 0, 'Yes': 1})

    # Add waiting time feature
    df['waiting_time'] = (df['appointment_day'] - df['scheduled_day']).dt.days

    # Remove rows with negative waiting times
    df = df[df['waiting_time'] >= 0]

    # Handle missing, incorrect or outlier age values
    df = df[(df['age'] >= 0) & (df['age'] <= 105)]

    print("Data cleaned successfully.")
    return df
