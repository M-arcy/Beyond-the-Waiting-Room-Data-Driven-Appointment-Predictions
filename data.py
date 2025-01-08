import pandas as pd
# Disable scientific notation for float and int display
pd.set_option('display.float_format', '{:.0f}'.format)

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
    df.columns = df.columns.str.lower().str.replace(' ', '_', regex=True).str.replace('-', '_', regex=True)

    column_renames = {
        'hipertension': 'hypertension',
        'handcap': 'handicap',
        'patientid': 'patient_id',
        'appointmentid': 'appointment_id',
        'scheduledday': 'scheduled_day',
        'appointmentday': 'appointment_day'
    }
    df = df.rename(columns=column_renames)
    
    #verify expected columns
    required_columns = ['scheduled_day', 'appointment_day', 'age', 'no_show', 'gender']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns in data: {missing_cols}")

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
    
    df['age'] = df['age'].fillna(df['age'].median())  # Fill missing age values with median
    #convert patient and appointment_ID to int
    df['patient_id'] = df['patient_id'].astype('int64')
    df['appointment_id'] = df['appointment_id'].astype('int64')

    # One-hot encode gender column (drop 'F' and keep 'M' as 'gender_M')
    df = pd.get_dummies(df, columns=['gender'], drop_first=True)
    
    df['gender_M'] = df['gender_M'].astype(int)  # Convert gender to 0/1

    
    print("Data cleaned successfully.")
    return df

if __name__ == "__main__":
    df = load_data("data/noshowappointments-kagglev2-may-2016.csv")
    if df is not None:
        cleaned_df = clean_data(df)  # Call clean_data
        print(cleaned_df.head(1))  # Show the first row of the cleaned data to confirm
        print(cleaned_df.info())  # Show summary of columns and data types
        print(cleaned_df.describe())  # Show basic statistics summary
        cleaned_df.to_csv("data/cleaned_appointments.csv", index=False)
