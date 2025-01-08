import os
from sklearn.model_selection import train_test_split
from ml.data import load_data, clean_data
from ml.model import train_model, evaluate_model, save_model

# File paths
DATA_PATH = "./data/noshowappointments-kagglev2-may-2016.csv"
MODEL_PATH = "./ml/logistic_model.pkl"

def main():
    # Load data
    df = load_data(DATA_PATH)
    if df is None:
        return

    # Clean data
    df = clean_data(df)

    # Define features and target
    X = df.drop(columns=['no_show'])
    y = df['no_show']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate model
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    # Save model
    save_model(model, MODEL_PATH)

if __name__ == "__main__":
    main()
