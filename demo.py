import pandas as pd
from sklearn.metrics import classification_report
import joblib

# Load the test dataset to get feature names
def load_sample_data(file_path):
    """Loads sample data to ensure consistent feature structure."""
    df = pd.read_csv(file_path)
    sample_row = df.sample(n=1, random_state=42)  # Get one row to simulate a demo
    return sample_row.drop(columns=['no_show']), sample_row['no_show']

# Load pre-trained model
def load_pretrained_model(model_path):
    """Loads the pre-trained model from a .pkl file."""
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    print("Model loaded successfully.")
    return model

# Demonstrate model predictions
def demo_predictions(model, X_test, y_test=None):
    """Makes predictions using the pre-trained model and displays output."""
    print("\n--- Making Predictions ---")
    predictions = model.predict(X_test)
    
    # Print predictions
    print(f"Predicted class: {predictions[0]}")
    if y_test is not None:
        print(f"Actual class: {y_test.values[0]}")
        print("\nClassification Report (1-row demo):")
        print(classification_report(y_test, predictions, zero_division=1))

def main():
    # Use the cleaned dataset to generate consistent sample data
    data_path = "./data/dummy_encoded_dataset.csv"
    model_path = "./models/tuned_gradient_boosting_model.pkl"  # Or whichever is faster

    X_sample, y_sample = load_sample_data(data_path)  # Load a sample row of data
    model = load_pretrained_model(model_path)  # Load model

    demo_predictions(model, X_sample, y_sample)  # Demonstrate predictions

if __name__ == "__main__":
    main()
