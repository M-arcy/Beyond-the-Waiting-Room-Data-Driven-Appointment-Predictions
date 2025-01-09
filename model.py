import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Function to load the cleaned data
def load_data(file_path):
    """Loads the dummy-encoded dataset from the CSV file."""
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully from {file_path}")
    return df

# Function to split data into features and target
def split_data(df):
    """Splits the DataFrame into training and test sets."""
    X = df.drop(columns=['no_show'])  # Features
    y = df['no_show']  # Target
    return X, y

# Define models for comparison
def get_models():
    """Returns a dictionary of models to compare."""
    return {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(class_weight='balanced', random_state=42)
    }

# Function to compare models
def compare_models(X, y):
    """Compares multiple models and prints cross-validation scores."""
    models = get_models()
    print("--- Model Comparison ---")
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring='f1')
        print(f"{name}: F1-score mean: {scores.mean():.4f}, Std: {scores.std():.4f}")

# Function to evaluate the chosen model
def evaluate_model(model, X_test, y_test):
    """Evaluates the model and prints metrics."""
    y_pred = model.predict(X_test)
    print("--- Final Model Evaluation ---")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Function to save the trained model
def save_model(model, path):
    """Saves the trained logistic regression model."""
    joblib.dump(model, path)
    print(f"Model saved to {path}")

# Main function
def main():
    data_path = "./data/dummy_encoded_dataset.csv"
    model_path = "./models/best_model.pkl"

    # Load data
    df = load_data(data_path)
    if df is None or df.empty:
        print("Error: Data could not be loaded or is empty.")
        return

    # Split data into features and target
    X, y = split_data(df)

    # Compare models
    compare_models(X, y)

    # Train the best model (Logistic Regression in this case)
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the chosen model
    evaluate_model(model, X_test, y_test)

    # Save the model
    save_model(model, model_path)

if __name__ == "__main__":
    main()
