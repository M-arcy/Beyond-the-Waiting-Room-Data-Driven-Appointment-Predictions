import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Function to load the data
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
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Function to perform hyperparameter tuning for Gradient Boosting
def tune_gradient_boosting(X_train, y_train):
    """Tunes n_estimators for Gradient Boosting Classifier."""
    n_estimators_list = [100, 200, 400, 600, 800, 1000, 1200]
    best_n_estimators = None
    best_accuracy = 0

    print("\n--- Hyperparameter Tuning: Gradient Boosting ---")
    for n_estimators in n_estimators_list:
        gbc = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
        scores = cross_val_score(gbc, X_train, y_train, cv=3, scoring='accuracy')  # Reduced to `cv=3` for faster runtime
        mean_accuracy = scores.mean()
        print(f"n_estimators: {n_estimators}, Accuracy: {mean_accuracy:.4f}")

        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_n_estimators = n_estimators

    print(f"\nBest n_estimators: {best_n_estimators} with Accuracy: {best_accuracy:.4f}")
    return best_n_estimators

# Function to train and evaluate the final model
def train_final_model(X_train, X_test, y_train, y_test, best_n_estimators):
    """Trains and evaluates the Gradient Boosting model with the best n_estimators."""
    gbc = GradientBoostingClassifier(n_estimators=best_n_estimators, random_state=42)
    gbc.fit(X_train, y_train)
    print("\nFinal Gradient Boosting Model Trained!")
    y_pred = gbc.predict(X_test)

    # Display evaluation results
    print("\n--- Final Model Evaluation ---")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save the final model
    joblib.dump(gbc, "./models/gradient_boosting_model.pkl")
    print("Final model saved to './models/gradient_boosting_model.pkl'")

# Main function
def main():
    data_path = "./data/dummy_encoded_dataset.csv"

    # Load data
    df = load_data(data_path)
    if df is None or df.empty:
        print("Error: Data could not be loaded or is empty.")
        return

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = split_data(df)

    # Tune the Gradient Boosting model to find the best n_estimators
    best_n_estimators = tune_gradient_boosting(X_train, y_train)

    # Train and evaluate the final model
    train_final_model(X_train, X_test, y_train, y_test, best_n_estimators)

if __name__ == "__main__":
    main()
