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

# Function to tune learning rate, max depth, and min samples split
def tune_gradient_boosting(X_train, y_train):
    """Tunes hyperparameters for Gradient Boosting Classifier."""
    # Hyperparameter grid
    learning_rates = [0.05, 0.1]
    max_depths = [3, 5]
    min_samples_splits = [2, 5]
    
    best_params = None
    best_accuracy = 0

    print("\n--- Hyperparameter Tuning: Gradient Boosting ---")
    for lr in learning_rates:
        for depth in max_depths:
            for min_split in min_samples_splits:
                gbc = GradientBoostingClassifier(
                    learning_rate=lr,
                    max_depth=depth,
                    min_samples_split=min_split,
                    n_estimators=600,  # Using the best n_estimators from previous tuning
                    random_state=42
                )
                scores = cross_val_score(gbc, X_train, y_train, cv=5, scoring='accuracy')
                mean_accuracy = scores.mean()
                print(f"learning_rate: {lr}, max_depth: {depth}, min_samples_split: {min_split}, Accuracy: {mean_accuracy:.4f}")

                if mean_accuracy > best_accuracy:
                    best_accuracy = mean_accuracy
                    best_params = {
                        'learning_rate': lr,
                        'max_depth': depth,
                        'min_samples_split': min_split
                    }

    print(f"\nBest Parameters: {best_params} with Accuracy: {best_accuracy:.4f}")
    return best_params

# Function to train and evaluate the final model
def train_final_model(X_train, X_test, y_train, y_test, best_params):
    """Trains and evaluates the Gradient Boosting model with the best hyperparameters."""
    gbc = GradientBoostingClassifier(
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        n_estimators=600,  # Best n_estimators from previous tuning
        random_state=42
    )
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
    joblib.dump(gbc, "./models/tuned_gradient_boosting_model.pkl")
    print("Final model saved to './models/tuned_gradient_boosting_model.pkl'")

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

    # Tune the Gradient Boosting model to find the best parameters
    best_params = tune_gradient_boosting(X_train, y_train)

    # Train and evaluate the final model
    train_final_model(X_train, X_test, y_train, y_test, best_params)

if __name__ == "__main__":
    main()
