import pandas as pd
from model import load_data, split_data, evaluate_model, save_model, get_models
from sklearn.model_selection import train_test_split, cross_val_score

# Function to compare models and select the best one
def compare_and_select_best_model(X, y):
    """
    Compares Logistic Regression, Random Forest, and Gradient Boosting models and selects the best based on F1-scores.
    Prints evaluation metrics for all models and returns the best model.
    """
    # Define only the selected models
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=600, random_state=42)  # Updated to use n_estimators=600
    }

    model_scores = {}

    print("--- Model Comparison ---")
    for name, model in models.items():
        accuracy_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        f1_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
        recall_scores = cross_val_score(model, X, y, cv=5, scoring='recall')
        precision_scores = cross_val_score(model, X, y, cv=5, scoring='precision')

        # Store F1-score for ranking
        model_scores[name] = f1_scores.mean()

        # Print performance metrics for each model
        print(f"\n{name}:")
        print(f"  - Accuracy mean: {accuracy_scores.mean():.4f}")
        print(f"  - F1-score mean: {f1_scores.mean():.4f}")
        print(f"  - Precision mean: {precision_scores.mean():.4f}")
        print(f"  - Recall mean: {recall_scores.mean():.4f}")

    # Select the model with the best F1-score
    best_model_name = max(model_scores, key=model_scores.get)
    best_model = models[best_model_name]

    print(f"\nBest Model Selected: {best_model_name} (F1-score: {model_scores[best_model_name]:.4f})")
    return best_model

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

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Compare models and select the best one
    best_model = compare_and_select_best_model(X_train, y_train)

    # Train the selected model
    best_model.fit(X_train, y_train)

    # Evaluate the selected model
    evaluate_model(best_model, X_test, y_test)

    # Save the selected model
    save_model(best_model, model_path)

if __name__ == "__main__":
    main()
