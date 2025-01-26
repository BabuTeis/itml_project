import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, log_loss
from preprocessing import preprocess_data


def preprocess_rf_data(shorten: bool = False) -> tuple:
    """
    Extract the features (longitude, latitude) and labels (species) from the preprocessed dataset for the model.
    We do not use a validation set because Random Forest functions without it.

    Returns:
        tuple:
            - X_train: Training features (longitude, latitude).
            - y_train: Training labels (species).
            - X_test: Test features (longitude, latitude).
            - y_test: Test labels (species).
    """
    preprocessed_data = preprocess_data()
    train_data = preprocessed_data["train_data"]
    test_data = preprocessed_data["test_data"]

    # Optionally shorten the dataset for results with more unique coordinates
    if shorten == True:
        train_data = train_data[:150]
        test_data = test_data[:40]

    X_train = np.array([(row['dd long'], row['dd lat']) for row in train_data])
    y_train = np.array([row['species'] for row in train_data])

    X_test = np.array([(row['dd long'], row['dd lat']) for row in test_data])
    y_test = np.array([row['species'] for row in test_data])

    return X_train, y_train, X_test, y_test


def train_rf_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Train the random Forest model using the training data.

    Args:
        X_train (np.ndarray): Training features (longitude, latitude).
        y_train (np.ndarray): Training labels (species).

    Returns:
        RandomForestClassifier: The trained Random Forest model
    """
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)
    return rf_model


def evaluate_rf_model(rf_model: RandomForestClassifier, X: np.ndarray, y: np.ndarray, set_name: str) -> tuple:
    """
    Evaluate the trained Random Forest model and print the metrics.

    Args:
        model (RandomForestClassifier): The trained model.
        X (np.ndarray): The longitude and latitude for evaluation.
        y (np.ndarray): True labels for evaluation.
        set_name (str): Name of the dataset.

    Returns:
        tuple:
            - accuracy: Accuracy metric.
            - log_loss_value: The logarithmic loss.
            - precision: Precision metric
            - recall: Recall metric.
            - auc: Area Under the ROC Curve (AUC) metric.
    """
    # Predict labels and probabilities
    y_pred = rf_model.predict(X)
    y_pred_proba = rf_model.predict_proba(X)

    # Calculate all of the metrics
    accuracy = accuracy_score(y, y_pred)
    log_loss_value = log_loss(y, y_pred_proba)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    auc = roc_auc_score(y, y_pred_proba[:, 1])

    print(f"\n--- {set_name} Performance ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"ROC AUC:   {auc:.4f}")

    return accuracy, log_loss_value, precision, recall, auc


def inspect_predictions(model: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Inspect predictions for a test point that the user selected.

    Args:
        model (RandomForestClassifier): The model.
        X_test (np.ndarray): Test features (longitude, latitude).
        y_test (np.ndarray): True test labels.
    """
    y_pred = model.predict(X_test)

    while True:
        try:
            index = int(input(f"Enter an index (0 to {len(X_test) - 1}) to inspect: "))
            if 0 <= index < len(X_test):
                break
            else:
                print("Invalid index. Please enter a number within the valid range.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    long, lat = X_test[index]
    true_label = y_test[index]
    pred_label = y_pred[index]

    print("\n=== Inspecting Selected Prediction ===")
    print(f"Point {index} -> (Longtitude={long:.3f}, Latitude={lat:.3f}): ")
    print(f"Actual = {true_label}, Predicted = {pred_label}")
