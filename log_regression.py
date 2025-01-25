import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, log_loss
from preprocessing import preprocess_data


def preprocess_lr_data() -> tuple:
    """
    Extract the features (longitude, latitude) and labels (species) from the preprocessed dataset for the model.

    Returns:
        tuple:
            - X_train: Training features (longitude, latitude).
            - y_train: Training labels (species).
            - X_val: Validation features (longitude, latitude).
            - y_val: Validation labels (species).
            - X_test: Test features (longitude, latitude).
            - y_test: Test labels (species).
    """
    preprocessed_data = preprocess_data()
    train_data = preprocessed_data["train_data"]
    val_data = preprocessed_data["val_data"]
    test_data = preprocessed_data["test_data"]

    X_train = np.array([(row['dd long'], row['dd lat']) for row in train_data])
    y_train = np.array([row['species'] for row in train_data])

    X_val = np.array([(row['dd long'], row['dd lat']) for row in val_data])
    y_val = np.array([row['species'] for row in val_data])

    X_test = np.array([(row['dd long'], row['dd lat']) for row in test_data])
    y_test = np.array([row['species'] for row in test_data])

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_lr_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """
    Train the Logistic Regression model using the training data.

    Args:
        X_train (np.ndarray): Training features (longitude, latitude).
        y_train (np.ndarray): Training labels (species).

    Returns:
        LogisticRegression: The trained Logistic Regression model
    """
    # Calculates class weights to handle imbalanced data
    unique_classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train)
    class_weights_dict = {cls: weight for cls, weight in zip(unique_classes, class_weights)}

    # Train the model.
    model = LogisticRegression(
        solver='liblinear',
        class_weight=class_weights_dict,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_lr_model(model: LogisticRegression, X: np.ndarray, y: np.ndarray, dataset_name: str ="Dataset") -> tuple:
    """
    Evaluate the trained Logistic Regression model and print the metrics.

    Args:
        model (LogisticRegression): The trained model.
        X (np.ndarray): The longitude and latitude for evaluation.
        y (np.ndarray): True labels for evaluation.
        dataset_name (str): Name of the dataset.

    Returns:
        tuple:
            - accuracy: Accuracy metric.
            - precision: Precision metric
            - recall: Recall metric.
            - auc: Area Under the ROC Curve (AUC) metric.
    """
    # Predict probabilities and labels
    y_pred_probs = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    # Find (and print) the metrics
    accuracy = accuracy_score(y, y_pred)
    log_loss_value = log_loss(y, y_pred_probs)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    auc = roc_auc_score(y, y_pred_probs)

    print(f"\n{dataset_name} Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"AUC:       {auc:.4f}")

    return accuracy, precision, recall, auc


def inspect_lr_predictions(model: LogisticRegression, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Inspect predictions for a test point that the user selected.

    Args:
        model (LogisticRegression): The model.
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
