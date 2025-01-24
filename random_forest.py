import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, log_loss
from preprocessing import preprocess_data

def main():
    # 1. Preprocess the data
    preprocessed_data = preprocess_data()
    train_data = preprocessed_data["train_data"]
    val_data = preprocessed_data["val_data"]
    test_data = preprocessed_data["test_data"]

    # 2. Extract features (longitude, latitude) and labels (species)
    X_train = np.array([(row['dd long'], row['dd lat']) for row in train_data])
    y_train = np.array([row['species'] for row in train_data])

    X_val = np.array([(row['dd long'], row['dd lat']) for row in val_data])
    y_val = np.array([row['species'] for row in val_data])

    X_test = np.array([(row['dd long'], row['dd lat']) for row in test_data])
    y_test = np.array([row['species'] for row in test_data])

    # 3. Scale the features
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # 4. Define and train the Random Forest model
    #    (Using class_weight='balanced' for potential imbalances)
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)

    # 5. Evaluate on TRAIN, VAL, and TEST sets

    # ---- TRAIN EVALUATION ----
    y_pred_train = rf_model.predict(X_train)
    y_pred_train_proba = rf_model.predict_proba(X_train)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    train_log_loss = log_loss(y_train, y_pred_train_proba)


    # ---- VALIDATION EVALUATION ----
    y_pred_val = rf_model.predict(X_val)
    y_pred_val_proba = rf_model.predict_proba(X_val)
    val_accuracy = accuracy_score(y_val, y_pred_val)
    val_log_loss = log_loss(y_val, y_pred_val_proba)

    # ---- TEST EVALUATION ----
    y_pred_test = rf_model.predict(X_test)
    y_pred_test_proba = rf_model.predict_proba(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_log_loss = log_loss(y_test, y_pred_test_proba)

    # 6. Additional metrics on TEST set (precision, recall, ROC-AUC)
    test_precision = precision_score(y_test, y_pred_test)
    test_recall = recall_score(y_test, y_pred_test)
    test_auc = roc_auc_score(y_test, y_pred_test_proba[:, 1])

    # 7. Print results for comparison
    print("=== Random Forest Results ===")
    print("\n--- Training Performance ---")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Train Log Loss: {train_log_loss:.4f}")

    print("\n--- Validation Performance ---")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Log Loss: {val_log_loss:.4f}")

    print("\n--- Test Performance ---")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Log Loss: {test_log_loss:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall:    {test_recall:.4f}")
    print(f"Test ROC AUC:   {test_auc:.4f}")

    inspect_predictions(rf_model, X_test, y_test, num_samples=5, label_encoder=None)

import random

# Suppose 'X_test' and 'y_test' are already defined
# Suppose 'model' is your trained classifier (NN or Random Forest)

def inspect_predictions(model, X_test, y_test, num_samples=5, label_encoder=None):
    y_pred = model.predict(X_test)

    if y_pred.ndim > 1:
        y_pred = y_pred.flatten()
    if np.issubdtype(y_pred.dtype, np.floating):
        y_pred = (y_pred > 0.5).astype(int)

    indices = list(range(len(X_test)))
    random.shuffle(indices)  # shuffle indices
    selected_indices = indices[:num_samples]

    print("\n=== Inspecting Random Predictions ===")
    for i in selected_indices:
        lon, lat = X_test[i]
        true_label = y_test[i]
        pred_label = y_pred[i]

        if label_encoder is not None:
            true_label_str = label_encoder.inverse_transform([true_label])[0]
            pred_label_str = label_encoder.inverse_transform([pred_label])[0]
        else:
            # Just show numeric 0/1 if no label encoder
            true_label_str = str(true_label)
            pred_label_str = str(pred_label)

        print(f"Point {i} -> (Lon={lon:.3f}, Lat={lat:.3f}): "
              f"Actual = {true_label_str}, Predicted = {pred_label_str}")


if __name__ == "__main__":
    main()