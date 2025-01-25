import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
from preprocessing import preprocess_data


def preprocess_nn_data() -> tuple:
    """
    Preprocesses the data for the neural network by extracting features (longitude, latitude)
    and labels (species) from the dataset. The features are normalized for consistent scaling.

    Returns:
        tuple: Contains the following:
            - X_train: Normalized training features (longitude, latitude).
            - y_train: Training labels (species).
            - X_val: Normalized validation features (longitude, latitude).
            - y_val: Validation labels (species).
            - X_test: Normalized test features (longitude, latitude).
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

    # Normalize features
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_nn_model(input_shape: int) -> tf.keras.Model:
    """
    Build and compile a neural network model for binary classification.

    Args:
        input_shape (int): The number of input features (like 2 for longitude and latitude).

    Returns:
        tf.keras.Model: Compiled neural network model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_nn_model(model: tf.keras.Model, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> tf.keras.callbacks.History:
    """
    Traisn the neural network model using the training data and validate on the validation set.

    Args:
        model (tf.keras.Model): The neural network model.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation labels.

    Returns:
        tf.keras.callbacks.History: Training history object containing loss and accuracy metrics.
    """
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))
    print(f"Class Weights: {class_weights_dict}")

    # Early stopping callback to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    print("Starting model training...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        class_weight=class_weights_dict,
        callbacks=[early_stopping],
        verbose=1
    )
    print("Model training completed.")
    return history


def evaluate_nn_model(model: tf.keras.Model, X_val: np.ndarray, y_val, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Evaluate the neural network model on validation and test datasets.

    Args:
        model (tf.keras.Model): The trained neural network model.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation labels.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
    """
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


def compute_nn_test_metrics(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Compute and display additional evaluation metrics for the test dataset.

    Args:
        model (tf.keras.Model): The trained neural network model.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
    """
    y_pred_probs = model.predict(X_test).flatten()  # Probabilities
    y_pred = (y_pred_probs > 0.5).astype(int)  # Binary predictions

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_probs)

    print("\nTest Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"AUC:       {auc:.4f}")


def plot_nn_training_history(history: tf.keras.callbacks.History) -> None:
    """
    Plot the training and validation accuracy and loss over epochs.

    Args:
        history (tf.keras.callbacks.History): Training history object containing metrics.
    """
    # Accuracy Plot
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Loss Plot
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
