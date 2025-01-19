import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from preprocessing import preprocess_data


def main() -> None:
    """
    Main function to preprocess data, train a neural network,
    and evaluate performance.
    """
    # Preprocess the data
    preprocessed_data = preprocess_data()

    train_data = preprocessed_data["train_data"]
    val_data = preprocessed_data["val_data"]
    test_data = preprocessed_data["test_data"]

    # Extract features and labels
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

    # Compute class weights
    class_weights = compute_class_weight('balanced',
                                         classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))
    print(f"Class Weights: {class_weights_dict}")

    # Define the neural network
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Early stopping to avoid overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    # Train the model
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

    # Evaluate the model
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}, "
          f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Plot training history
    plot_training_history(history)


def plot_training_history(history: tf.keras.callbacks.History) -> None:
    """
    Plot the training and validation accuracy and loss over epochs.

    Args:
        history (tf.keras.callbacks.History): Training history object.
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


if __name__ == "__main__":
    main()
