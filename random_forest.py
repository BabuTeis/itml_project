from preprocessing import preprocess_data
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1. Load preprocessed data
    data_dict = preprocess_data()

    # 2. Extract relevant parts
    train_data = data_dict["train_data"]       # structured array with fields: species, dd long, dd lat
    val_data   = data_dict["val_data"]
    test_data  = data_dict["test_data"]
    coverages  = data_dict["coverages"]

    # -------------------------------------------------------------------------
    # A) PLOT THE TWO SPECIES IN THE TRAIN SET
    # -------------------------------------------------------------------------
    # The train_data has .species = 0 or 1, .['dd long'], .['dd lat'] for each occurrence
    train_species = train_data['species']      # array of 0 or 1
    train_long    = train_data['dd long']
    train_lat     = train_data['dd lat']

    plt.figure(figsize=(10, 6))
    for s in np.unique(train_species):  # should be [0, 1]
        mask = (train_species == s)
        plt.scatter(train_long[mask], train_lat[mask],
                    label=f"Species {s}", s=10)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Species Distribution in Training Data (Encoded 0 or 1)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # -------------------------------------------------------------------------
    # B) VISUALIZE ONE COVERAGE LAYER (optional, e.g. layer 3)
    # -------------------------------------------------------------------------
    layer = coverages[3]
    plt.figure(figsize=(10, 6))
    plt.imshow(layer, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Environmental Feature Value')
    plt.title('Environmental Feature Layer 3')
    plt.show()

    # -------------------------------------------------------------------------
    # C) RANDOM FOREST CLASSIFIER
    # -------------------------------------------------------------------------
    # For a *simple demonstration*, let's predict species from [longitude, latitude].
    # (A real approach might incorporate coverage data or other features.)
    X_train = np.column_stack((train_data['dd long'], train_data['dd lat']))
    y_train = train_data['species']

    X_val = np.column_stack((val_data['dd long'], val_data['dd lat']))
    y_val = val_data['species']

    X_test = np.column_stack((test_data['dd long'], test_data['dd lat']))
    y_test = test_data['species']

    # Create and train the Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Evaluate on validation and test
    val_accuracy = rf.score(X_val, y_val)
    test_accuracy = rf.score(X_test, y_test)

    print(f"Random Forest Validation Accuracy: {val_accuracy:.3f}")
    print(f"Random Forest Test Accuracy:       {test_accuracy:.3f}")

if __name__ == "__main__":
    main()
