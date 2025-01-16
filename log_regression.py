from preprocessing import preprocess_data
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1. Load preprocessed data from our function
    data_dict = preprocess_data()

    # 2. Extract relevant parts
    train_data = data_dict["train_data"]
    val_data = data_dict["val_data"]
    test_data = data_dict["test_data"]
    coverages = data_dict["coverages"]

    print("preprocessing:")
    print(len(train_data))
    print(len(test_data))
    print(len(val_data))


if __name__ == "__main__":
    main()

