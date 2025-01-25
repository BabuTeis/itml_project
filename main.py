from log_regression import (
    preprocess_lr_data,
    train_lr_model,
    evaluate_lr_model,
    inspect_lr_predictions,
)
from random_forest import (
    preprocess_rf_data,
    train_rf_model,
    evaluate_rf_model,
    inspect_predictions,
)
from neural_networks import (
    preprocess_nn_data,
    build_nn_model,
    train_nn_model,
    evaluate_nn_model,
    compute_nn_test_metrics,
    plot_nn_training_history,
)
from visualize import visual_plots


def main():
    print("Choose a model to train and evaluate:")
    print("1. Logistic Regression")
    print("2. Random Forest")
    print("3. Realistic Random Forest")
    print("4. Neural Network")
    print("5. Visualize the dataset")
    choice = input("Enter 1, 2, 3, 4 or 5: ")

    if choice == "1":
        print("\n=== Logistic Regression ===")
        X_train, y_train, X_val, y_val, X_test, y_test = preprocess_lr_data()
        model = train_lr_model(X_train, y_train)
        evaluate_lr_model(model, X_train, y_train, "Training")
        evaluate_lr_model(model, X_val, y_val, "Validation")
        evaluate_lr_model(model, X_test, y_test, "Test")
        inspect_lr_predictions(model, X_test, y_test)

    elif choice == "2":
        print("\n=== Random Forest ===")
        X_train, y_train, X_test, y_test = preprocess_rf_data()
        model = train_rf_model(X_train, y_train)
        evaluate_rf_model(model, X_train, y_train, "Training")
        evaluate_rf_model(model, X_test, y_test, "Test")
        inspect_predictions(model, X_test, y_test)

    elif choice == "3":
        print("\n=== Realistic Random Forest ===")
        X_train, y_train, X_test, y_test = preprocess_rf_data(shorten=True)
        model = train_rf_model(X_train, y_train)
        evaluate_rf_model(model, X_train, y_train, "Training")
        evaluate_rf_model(model, X_test, y_test, "Test")
        inspect_predictions(model, X_test, y_test)

    elif choice == "4":
        print("\n=== Neural Network ===")
        X_train, y_train, X_val, y_val, X_test, y_test = preprocess_nn_data()
        model = build_nn_model(input_shape=X_train.shape[1])
        history = train_nn_model(model, X_train, y_train, X_val, y_val)
        evaluate_nn_model(model, X_val, y_val, X_test, y_test)
        compute_nn_test_metrics(model, X_test, y_test)
        plot_nn_training_history(history)

    elif choice == "5":
        while True:
            try:
                print("\nWrite a number corresponding to the coverage layer of interest (0-13):")
                number = int(input())
                if 0 <= number < 14:
                    break
                else:
                    print("Invalid number, please try again.")
            except ValueError:
                print("Invalid input. Please enter an integer.")

        print("\nShowing the plots...")
        visual_plots(number)

    else:
        print("Invalid choice. Please enter 1, 2, 3, 4 or 5.")


if __name__ == "__main__":
    main()
