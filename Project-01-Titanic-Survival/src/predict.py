import pickle
import pandas as pd
from preprocess import preprocess_data

def predict_survival(test_data_path, model_path):
    # Load test data
    test_df = pd.read_csv(test_data_path)

    # Preprocess test data
    X_test, passenger_id = preprocess_data(test_df, is_train=False)

    # Load trained model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Predict
    predictions = model.predict(X_test)

    # Create submission dataframe
    submission = pd.DataFrame({
        "PassengerId": passenger_id,
        "Survived": predictions
    })

    return submission


if __name__ == "__main__":
    submission = predict_survival(
        "../data/test.csv",
        "../models/titanic_model.pkl"
    )
    submission.to_csv("../models/predictions.csv", index=False)
    print("Predictions saved to models/predictions.csv")
