# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using training dataset and evaluates using test dataset. Saves trained model.
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

def parse_args():
    """Step 1: Define arguments for train data, test data, model output, and RandomForest hyperparameters."""
    parser = argparse.ArgumentParser("train")

    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to directory containing the training dataset (CSV).")
    parser.add_argument("--test_data", type=str, required=True,
                        help="Path to directory containing the test dataset (CSV).")
    parser.add_argument("--model_output", type=str, required=True,
                        help="Path where the trained model will be saved.")

    # RandomForest hyperparameters
    parser.add_argument("--n_estimators", type=int, default=100,
                        help="Number of trees in the random forest.")
    parser.add_argument("--max_depth", type=int, default=None,
                        help="Maximum depth of each decision tree.")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for reproducibility.")

    args = parser.parse_args()
    return args


def main(args):
    """Train and evaluate a RandomForestRegressor using MLflow tracking."""

    # Step 2: Read the train and test datasets from provided paths
    train_df = pd.read_csv(Path(args.train_data) / "train.csv")
    test_df = pd.read_csv(Path(args.test_data) / "test.csv")

    # Step 3: Split data into features (X) and target (y)
    target_col = "price"  # Adjust to your dataset’s target column name
    y_train = train_df[target_col]
    X_train = train_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    X_test = test_df.drop(columns=[target_col])

    # Step 4: Initialize and train RandomForestRegressor
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state
    )
    model.fit(X_train, y_train)

    # Step 5: Log hyperparameters to MLflow
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_param("random_state", args.random_state)

    # Step 6: Make predictions and evaluate using Mean Squared Error
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on test data: {mse:.4f}")

    # Step 7: Log the evaluation metric and save model
    mlflow.log_metric("Mean_Squared_Error", float(mse))
    mlflow.sklearn.save_model(sk_model=model, path=args.model_output)

    print(f"Model saved to: {args.model_output}")


if __name__ == "__main__":
    mlflow.start_run()

    args = parse_args()

    print("\n===== TRAINING CONFIGURATION =====")
    print(f"Train data path: {args.train_data}")
    print(f"Test data path: {args.test_data}")
    print(f"Model output path: {args.model_output}")
    print(f"Number of estimators: {args.n_estimators}")
    print(f"Max depth: {args.max_depth}")
    print(f"Random state: {args.random_state}")
    print("==================================\n")

    main(args)

    mlflow.end_run()


