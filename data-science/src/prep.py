# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Prepares raw data and provides training and test datasets.
"""

import argparse
from pathlib import Path
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlflow


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description="Data preparation for model training")

    parser.add_argument("--raw_data", type=str, required=True,
                        help="Path to the raw dataset (CSV file)")
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to directory for saving the training dataset")
    parser.add_argument("--test_data", type=str, required=True,
                        help="Path to directory for saving the test dataset")
    parser.add_argument("--test_train_ratio", type=float, default=0.2,
                        help="Proportion of dataset to include in the test split (default=0.2)")

    args = parser.parse_args()
    return args


def main(args):
    """Read, preprocess, split, and save datasets"""
    print("\n=== DATA PREPARATION STARTED ===")

    # Step 1: Read and preprocess data
    print(f"Reading data from: {args.raw_data}")
    df = pd.read_csv(args.raw_data)

    # Handle missing values (optional cleanup)
    df.dropna(inplace=True)

    # Perform label encoding for categorical columns
    print("Encoding categorical features...")
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Step 2: Split dataset into train and test sets
    print("Splitting data into train and test sets...")
    train_df, test_df = train_test_split(df, test_size=args.test_train_ratio, random_state=42)

    print(f"Training set size: {train_df.shape[0]} rows")
    print(f"Testing set size: {test_df.shape[0]} rows")

    # Step 3: Save datasets as CSV
    os.makedirs(args.train_data, exist_ok=True)
    os.makedirs(args.test_data, exist_ok=True)

    train_output_path = Path(args.train_data) / "train.csv"
    test_output_path = Path(args.test_data) / "test.csv"

    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)

    print(f"Training data saved to: {train_output_path}")
    print(f"Test data saved to: {test_output_path}")

    # Step 4: Log dataset sizes as MLflow metrics
    mlflow.log_metric("train_rows", train_df.shape[0])
    mlflow.log_metric("test_rows", test_df.shape[0])

    print("=== DATA PREPARATION COMPLETED ===\n")


if __name__ == "__main__":
    mlflow.start_run()

    args = parse_args()

    print("\n========= DATA PREPARATION CONFIGURATION =========")
    print(f"Raw data path: {args.raw_data}")
    print(f"Train dataset output path: {args.train_data}")
    print(f"Test dataset output path: {args.test_data}")
    print(f"Test-train ratio: {args.test_train_ratio}")
    print("=================================================\n")

    main(args)

    mlflow.end_run()
