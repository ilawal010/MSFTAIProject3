# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Registers the best-trained ML model from the sweep job.
"""

import argparse
from pathlib import Path
import mlflow
import mlflow.sklearn
import json
import os


def parse_args():
    """Parse input arguments for model registration."""
    parser = argparse.ArgumentParser(description="Register the best-trained ML model in MLflow Model Registry")

    parser.add_argument('--model_name', type=str, required=True,
                        help='Name under which the model will be registered in MLflow')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the directory containing the trained model (e.g., ./outputs/model)')
    parser.add_argument('--model_info_output_path', type=str, required=True,
                        help='Path to save model registration info as JSON (e.g., ./outputs/model_info.json)')

    args, _ = parser.parse_known_args()
    print(f"Arguments received: {args}")
    return args


def main(args):
    """Load, log, and register the model in MLflow."""
    print(f"\n=== MODEL REGISTRATION STARTED FOR: {args.model_name} ===")

    # Step 1: Load the model from the specified path
    print(f"Loading model from path: {args.model_path}")
    model = mlflow.sklearn.load_model(args.model_path)

    # Step 2: Log the loaded model in MLflow for versioning/tracking
    print("Logging model to MLflow...")
    mlflow.sklearn.log_model(sk_model=model, artifact_path="model", registered_model_name=args.model_name)

    # Retrieve model URI for registration
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"

    # Step 3: Register the logged model under the specified model name
    print("Registering model in MLflow registry...")
    registered_model = mlflow.register_model(model_uri=model_uri, name=args.model_name)

    # Step 4: Save registration info (name and version) to JSON
    model_info = {
        "model_name": args.model_name,
        "model_version": registered_model.version,
        "run_id": mlflow.active_run().info.run_id,
        "model_uri": model_uri
    }

    os.makedirs(Path(args.model_info_output_path).parent, exist_ok=True)
    with open(args.model_info_output_path, "w") as f:
        json.dump(model_info, f, indent=4)

    print(f"Model registered successfully as '{args.model_name}' (Version {registered_model.version})")
    print(f"Model info saved to: {args.model_info_output_path}")
    print("=== MODEL REGISTRATION COMPLETED ===\n")


if __name__ == "__main__":
    mlflow.start_run()

    args = parse_args()

    print("\n========= MODEL REGISTRATION CONFIGURATION =========")
    print(f"Model name: {args.model_name}")
    print(f"Model path: {args.model_path}")
    print(f"Model info output path: {args.model_info_output_path}")
    print("====================================================\n")

    main(args)

    mlflow.end_run()
