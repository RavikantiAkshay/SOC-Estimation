"""
model_knn.py
============
Implementation and evaluation of K-Nearest Neighbors (KNN) Regressor for SOC estimation.
Demonstrates instance-based distance learning, inference latency, and comparison with Random Forest.
"""

import sys
import time
from pathlib import Path
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Random_Forest.src.evaluate import compute_all_metrics, print_metrics
from KNN.src.config import KNN_PARAMS


def train_and_evaluate_knn(X_train, y_train, X_test, y_test, feature_names):
    """
    Fits StandardScaler and KNeighborsRegressor on training data and evaluates on test set.
    """
    print(f"\n--- Training KNN Regressor (k={KNN_PARAMS['n_neighbors']}, weights='{KNN_PARAMS['weights']}') ---")

    # Step 1: Feature Scaling (Mandatory for distance-based algorithms)
    print("  Scaling features with StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 2: Fit KNN model (Builds KD-Tree index)
    knn = KNeighborsRegressor(
        n_neighbors=KNN_PARAMS["n_neighbors"],
        weights=KNN_PARAMS["weights"],
        algorithm=KNN_PARAMS["algorithm"],
        leaf_size=KNN_PARAMS["leaf_size"],
        n_jobs=KNN_PARAMS["n_jobs"],
    )

    start_fit = time.time()
    knn.fit(X_train_scaled, y_train)
    fit_time = time.time() - start_fit
    print(f"  KD-Tree index built in {fit_time:.4f} seconds.")

    # Step 3: Run Inference on Test Set
    print(f"  Running inference across {len(X_test):,} test instances...")
    start_pred = time.time()
    y_pred = knn.predict(X_test_scaled)
    pred_time = time.time() - start_pred

    latency_per_sample_ms = (pred_time / len(X_test)) * 1000.0
    print(f"  Total inference time: {pred_time:.2f} seconds ({latency_per_sample_ms:.4f} ms per sample).")

    # Step 4: Compute Metrics
    metrics = compute_all_metrics(y_test, y_pred)
    print_metrics(metrics, label=f"KNN Regressor (k={KNN_PARAMS['n_neighbors']})")

    # Package model and scaler together for deployment
    pipeline_artifact = {
        "scaler": scaler,
        "knn_model": knn,
        "feature_names": feature_names,
        "k": KNN_PARAMS["n_neighbors"],
    }

    return {
        "pipeline_artifact": pipeline_artifact,
        "predictions": y_pred,
        "metrics": metrics,
        "fit_time_seconds": fit_time,
        "inference_time_seconds": pred_time,
        "latency_per_sample_ms": latency_per_sample_ms,
    }


def save_knn_model(pipeline_artifact, output_path):
    """Saves serialized scaler + model dictionary to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline_artifact, output_path, compress=3)
    print(f"\nKNN model artifact saved to: {output_path}")
