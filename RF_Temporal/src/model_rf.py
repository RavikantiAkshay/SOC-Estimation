"""
model_rf.py
===========
Multi-run Random Forest training and tree count evaluation for the
15-feature Temporal & Physics-Compensated SOC estimation model (RF_Temporal).
"""

import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from config import (
    RF_TREE_COUNTS,
    RF_BASE_PARAMS,
    NUM_REPEATS,
    RANDOM_SEED,
)
from evaluate import compute_all_metrics


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int,
    random_state: int,
    sample_weight: np.ndarray = None,
) -> tuple:
    """
    Train a 15-feature Random Forest model with sample weighting on the 60-trip
    training set and evaluate on the 10-trip unseen test set.
    """
    params = {**RF_BASE_PARAMS}
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        **params,
    )

    start = time.time()
    model.fit(X_train, y_train, sample_weight=sample_weight)
    train_time = time.time() - start

    y_pred = model.predict(X_test)
    metrics = compute_all_metrics(y_test, y_pred)

    return model, y_pred, metrics, train_time


def repeated_train_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int = 25,
    n_repeats: int = NUM_REPEATS,
    sample_weight: np.ndarray = None,
) -> dict:
    """
    Train and evaluate n_repeats times with distinct seeds to assess
    stability and extract both best-run and average metrics.
    """
    print(f"\n  --- {n_estimators} trees x {n_repeats} runs (15 Features) ---")

    all_metrics = []
    all_predictions = []
    all_models = []
    total_time = 0.0

    for run in range(n_repeats):
        seed = RANDOM_SEED + run * 100
        model, y_pred, metrics, elapsed = train_and_evaluate(
            X_train, y_train, X_test, y_test, n_estimators, seed, sample_weight=sample_weight
        )
        all_metrics.append(metrics)
        all_predictions.append(y_pred)
        all_models.append(model)
        total_time += elapsed

        print(
            f"    Run {run + 1}/{n_repeats}  "
            f"RMSE={metrics['RMSE']:.4f}%  "
            f"MAE={metrics['MAE']:.4f}%  "
            f"MAX={metrics['MAX_ERROR']:.4f}%  "
            f"({elapsed:.1f}s)"
        )

    # Best run = lowest RMSE on unseen test set
    best_idx = int(np.argmin([m["RMSE"] for m in all_metrics]))
    best_metrics = all_metrics[best_idx]

    # Average across runs
    avg_metrics = {}
    for key in all_metrics[0]:
        avg_metrics[key] = float(np.mean([m[key] for m in all_metrics]))

    print(f"    >> Best  (run #{best_idx + 1}):  RMSE = {best_metrics['RMSE']:.4f}% | MAX = {best_metrics['MAX_ERROR']:.4f}%")
    print(f"    >> Average:              RMSE = {avg_metrics['RMSE']:.4f}% | MAX = {avg_metrics['MAX_ERROR']:.4f}%")
    print(f"    >> Total time: {total_time:.1f}s")

    return {
        "n_estimators": n_estimators,
        "all_run_metrics": all_metrics,
        "best_metrics": best_metrics,
        "avg_metrics": avg_metrics,
        "best_model": all_models[best_idx],
        "best_predictions": all_predictions[best_idx],
        "best_run_index": best_idx,
        "total_time": total_time,
    }


def evaluate_all_tree_counts(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    tree_counts: list[int] = RF_TREE_COUNTS,
    n_repeats: int = NUM_REPEATS,
    sample_weight: np.ndarray = None,
) -> dict:
    """Evaluate all configured tree sizes across multi-seed runs."""
    results = {}
    for n_trees in tree_counts:
        result = repeated_train_evaluate(
            X_train, y_train, X_test, y_test,
            n_estimators=n_trees, n_repeats=n_repeats,
            sample_weight=sample_weight,
        )
        results[n_trees] = result

    # Print summary table
    print(f"\n{'=' * 75}")
    print(f"{'15-FEATURE TEMPORAL RANDOM FOREST TREE COUNT SUMMARY':^75}")
    print(f"{'=' * 75}")
    print(f"  {'Trees':<7} | {'Best RMSE':>11} | {'Avg RMSE':>11} | {'Best MAE':>11} | {'Best MAX':>11}")
    print("  " + "-" * 65)
    for n_trees in tree_counts:
        b_m = results[n_trees]["best_metrics"]
        a_m = results[n_trees]["avg_metrics"]
        print(
            f"  {n_trees:<7} | "
            f"{b_m['RMSE']:>10.4f}% | "
            f"{a_m['RMSE']:>10.4f}% | "
            f"{b_m['MAE']:>10.4f}% | "
            f"{b_m['MAX_ERROR']:>10.4f}%"
        )
    print("=" * 75)

    return results
