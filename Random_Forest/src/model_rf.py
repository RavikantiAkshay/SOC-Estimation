"""
model_rf.py
===========
Random Forest training and evaluation -- replicating the paper's methodology.

Paper procedure (Table 2):
    For each tree count in [25, 50, 75, 100]:
        Run 5 independent times (different random seeds):
            - Train RF on the FULL training set (trips 1-60)
            - Evaluate on the TEST set (trips 61-70)
        Report "best run" and "average across runs" test-set metrics.
    Select the tree count with lowest best RMSE on the test set.
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
from evaluate import compute_all_metrics, print_metrics


# ----------------------------------------------
# Core: train once, evaluate on test set
# ----------------------------------------------
def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int,
    random_state: int,
) -> tuple:
    """
    Train a single RF model on the full training set and
    evaluate on the test set.

    Returns
    -------
    (model, y_pred, metrics, train_time)
    """
    params = {**RF_BASE_PARAMS}
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        **params,
    )

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)
    metrics = compute_all_metrics(y_test, y_pred)

    return model, y_pred, metrics, train_time


# ----------------------------------------------
# Repeated train+test for one tree count
# (Paper: 5 independent runs per tree count)
# ----------------------------------------------
def repeated_train_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int = 25,
    n_repeats: int = NUM_REPEATS,
) -> dict:
    """
    Train on full training set and evaluate on test set,
    repeated n_repeats times with different random seeds.

    Returns dict with best/avg metrics, best model, and best predictions.
    """
    print(f"\n  --- {n_estimators} trees x {n_repeats} runs ---")

    all_metrics = []
    all_predictions = []
    all_models = []
    total_time = 0.0

    for run in range(n_repeats):
        seed = RANDOM_SEED + run * 100
        model, y_pred, metrics, elapsed = train_and_evaluate(
            X_train, y_train, X_test, y_test, n_estimators, seed
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

    # Best run = lowest RMSE on test set
    best_idx = int(np.argmin([m["RMSE"] for m in all_metrics]))
    best_metrics = all_metrics[best_idx]

    # Average across all runs
    avg_metrics = {}
    for key in all_metrics[0]:
        avg_metrics[key] = float(np.mean([m[key] for m in all_metrics]))

    print(f"    >> Best  (run #{best_idx + 1}):  RMSE = {best_metrics['RMSE']:.4f}%")
    print(f"    >> Average:              RMSE = {avg_metrics['RMSE']:.4f}%")
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


# ----------------------------------------------
# Evaluate all tree counts (Paper Table 2)
# ----------------------------------------------
def evaluate_all_tree_counts(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    tree_counts: list[int] = RF_TREE_COUNTS,
    n_repeats: int = NUM_REPEATS,
) -> dict:
    """
    Run repeated_train_evaluate for every tree count.

    Returns dict mapping n_estimators -> result dict.
    """
    results = {}

    for n_trees in tree_counts:
        result = repeated_train_evaluate(
            X_train, y_train, X_test, y_test,
            n_estimators=n_trees, n_repeats=n_repeats,
        )
        results[n_trees] = result

    # ---- Summary table ----
    print(f"\n{'=' * 85}")
    print(f"{'TREE COUNT COMPARISON  (Paper Table 2 Replication)':^85}")
    print(f"{'=' * 85}")
    print(
        f"  {'Trees':<8} {'Best RMSE':>10} {'Avg RMSE':>10} "
        f"{'Best MAE':>10} {'Avg MAE':>10} "
        f"{'Best MAX':>10} {'Time (s)':>10}"
    )
    print(f"  {'-' * 75}")

    for n_trees in tree_counts:
        r = results[n_trees]
        print(
            f"  {n_trees:<8} "
            f"{r['best_metrics']['RMSE']:>10.4f} "
            f"{r['avg_metrics']['RMSE']:>10.4f} "
            f"{r['best_metrics']['MAE']:>10.4f} "
            f"{r['avg_metrics']['MAE']:>10.4f} "
            f"{r['best_metrics']['MAX_ERROR']:>10.4f} "
            f"{r['total_time']:>10.1f}"
        )

    # Paper reference row
    print(f"  {'-' * 75}")
    print(f"  {'Paper':>8}")
    paper_best = {
        25:  (5.9028, 6.0622, 4.4321, 4.5129, 24.2175),
        50:  (5.9256, 5.9992, 4.4152, 4.5106, 24.5568),
        75:  (5.9949, 6.0226, 4.4788, 4.5090, 24.6938),
        100: (5.9547, 5.9895, 4.4767, 4.4894, 25.7958),
    }
    for n_trees in tree_counts:
        b_rmse, a_rmse, b_mae, a_mae, b_max = paper_best[n_trees]
        print(
            f"  {n_trees:<8} "
            f"{b_rmse:>10.4f} "
            f"{a_rmse:>10.4f} "
            f"{b_mae:>10.4f} "
            f"{a_mae:>10.4f} "
            f"{b_max:>10.4f} "
            f"{'(paper)':>10}"
        )

    print(f"{'=' * 85}\n")
    return results
