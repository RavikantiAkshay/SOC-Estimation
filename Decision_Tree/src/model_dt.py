"""
model_dt.py
===========
Implementation and multi-seed evaluation of Single Decision Tree Regressor for SOC estimation.
Demonstrates the performance, variance, and feature importance of a single tree baseline.
"""

import sys
import time
from pathlib import Path
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import joblib

# Add project root to sys.path to access Random_Forest data loading and metric functions
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Random_Forest" / "src"))

from Random_Forest.src.evaluate import compute_all_metrics, print_metrics
from Decision_Tree.src.config import DT_BASE_PARAMS, DT_SEEDS


def train_single_tree(X_train, y_train, random_state=42):
    """Instantiates and trains a single Decision Tree Regressor."""
    model = DecisionTreeRegressor(
        criterion=DT_BASE_PARAMS["criterion"],
        min_samples_leaf=DT_BASE_PARAMS["min_samples_leaf"],
        random_state=random_state,
    )
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    return model, train_time


def evaluate_decision_tree_multi_seed(X_train, y_train, X_test, y_test, feature_names):
    """
    Evaluates Decision Tree across 5 independent seeds (matching the 5 runs used for RF).
    Calculates best-run metrics, average metrics, and variance.
    """
    all_runs = []
    models = []
    predictions_list = []
    train_times = []

    print(f"\n--- Training Single Decision Tree across {len(DT_SEEDS)} Seeds ---")

    for idx, seed in enumerate(DT_SEEDS):
        model, t_time = train_single_tree(X_train, y_train, random_state=seed)
        preds = model.predict(X_test)
        run_metrics = compute_all_metrics(y_test, preds)

        all_runs.append(run_metrics)
        models.append(model)
        predictions_list.append(preds)
        train_times.append(t_time)

        print(
            f"  Run {idx + 1}/5 (Seed {seed:>3}): "
            f"RMSE = {run_metrics['RMSE']:.4f}%, "
            f"MAE = {run_metrics['MAE']:.4f}%, "
            f"MAX = {run_metrics['MAX_ERROR']:.4f}% "
            f"({t_time:.2f}s)"
        )

    # Find best run based on lowest RMSE
    rmses = [r["RMSE"] for r in all_runs]
    best_idx = int(np.argmin(rmses))
    best_metrics = all_runs[best_idx]
    best_model = models[best_idx]
    best_predictions = predictions_list[best_idx]

    # Compute average metrics across seeds
    avg_metrics = {}
    for metric_key in ["RMSE", "MAE", "MAX_ERROR", "STD_DEV"]:
        avg_metrics[metric_key] = float(np.mean([r[metric_key] for r in all_runs]))

    # Tree complexity statistics
    tree_stats = {
        "max_depth": int(best_model.tree_.max_depth),
        "node_count": int(best_model.tree_.node_count),
        "leaf_count": int(best_model.tree_.n_leaves),
        "best_seed": DT_SEEDS[best_idx],
    }

    # Feature importances
    feature_importances = {
        name: float(imp) for name, imp in zip(feature_names, best_model.feature_importances_)
    }

    print("\n--- Best Decision Tree Architecture ---")
    print(f"  Maximum Depth:  {tree_stats['max_depth']}")
    print(f"  Total Nodes:    {tree_stats['node_count']:,}")
    print(f"  Terminal Leaves:{tree_stats['leaf_count']:,}")
    print(f"  Average Train Time: {np.mean(train_times):.2f}s")

    print_metrics(best_metrics, label=f"Best Decision Tree (Seed {DT_SEEDS[best_idx]})")

    return {
        "best_model": best_model,
        "best_predictions": best_predictions,
        "best_metrics": best_metrics,
        "avg_metrics": avg_metrics,
        "all_runs": all_runs,
        "tree_stats": tree_stats,
        "feature_importances": feature_importances,
        "total_train_time_seconds": float(np.sum(train_times)),
        "avg_train_time_seconds": float(np.mean(train_times)),
    }


def save_dt_model(model, output_path):
    """Saves serialized model to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"\nDecision Tree model saved to: {output_path}")
