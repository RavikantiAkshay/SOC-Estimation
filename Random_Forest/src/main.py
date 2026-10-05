"""
main.py
=======
End-to-end orchestration for the SOC estimation project.

Workflow (mirrors the paper step by step):
    1. Load and preprocess all 70 trip CSVs
    2. Split into training (trips 1-60) and testing (trips 61-70)
    3. For each tree count (25, 50, 75, 100):
       - Train on full training set 5 times (different seeds)
       - Evaluate each on the test set
       - Record best-of-5 and average-of-5 metrics
    4. Select best tree count (lowest best RMSE)
    5. Compare all results against the paper's Table 2
    6. Save the best model per tree count, results JSON, and all plots

Usage:
    python -u src/main.py
"""

import sys
import os
import time
import json
import numpy as np
import joblib

# Ensure src/ is on the import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    RF_TREE_COUNTS,
    RESULTS_DIR,
    MODELS_DIR,
)
from data_loader import prepare_train_test
from model_rf import evaluate_all_tree_counts
from evaluate import print_metrics
from plots import generate_all_plots


# ----------------------------------------------
# Save results
# ----------------------------------------------
def save_results(results: dict, best_tree_count: int) -> None:
    """Save all tree-count metrics to results.json."""
    output = {
        "best_tree_count": best_tree_count,
        "final_test_metrics": results[best_tree_count]["best_metrics"],
        "tree_comparison": {},
    }

    for n_trees, result in results.items():
        output["tree_comparison"][str(n_trees)] = {
            "best_metrics": result["best_metrics"],
            "avg_metrics": result["avg_metrics"],
            "all_runs": result["all_run_metrics"],
            "best_run_index": result["best_run_index"],
            "total_time_seconds": result["total_time"],
        }

    path = os.path.join(RESULTS_DIR, "results.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to: {path}")


# ----------------------------------------------
# Save models
# ----------------------------------------------
def save_best_model(results: dict, best_tree_count: int) -> None:
    """Save only the final best model (to avoid multi-GB disk usage)."""
    # Clean out old model files first
    for f in os.listdir(MODELS_DIR):
        if f.endswith(".joblib"):
            os.remove(os.path.join(MODELS_DIR, f))

    model = results[best_tree_count]["best_model"]
    path = os.path.join(MODELS_DIR, f"rf_{best_tree_count}_trees_final.joblib")
    joblib.dump(model, path)
    print(f"  Model saved to: {path}")


# ----------------------------------------------
# Main pipeline
# ----------------------------------------------
def main():
    print("=" * 60)
    print("  SOC Estimation using Random Forest")
    print("  Replicating: Sulaiman & Mustaffa, 2024")
    print("=" * 60)

    start_total = time.time()

    # -- Step 1: Load data --
    print("\n[Step 1] Loading and preprocessing data...")
    data = prepare_train_test()

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    feature_names = data["feature_names"]

    # -- Step 2: Evaluate all tree counts (5 runs each on TEST set) --
    print("\n[Step 2] Evaluating tree counts (5 runs each, evaluated on test set)...")
    results = evaluate_all_tree_counts(X_train, y_train, X_test, y_test)

    # -- Step 3: Select best tree count --
    best_n = min(
        results.keys(),
        key=lambda n: results[n]["best_metrics"]["RMSE"],
    )
    best_result = results[best_n]

    print(f"[Step 3] Best tree count from our runs: {best_n} trees")
    print_metrics(best_result["best_metrics"], f"Best Run - RF ({best_n} trees)")

    # -- Step 4: Detailed comparison with paper for ALL tree counts --
    print("\n" + "=" * 80)
    print("  [Step 4] Replication Comparison with Paper (Sulaiman & Mustaffa, Table 2)")
    print("=" * 80)

    paper_best = {
        25:  {"RMSE": 5.9028, "MAE": 4.4321, "MAX_ERROR": 24.2175, "STD_DEV": 5.8999},
        50:  {"RMSE": 5.9256, "MAE": 4.4152, "MAX_ERROR": 24.5568, "STD_DEV": 5.9231},
        75:  {"RMSE": 5.9949, "MAE": 4.4788, "MAX_ERROR": 24.6938, "STD_DEV": 5.9948},
        100: {"RMSE": 5.9547, "MAE": 4.4767, "MAX_ERROR": 25.7958, "STD_DEV": 5.9545},
    }
    paper_avg = {
        25:  {"RMSE": 6.0622, "MAE": 4.5129, "MAX_ERROR": 25.9245, "STD_DEV": 6.0616},
        50:  {"RMSE": 5.9992, "MAE": 4.5106, "MAX_ERROR": 24.2992, "STD_DEV": 5.9982},
        75:  {"RMSE": 6.0226, "MAE": 4.5090, "MAX_ERROR": 25.1596, "STD_DEV": 6.0216},
        100: {"RMSE": 5.9895, "MAE": 4.4894, "MAX_ERROR": 24.7869, "STD_DEV": 5.9892},
    }

    print(f"\n  {'Trees':<7} | {'Metric':<9} | {'Paper Best':>10} {'Ours Best':>10} {'Diff':>8} | {'Paper Avg':>10} {'Ours Avg':>10} {'Diff':>8}")
    print("  " + "-" * 76)

    for n_trees in RF_TREE_COUNTS:
        best_m = results[n_trees]["best_metrics"]
        avg_m = results[n_trees]["avg_metrics"]
        p_best = paper_best[n_trees]
        p_avg = paper_avg[n_trees]

        for i, metric in enumerate(["RMSE", "MAE", "MAX_ERROR", "STD_DEV"]):
            tree_label = f"{n_trees}" if i == 0 else ""
            d_best = best_m[metric] - p_best[metric]
            d_avg = avg_m[metric] - p_avg[metric]
            s_best = "+" if d_best > 0 else ""
            s_avg = "+" if d_avg > 0 else ""
            metric_label = "MAX" if metric == "MAX_ERROR" else ("STD" if metric == "STD_DEV" else metric)
            print(
                f"  {tree_label:<7} | {metric_label:<9} | "
                f"{p_best[metric]:>10.4f} {best_m[metric]:>10.4f} {s_best}{d_best:>7.4f} | "
                f"{p_avg[metric]:>10.4f} {avg_m[metric]:>10.4f} {s_avg}{d_avg:>7.4f}"
            )
        print("  " + "-" * 76)

    # -- Step 5: Save everything --
    print("[Step 5] Saving model, results, and plots...")

    save_best_model(results, best_n)
    save_results(results, best_n)

    # Build cv_results dict for the tree-count comparison plot
    cv_results_for_plot = {}
    for n_trees, r in results.items():
        cv_results_for_plot[n_trees] = {
            "best_metrics": r["best_metrics"],
            "avg_metrics": r["avg_metrics"],
        }

    generate_all_plots(
        y_test,
        best_result["best_predictions"],
        best_result["best_model"],
        feature_names,
        cv_results_for_plot,
        best_n,
        test_trips=data["test_trips"],
        test_sizes=data["test_sizes"],
    )

    elapsed = time.time() - start_total
    print(f"\n{'=' * 60}")
    print(f"  Complete!  Total time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
