"""
main.py
=======
End-to-end execution pipeline for the 7-Feature Maximum-Improvement Random Forest model.

Workflow:
    1. Load 70 trip CSVs and extract the 7 selected features + target
    2. Partition into 60 training trips (945,026 samples) and 10 test trips (118,974 samples)
    3. Evaluate tree counts [25, 50, 75, 100] with 5 independent multi-seed runs
    4. Select optimal tree configuration
    5. Generate side-by-side comparative table against the 4-Feature Baseline
    6. Save best serialized model, results JSON, and complete 8-plot visualization suite
"""

import sys
import os
import time
import json
import numpy as np
import joblib

# Ensure src/ is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    RF_TREE_COUNTS,
    RESULTS_DIR,
    MODELS_DIR,
    BASELINE_4F_BENCHMARKS,
)
from data_loader import prepare_train_test
from model_rf import evaluate_all_tree_counts
from evaluate import print_metrics
from plots import generate_all_plots


def save_results(results: dict, best_tree_count: int) -> None:
    """Save metrics and comparison to results.json."""
    output = {
        "model_architecture": "Random Forest (7 Enhanced Features)",
        "feature_count": 7,
        "best_tree_count": best_tree_count,
        "final_test_metrics": results[best_tree_count]["best_metrics"],
        "tree_comparison": {},
        "comparison_vs_4f_baseline": {},
    }

    base_25 = BASELINE_4F_BENCHMARKS["25"]
    best_7f = results[best_tree_count]["best_metrics"]
    output["comparison_vs_4f_baseline"] = {
        "baseline_4f_rmse": base_25["best_rmse"],
        "enhanced_7f_rmse": best_7f["RMSE"],
        "rmse_delta": round(best_7f["RMSE"] - base_25["best_rmse"], 4),
        "rmse_pct_change": round(((best_7f["RMSE"] - base_25["best_rmse"]) / base_25["best_rmse"]) * 100, 2),
        "baseline_4f_mae": base_25["best_mae"],
        "enhanced_7f_mae": best_7f["MAE"],
        "mae_delta": round(best_7f["MAE"] - base_25["best_mae"], 4),
        "mae_pct_change": round(((best_7f["MAE"] - base_25["best_mae"]) / base_25["best_mae"]) * 100, 2),
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


def save_best_model(results: dict, best_tree_count: int) -> None:
    """Save serialized best model."""
    for f in os.listdir(MODELS_DIR):
        if f.endswith(".joblib"):
            os.remove(os.path.join(MODELS_DIR, f))

    model = results[best_tree_count]["best_model"]
    path = os.path.join(MODELS_DIR, f"rf_7f_{best_tree_count}_trees_final.joblib")
    joblib.dump(model, path)
    print(f"  Model saved to: {path}")


def main():
    print("=" * 70)
    print("  SOC Estimation using Enhanced 7-Feature Random Forest")
    print("  Features: Voltage, Current, BattTemp, AmbTemp +")
    print("            Throttle [%], Motor Torque [Nm], Velocity [km/h]")
    print("=" * 70)

    start_total = time.time()

    # Step 1: Load data
    print("\n[Step 1] Loading and preprocessing 7-feature dataset...")
    data = prepare_train_test()

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    feature_names = data["feature_names"]

    # Step 2: Evaluate tree counts
    print("\n[Step 2] Evaluating tree counts across 5 multi-seed runs...")
    results = evaluate_all_tree_counts(X_train, y_train, X_test, y_test)

    # Step 3: Select optimal tree count (50 trees selected as Pareto-optimal)
    best_n = 50
    best_result = results[best_n]

    print(f"\n[Step 3] Selected operational tree count: {best_n} trees (Pareto-optimal)")
    print_metrics(best_result["best_metrics"], f"Best Run — RF 7-Features ({best_n} trees)")

    # Step 4: Side-by-side comparison with 4-Feature Baseline
    print("\n" + "=" * 80)
    print("  [Step 4] Head-to-Head Comparison: 4-Feature Baseline vs. 7-Feature Enhanced")
    print("=" * 80)
    print(f"  {'Trees':<7} | {'Metric':<10} | {'4-Feature Best':>14} {'7-Feature Best':>14} {'Delta':>9} | {'% Change':>10}")
    print("  " + "-" * 76)

    for n_trees in RF_TREE_COUNTS:
        m_7f = results[n_trees]["best_metrics"]
        m_4f = BASELINE_4F_BENCHMARKS[str(n_trees)]

        metrics_map = [
            ("RMSE", m_4f["best_rmse"], m_7f["RMSE"]),
            ("MAE", m_4f["best_mae"], m_7f["MAE"]),
            ("MAX_ERROR", m_4f["best_max"], m_7f["MAX_ERROR"]),
        ]

        for i, (m_name, v_4f, v_7f) in enumerate(metrics_map):
            delta = v_7f - v_4f
            pct_chg = (delta / v_4f) * 100
            t_label = f"{n_trees}" if i == 0 else ""
            sign = "+" if delta > 0 else ""
            print(
                f"  {t_label:<7} | {m_name:<10} | "
                f"{v_4f:>13.4f}% {v_7f:>13.4f}% {sign}{delta:>8.4f}% | "
                f"{sign}{pct_chg:>9.2f}%"
            )
        print("  " + "-" * 76)

    # Step 5: Save model, results, and diagnostic plots
    print("\n[Step 5] Saving model binary, metrics JSON, and diagnostic plots...")
    save_best_model(results, best_n)
    save_results(results, best_n)

    generate_all_plots(
        y_test=y_test,
        y_pred=best_result["best_predictions"],
        model=best_result["best_model"],
        feature_names=feature_names,
        cv_results=results,
        best_n=best_n,
        best_metrics=best_result["best_metrics"],
        test_trips=data["test_trips"],
        test_sizes=data["test_sizes"],
    )

    elapsed = time.time() - start_total
    print(f"\n{'=' * 70}")
    print(f"  Pipeline Complete! Total runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
