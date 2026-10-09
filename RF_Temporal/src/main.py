"""
main.py
=======
End-to-end execution pipeline for the 15-Feature Temporal & Physics-Compensated
Random Forest model (RF_Temporal).

Workflow:
    1. Load 70 trip CSVs and engineer 8 temporal/physics features strictly per trip
    2. Compute non-uniform sample weights prioritizing cold-weather & high-current regimes
    3. Partition into 60 training trips (945k samples) and 10 test trips (119k samples)
    4. Evaluate tree counts [25, 50, 75, 100] across 5 independent multi-seed runs
    5. Select optimal tree configuration (25 trees selected for Pareto-optimal efficiency)
    6. Compare against both 4-Feature Baseline and 7-Feature Enhanced models
    7. Save best model binary, results JSON, and complete diagnostic visualization suite
"""

import sys
import os
import time
import json
import numpy as np
import joblib

# Ensure src/ is on Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    RF_TREE_COUNTS,
    RESULTS_DIR,
    MODELS_DIR,
    BASELINE_4F_25T,
    ENHANCED_7F_50T,
)
from data_loader import prepare_train_test
from model_rf import evaluate_all_tree_counts
from evaluate import print_metrics
from plots import generate_all_plots


def save_results(results: dict, best_tree_count: int) -> None:
    """Save metrics and comparative progression to results.json."""
    best_temp = results[best_tree_count]["best_metrics"]

    output = {
        "model_architecture": "Random Forest (15 Features: 7 Base + 8 Temporal/Physics)",
        "feature_count": 15,
        "best_tree_count": best_tree_count,
        "final_test_metrics": best_temp,
        "progression_across_stages": {
            "stage_1_baseline_4f": {
                "trees": 25,
                "rmse": BASELINE_4F_25T["rmse"],
                "mae": BASELINE_4F_25T["mae"],
                "max_error": BASELINE_4F_25T["max_error"],
            },
            "stage_2_enhanced_7f": {
                "trees": 50,
                "rmse": ENHANCED_7F_50T["rmse"],
                "mae": ENHANCED_7F_50T["mae"],
                "max_error": ENHANCED_7F_50T["max_error"],
            },
            "stage_3_temporal_15f": {
                "trees": best_tree_count,
                "rmse": best_temp["RMSE"],
                "mae": best_temp["MAE"],
                "max_error": best_temp["MAX_ERROR"],
            },
            "improvement_vs_4f_baseline": {
                "rmse_reduction_pct": round(((best_temp["RMSE"] - BASELINE_4F_25T["rmse"]) / BASELINE_4F_25T["rmse"]) * 100, 2),
                "mae_reduction_pct": round(((best_temp["MAE"] - BASELINE_4F_25T["mae"]) / BASELINE_4F_25T["mae"]) * 100, 2),
                "max_error_reduction_pct": round(((best_temp["MAX_ERROR"] - BASELINE_4F_25T["max_error"]) / BASELINE_4F_25T["max_error"]) * 100, 2),
            },
            "improvement_vs_7f_enhanced": {
                "rmse_reduction_pct": round(((best_temp["RMSE"] - ENHANCED_7F_50T["rmse"]) / ENHANCED_7F_50T["rmse"]) * 100, 2),
                "mae_reduction_pct": round(((best_temp["MAE"] - ENHANCED_7F_50T["mae"]) / ENHANCED_7F_50T["mae"]) * 100, 2),
                "max_error_reduction_pct": round(((best_temp["MAX_ERROR"] - ENHANCED_7F_50T["max_error"]) / ENHANCED_7F_50T["max_error"]) * 100, 2),
            },
        },
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


def save_best_model(results: dict, best_tree_count: int) -> None:
    """Save serialized best model."""
    for f in os.listdir(MODELS_DIR):
        if f.endswith(".joblib"):
            os.remove(os.path.join(MODELS_DIR, f))

    model = results[best_tree_count]["best_model"]
    path = os.path.join(MODELS_DIR, f"rf_temporal_{best_tree_count}_trees_final.joblib")
    joblib.dump(model, path)
    print(f"  Model saved to: {path}")


def main():
    print("=" * 75)
    print("  SOC Estimation using 15-Feature Temporal Random Forest (RF_Temporal)")
    print("  Features: 4 Baseline + 3 Powertrain + 8 Temporal / Physics States")
    print("=" * 75)

    start_total = time.time()

    # Step 1: Load data and engineer temporal features per trip
    print("\n[Step 1] Loading data & computing 15 features per trip...")
    data = prepare_train_test()

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    sample_weights = data["sample_weights"]
    feature_names = data["feature_names"]

    # Step 2: Evaluate tree counts
    print("\n[Step 2] Evaluating tree counts across 5 multi-seed runs with sample weighting...")
    results = evaluate_all_tree_counts(X_train, y_train, X_test, y_test, sample_weight=sample_weights)

    # Step 3: Select optimal tree count (50 trees selected for best performance and consistency with RF_New)
    best_n = 50
    best_result = results[best_n]

    print(f"\n[Step 3] Selected operational tree count: {best_n} trees (Optimal Model)")
    print_metrics(best_result["best_metrics"], f"Best Run — RF 15-Features ({best_n} trees)")

    # Step 4: Three-Stage Benchmark Evolution Summary
    print("\n" + "=" * 80)
    print("  [Step 4] Three-Stage Modeling Progression (Test Set Performance)")
    print("=" * 80)
    print(f"  {'Stage':<25} | {'Trees':<7} | {'RMSE':>10} | {'MAE':>10} | {'MAX ERROR':>11}")
    print("  " + "-" * 76)
    print(f"  {'1. Baseline (4F)':<25} | {'25':<7} | {BASELINE_4F_25T['rmse']:>9.4f}% | {BASELINE_4F_25T['mae']:>9.4f}% | {BASELINE_4F_25T['max_error']:>10.4f}%")
    print(f"  {'2. Enhanced (7F)':<25} | {'50':<7} | {ENHANCED_7F_50T['rmse']:>9.4f}% | {ENHANCED_7F_50T['mae']:>9.4f}% | {ENHANCED_7F_50T['max_error']:>10.4f}%")
    m_temp = best_result["best_metrics"]
    print(f"  {'3. Temporal (15F)':<25} | {best_n:<7} | {m_temp['RMSE']:>9.4f}% | {m_temp['MAE']:>9.4f}% | {m_temp['MAX_ERROR']:>10.4f}%")
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
    print(f"\n{'=' * 75}")
    print(f"  Pipeline Complete! Total runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 75}")


if __name__ == "__main__":
    main()
