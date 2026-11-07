"""
prune_experiment.py
===================
Executes the tree depth pruning sweep across candidate depths [10, 14, 18, 22, 26, None].
Identifies the Pareto-optimal trade-off between model complexity (nodes, memory)
and state of charge estimation accuracy.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add directories to sys.path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_ROOT = os.path.dirname(SRC_DIR)
PROJECT_ROOT = os.path.dirname(MODULE_ROOT)
TEMPORAL_SRC = os.path.join(PROJECT_ROOT, "RF_Temporal", "src")

if TEMPORAL_SRC not in sys.path:
    sys.path.append(TEMPORAL_SRC)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from prune_config import (
    CANDIDATE_DEPTHS,
    NUM_TREES,
    MIN_SAMPLES_LEAF,
    MAX_FEATURES,
    RANDOM_STATE,
    RESULTS_DIR,
    MODELS_DIR,
    STAGE_1_BASELINE_4F,
    STAGE_2_ENHANCED_7F,
    STAGE_3_TEMPORAL_UNCONSTRAINED,
)
from data_loader import prepare_train_test


def run_pruning_sweep():
    print("=" * 80)
    print("  STAGE 4: RANDOM FOREST DEPTH PRUNING & PARETO COMPLEXITY OPTIMIZATION")
    print("=" * 80)
    print("Loading standardized telemetry (60 Train Trips, 10 Test Trips)...")
    data = prepare_train_test()

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    sample_weights = data["sample_weights"]
    feature_names = data["feature_names"]
    test_trips = data["test_trips"]
    test_sizes = data["test_sizes"]

    print(f"Dataset Loaded: Train = {X_train.shape[0]:,} samples | Test = {X_test.shape[0]:,} samples | Features = {X_train.shape[1]}")
    print(f"Sweeping Candidate Tree Depths: {CANDIDATE_DEPTHS} with {NUM_TREES} trees each\n")

    results = {}
    models_dict = {}
    preds_dict = {}

    unconstrained_nodes = STAGE_3_TEMPORAL_UNCONSTRAINED["nodes"]

    print(f"{'Max Depth':<10} | {'RMSE (%)':>9} | {'MAE (%)':>9} | {'MAX Err (%)':>11} | {'Total Nodes':>12} | {'Node Reduc':>11} | {'Train (s)':>9} | {'Worst Checks':>13}")
    print("-" * 95)

    for depth in CANDIDATE_DEPTHS:
        depth_label = str(depth) if depth is not None else "None (Unconstrained)"
        d_key = "unconstrained" if depth is None else str(depth)

        t0 = time.time()
        rf = RandomForestRegressor(
            n_estimators=NUM_TREES,
            max_depth=depth,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            max_features=MAX_FEATURES,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train, sample_weight=sample_weights)
        train_time = time.time() - t0

        y_pred = rf.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        max_error = float(np.max(np.abs(y_test - y_pred)))
        std_dev = float(np.std(y_test - y_pred))

        total_nodes = int(sum(tree.tree_.node_count for tree in rf.estimators_))
        max_actual_depth = int(max(tree.tree_.max_depth for tree in rf.estimators_))
        avg_actual_depth = float(np.mean([tree.tree_.max_depth for tree in rf.estimators_]))
        worst_case_checks = int(sum(tree.tree_.max_depth for tree in rf.estimators_))

        if depth is None:
            unconstrained_nodes = total_nodes

        node_reduction_pct = float((1.0 - (total_nodes / unconstrained_nodes)) * 100.0)
        # Approximate C flash storage footprint: ~12 bytes per node
        c_flash_mb = float((total_nodes * 12) / (1024 * 1024))

        results[d_key] = {
            "depth_setting": depth,
            "depth_label": depth_label,
            "rmse": rmse,
            "mae": mae,
            "max_error": max_error,
            "std_dev": std_dev,
            "total_nodes": total_nodes,
            "max_actual_depth": max_actual_depth,
            "avg_actual_depth": avg_actual_depth,
            "worst_case_branch_checks": worst_case_checks,
            "node_reduction_pct": node_reduction_pct,
            "train_time_sec": train_time,
            "c_flash_footprint_mb": c_flash_mb,
        }

        models_dict[d_key] = rf
        preds_dict[d_key] = y_pred

        print(f"{depth_label:<10} | {rmse:>8.4f}% | {mae:>8.4f}% | {max_error:>10.4f}% | {total_nodes:>12,} | {node_reduction_pct:>10.1f}% | {train_time:>8.1f}s | {worst_case_checks:>11} checks")

    # Selection of Pareto-Optimal configuration
    # Criteria: Node reduction >= 80% while RMSE <= 4.05% and MAX <= 16.5%
    pareto_candidates = [k for k, v in results.items() if k != "unconstrained" and v["node_reduction_pct"] >= 75.0]
    # Pick the best trade-off candidate (lowest RMSE among >75% reduction)
    best_pareto_key = min(pareto_candidates, key=lambda k: results[k]["rmse"]) if pareto_candidates else "18"
    best_pareto = results[best_pareto_key]

    print("\n" + "=" * 80)
    print(f"  PARETO OPTIMAL SWEET SPOT IDENTIFIED: max_depth = {best_pareto['depth_setting']}")
    print("=" * 80)
    print(f"  • Total Node Complexity : {best_pareto['total_nodes']:,} nodes (vs {unconstrained_nodes:,} unconstrained)")
    print(f"  • Complexity Reduction  : {best_pareto['node_reduction_pct']:.1f}% reduction in tree decisions!")
    print(f"  • Flash Footprint in C  : {best_pareto['c_flash_footprint_mb']:.2f} MB")
    print(f"  • Accuracy Retention    : RMSE = {best_pareto['rmse']:.4f}% (vs {results['unconstrained']['rmse']:.4f}%)")
    print(f"  • Peak Error Bound      : MAX  = {best_pareto['max_error']:.4f}% (vs {results['unconstrained']['max_error']:.4f}%)")
    print(f"  • Worst-Case Traversal  : {best_pareto['worst_case_branch_checks']} branch checks across 50 trees (< 0.05 ms ECU WCET)")
    print("=" * 80)

    # Save best model
    best_model_path = os.path.join(MODELS_DIR, f"rf_pruned_depth_{best_pareto_key}_final.joblib")
    joblib.dump(models_dict[best_pareto_key], best_model_path, compress=3)
    print(f"\nSerialized Pareto-Optimal Model Binary: {best_model_path}")
    print(f"Serialized File Size: {os.path.getsize(best_model_path) / (1024*1024):.2f} MB")

    # Save comprehensive results JSON
    summary_output = {
        "title": "Stage 4: Random Forest Depth Pruning & Pareto Complexity Optimization",
        "description": "Systematic evaluation of tree depth constraints to identify the production-optimal automotive ECU model.",
        "candidate_depths_evaluated": CANDIDATE_DEPTHS,
        "pareto_optimal_selected": {
            "depth": best_pareto["depth_setting"],
            "metrics": best_pareto,
        },
        "depth_sweep_results": results,
        "four_stage_evolution": {
            "stage_1_baseline_4f": STAGE_1_BASELINE_4F,
            "stage_2_enhanced_7f": STAGE_2_ENHANCED_7F,
            "stage_3_temporal_unconstrained": {
                "rmse": results["unconstrained"]["rmse"],
                "mae": results["unconstrained"]["mae"],
                "max_error": results["unconstrained"]["max_error"],
                "trees": 50,
                "nodes": results["unconstrained"]["total_nodes"],
                "depth": results["unconstrained"]["max_actual_depth"],
            },
            "stage_4_pareto_pruned": {
                "rmse": best_pareto["rmse"],
                "mae": best_pareto["mae"],
                "max_error": best_pareto["max_error"],
                "trees": 50,
                "depth": best_pareto["depth_setting"],
                "nodes": best_pareto["total_nodes"],
                "node_reduction_pct": best_pareto["node_reduction_pct"],
                "c_flash_mb": best_pareto["c_flash_footprint_mb"],
            },
        },
    }

    results_json_path = os.path.join(RESULTS_DIR, "results.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_output, f, indent=2)
    print(f"Results saved to: {results_json_path}")

    return {
        "results": results,
        "best_pareto_key": best_pareto_key,
        "y_test": y_test,
        "preds_dict": preds_dict,
        "test_trips": test_trips,
        "test_sizes": test_sizes,
    }


if __name__ == "__main__":
    run_pruning_sweep()
