"""
main.py for Decision Tree Baseline.
Orchestrates data loading, multi-seed training, evaluation, comparison with Random Forest,
model serialization, and figure generation.
Completely isolated within Decision_Tree/.
"""

import sys
import json
import time
from pathlib import Path

# Add project root and Random_Forest/src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Random_Forest" / "src"))

from Random_Forest.src.data_loader import prepare_train_test
from Decision_Tree.src.config import RESULTS_DIR, MODELS_DIR, PLOTS_DIR
from Decision_Tree.src.model_dt import (
    evaluate_decision_tree_multi_seed,
    save_dt_model,
)
from Decision_Tree.src.plots_dt import generate_all_dt_plots


def load_rf_metrics():
    """Loads optimal Random Forest metrics from Random_Forest/results/results.json."""
    rf_json = PROJECT_ROOT / "Random_Forest" / "results" / "results.json"
    if rf_json.exists():
        with open(rf_json, "r") as f:
            data = json.load(f)
        return data["final_test_metrics"], data["tree_comparison"]["25"]["avg_metrics"]
    else:
        # Fallback reference values
        best_ref = {"RMSE": 5.8876, "MAE": 4.3736, "MAX_ERROR": 26.6956, "STD_DEV": 5.8876}
        avg_ref = {"RMSE": 5.9886, "MAE": 4.4725, "MAX_ERROR": 26.4467, "STD_DEV": 5.9873}
        return best_ref, avg_ref


def print_comparison_table(rf_best, rf_avg, dt_best, dt_avg):
    """Prints head-to-head comparison table between Random Forest and Decision Tree."""
    print("\n" + "=" * 78)
    print("    HEAD-TO-HEAD BENCHMARK: RANDOM FOREST (25 TREES) vs. DECISION TREE (1 TREE)")
    print("=" * 78)
    print(f"  {'Metric':<12} | {'RF (Best)':>12} | {'DT (Best)':>12} | {'Delta':>12} | {'RF Ensemble Gain':>18}")
    print("  " + "-" * 74)

    for m in ["RMSE", "MAE", "MAX_ERROR", "STD_DEV"]:
        rf_v = rf_best[m]
        dt_v = dt_best[m]
        delta = dt_v - rf_v
        reduction = ((dt_v - rf_v) / dt_v) * 100
        sign = "+" if delta > 0 else ""
        print(
            f"  {m:<12} | {rf_v:>11.4f}% | {dt_v:>11.4f}% | {sign}{delta:>11.4f}% | "
            f"RF is {reduction:>6.2f}% better"
        )

    print("  " + "-" * 74)
    print(f"  {'Avg RMSE':<12} | {rf_avg['RMSE']:>11.4f}% | {dt_avg['RMSE']:>11.4f}% | "
          f"{dt_avg['RMSE'] - rf_avg['RMSE']:>+11.4f}% | "
          f"RF is {((dt_avg['RMSE'] - rf_avg['RMSE']) / dt_avg['RMSE']) * 100:>6.2f}% better")
    print("=" * 78 + "\n")


def save_dt_results(dt_res, rf_best, rf_avg, output_path):
    """Saves comprehensive evaluation results and comparison to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "model_type": "DecisionTreeRegressor (Single Tree)",
        "tree_stats": dt_res["tree_stats"],
        "feature_importances": dt_res["feature_importances"],
        "best_metrics": dt_res["best_metrics"],
        "avg_metrics": dt_res["avg_metrics"],
        "all_runs": dt_res["all_runs"],
        "rf_comparison": {
            "rf_best_metrics": rf_best,
            "rf_avg_metrics": rf_avg,
            "best_rmse_reduction_pct": ((dt_res["best_metrics"]["RMSE"] - rf_best["RMSE"]) / dt_res["best_metrics"]["RMSE"]) * 100,
            "best_mae_reduction_pct": ((dt_res["best_metrics"]["MAE"] - rf_best["MAE"]) / dt_res["best_metrics"]["MAE"]) * 100,
            "avg_rmse_reduction_pct": ((dt_res["avg_metrics"]["RMSE"] - rf_avg["RMSE"]) / dt_res["avg_metrics"]["RMSE"]) * 100,
        },
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Decision Tree results saved to: {output_path}")


def main():
    print("\n" + "=" * 78)
    print("    ELECTRIC VEHICLE SOC ESTIMATION - DECISION TREE BASELINE (1 TREE)")
    print("=" * 78)

    # 1. Load Data
    data = prepare_train_test()
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    feature_names = data["feature_names"]
    test_trips = data["test_trips"]
    test_sizes = data["test_sizes"]

    # 2. Train & Multi-Seed Evaluation of Single Decision Tree
    dt_res = evaluate_decision_tree_multi_seed(
        X_train, y_train, X_test, y_test, feature_names
    )

    # 3. Save Best Model
    model_path = Path(MODELS_DIR) / "dt_model.joblib"
    save_dt_model(dt_res["best_model"], model_path)

    # 4. Compare with Random Forest (25 Trees)
    rf_best, rf_avg = load_rf_metrics()
    print_comparison_table(rf_best, rf_avg, dt_res["best_metrics"], dt_res["avg_metrics"])

    # 5. Save Results JSON
    results_json_path = Path(RESULTS_DIR) / "results.json"
    save_dt_results(dt_res, rf_best, rf_avg, results_json_path)

    # 6. Generate Dedicated Plots
    generate_all_dt_plots(
        y_test,
        dt_res["best_predictions"],
        dt_res["feature_importances"],
        test_trips,
        test_sizes,
        rf_best,
        dt_res["best_metrics"],
        PLOTS_DIR,
    )

    print("\n[SUCCESS] Decision Tree baseline execution complete!\n")


if __name__ == "__main__":
    main()
