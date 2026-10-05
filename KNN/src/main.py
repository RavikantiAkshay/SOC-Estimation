"""
main.py for K-Nearest Neighbors (KNN) Baseline.
Orchestrates data loading, feature scaling, model fitting, evaluation,
comparison with Random Forest and Decision Tree, artifact serialization, and plot generation.
Completely isolated within KNN/.
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
from KNN.src.config import RESULTS_DIR, MODELS_DIR, PLOTS_DIR, KNN_PARAMS
from KNN.src.model_knn import (
    train_and_evaluate_knn,
    save_knn_model,
)
from KNN.src.plots_knn import generate_all_knn_plots


def load_rf_metrics():
    """Loads optimal Random Forest metrics from Random_Forest/results/results.json."""
    rf_json = PROJECT_ROOT / "Random_Forest" / "results" / "results.json"
    if rf_json.exists():
        with open(rf_json, "r") as f:
            data = json.load(f)
        return data["final_test_metrics"], data["tree_comparison"]["25"]["avg_metrics"]
    else:
        best_ref = {"RMSE": 5.8876, "MAE": 4.3736, "MAX_ERROR": 26.6956, "STD_DEV": 5.8876}
        avg_ref = {"RMSE": 5.9886, "MAE": 4.4725, "MAX_ERROR": 26.4467, "STD_DEV": 5.9873}
        return best_ref, avg_ref


def load_dt_metrics():
    """Loads optimal Decision Tree metrics from Decision_Tree/results/results.json if present."""
    dt_json = PROJECT_ROOT / "Decision_Tree" / "results" / "results.json"
    if dt_json.exists():
        with open(dt_json, "r") as f:
            data = json.load(f)
        return data.get("best_metrics", None)
    return None


def print_comparison_table(rf_metrics, knn_metrics, dt_metrics=None):
    """Prints clear head-to-head comparison table across models."""
    print("\n" + "=" * 88)
    print("    COMPREHENSIVE BENCHMARK: RANDOM FOREST vs. KNN (k=5) vs. DECISION TREE (1 Tree)")
    print("=" * 88)
    header = f"  {'Metric':<12} | {'RF (25 Trees)':>14} | {'KNN (k=5)':>14} | {'RF vs KNN Gain':>18}"
    if dt_metrics:
        header += f" | {'DT (1 Tree)':>14}"
    print(header)
    print("  " + "-" * (84 if not dt_metrics else 100))

    for m in ["RMSE", "MAE", "MAX_ERROR", "STD_DEV"]:
        rf_v = rf_metrics[m]
        knn_v = knn_metrics[m]
        gain = ((knn_v - rf_v) / knn_v) * 100
        line = f"  {m:<12} | {rf_v:>13.4f}% | {knn_v:>13.4f}% | RF is {gain:>6.2f}% better"
        if dt_metrics:
            dt_v = dt_metrics.get(m, 0.0)
            line += f" | {dt_v:>13.4f}%"
        print(line)

    print("=" * 88 + "\n")


def save_knn_results(knn_res, rf_metrics, dt_metrics, output_path):
    """Saves comprehensive evaluation results, inference latency, and comparison to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "model_type": "KNeighborsRegressor (Distance-Weighted, KD-Tree)",
        "knn_params": KNN_PARAMS,
        "metrics": knn_res["metrics"],
        "timing": {
            "fit_time_seconds": knn_res["fit_time_seconds"],
            "inference_time_seconds": knn_res["inference_time_seconds"],
            "latency_per_sample_ms": knn_res["latency_per_sample_ms"],
        },
        "rf_comparison": {
            "rf_metrics": rf_metrics,
            "rmse_reduction_pct": ((knn_res["metrics"]["RMSE"] - rf_metrics["RMSE"]) / knn_res["metrics"]["RMSE"]) * 100,
            "mae_reduction_pct": ((knn_res["metrics"]["MAE"] - rf_metrics["MAE"]) / knn_res["metrics"]["MAE"]) * 100,
            "max_error_reduction_pct": ((knn_res["metrics"]["MAX_ERROR"] - rf_metrics["MAX_ERROR"]) / knn_res["metrics"]["MAX_ERROR"]) * 100,
        },
    }
    if dt_metrics:
        summary["dt_comparison"] = {
            "dt_metrics": dt_metrics,
        }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"KNN results saved to: {output_path}")


def main():
    print("\n" + "=" * 80)
    print("    ELECTRIC VEHICLE SOC ESTIMATION - K-NEAREST NEIGHBORS (KNN) BASELINE")
    print("=" * 80)

    # 1. Load Data
    data = prepare_train_test()
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    feature_names = data["feature_names"]
    test_trips = data["test_trips"]
    test_sizes = data["test_sizes"]

    # 2. Train & Evaluate KNN
    knn_res = train_and_evaluate_knn(
        X_train, y_train, X_test, y_test, feature_names
    )

    # 3. Save Model Artifact (Scaler + Model)
    model_path = Path(MODELS_DIR) / "knn_model.joblib"
    save_knn_model(knn_res["pipeline_artifact"], model_path)

    # 4. Compare with Random Forest & Decision Tree
    rf_best, _ = load_rf_metrics()
    dt_metrics = load_dt_metrics()
    print_comparison_table(rf_best, knn_res["metrics"], dt_metrics)

    # 5. Save Results JSON
    results_json_path = Path(RESULTS_DIR) / "results.json"
    save_knn_results(knn_res, rf_best, dt_metrics, results_json_path)

    # 6. Generate Dedicated Plots
    generate_all_knn_plots(
        y_test,
        knn_res["predictions"],
        test_trips,
        test_sizes,
        rf_best,
        knn_res["metrics"],
        PLOTS_DIR,
    )

    print("\n[SUCCESS] KNN baseline execution complete!\n")


if __name__ == "__main__":
    main()
