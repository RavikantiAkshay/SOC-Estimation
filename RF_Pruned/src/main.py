"""
main.py
=======
Master pipeline for Stage 4: Random Forest Depth Pruning & Pareto Complexity Optimization.
Runs the depth sweep (if not already completed), evaluates all candidate depths,
serializes the Pareto-optimal model, and generates the complete publication visualization suite.
"""

import os
import sys
import json
import time
import numpy as np
import joblib

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_ROOT = os.path.dirname(SRC_DIR)
PROJECT_ROOT = os.path.dirname(MODULE_ROOT)
TEMPORAL_SRC = os.path.join(PROJECT_ROOT, "RF_Temporal", "src")

for p in [SRC_DIR, TEMPORAL_SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

from prune_config import (
    RESULTS_DIR,
    PLOTS_DIR,
    MODELS_DIR,
)
from prune_experiment import run_pruning_sweep
from plots_pruned import (
    plot_pareto_frontier,
    plot_depth_sweep_comparison,
    plot_four_stage_evolution,
    plot_soc_estimation_pruned,
)


def main():
    results_json_path = os.path.join(RESULTS_DIR, "results.json")
    if os.path.exists(results_json_path):
        print(f"Loading existing pruning sweep results from: {results_json_path}")
        with open(results_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = data["depth_sweep_results"]
        best_pareto_key = str(data["pareto_optimal_selected"]["depth"])
        four_stages = data["four_stage_evolution"]

        # Load y_test and predictions for plotting
        from data_loader import prepare_train_test
        telemetry = prepare_train_test()
        y_test = telemetry["y_test"]

        best_model_path = os.path.join(MODELS_DIR, f"rf_pruned_depth_{best_pareto_key}_final.joblib")
        best_model = joblib.load(best_model_path)
        y_pred = best_model.predict(telemetry["X_test"])
    else:
        exp_output = run_pruning_sweep()
        results = exp_output["results"]
        best_pareto_key = exp_output["best_pareto_key"]
        y_test = exp_output["y_test"]
        y_pred = exp_output["preds_dict"][best_pareto_key]
        with open(results_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        four_stages = data["four_stage_evolution"]

    print("\n[Step 2] Generating publication-quality Pareto diagnostic plots...")
    plot_pareto_frontier(
        results=results,
        best_depth_key=best_pareto_key,
        output_path=os.path.join(PLOTS_DIR, "pareto_frontier_tradeoff.png"),
    )
    plot_depth_sweep_comparison(
        results=results,
        best_depth_key=best_pareto_key,
        output_path=os.path.join(PLOTS_DIR, "depth_sweep_comparison.png"),
    )
    plot_four_stage_evolution(
        four_stages=four_stages,
        output_path=os.path.join(PLOTS_DIR, "four_stage_evolution.png"),
    )
    plot_soc_estimation_pruned(
        y_test=y_test,
        y_pred=y_pred,
        output_path=os.path.join(PLOTS_DIR, "soc_estimation_pruned.png"),
    )

    print("\n" + "=" * 80)
    print("  STAGE 4 COMPLETE: ALL PLOTS & PARETO METRICS SERIALIZED")
    print("=" * 80)


if __name__ == "__main__":
    main()

