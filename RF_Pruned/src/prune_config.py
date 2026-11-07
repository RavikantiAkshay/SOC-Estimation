"""
config.py
=========
Configuration for Stage 4: Random Forest Depth Pruning & Pareto Complexity Optimization (RF_Pruned).

Investigates tree depth constraints to find the optimal trade-off between
model complexity (node count, storage size, RAM usage) and SOC estimation accuracy.
"""

import os
import sys

# Paths
MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(MODULE_ROOT)
RESULTS_DIR = os.path.join(MODULE_ROOT, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")

for _d in [RESULTS_DIR, PLOTS_DIR, MODELS_DIR]:
    os.makedirs(_d, exist_ok=True)

# Depth sweep configuration
CANDIDATE_DEPTHS = [10, 14, 18, 22, 26, None]
NUM_TREES = 50
MIN_SAMPLES_LEAF = 5
MAX_FEATURES = "sqrt"
RANDOM_STATE = 42

# Historical baselines for reference
STAGE_1_BASELINE_4F = {"rmse": 5.8876, "mae": 4.3736, "max_error": 26.6956, "trees": 25, "nodes": 3752787, "size_mb": 257.7}
STAGE_2_ENHANCED_7F = {"rmse": 4.5734, "mae": 3.3158, "max_error": 27.0113, "trees": 50, "nodes": 7720000, "size_mb": 558.8}
STAGE_3_TEMPORAL_UNCONSTRAINED = {"rmse": 3.8500, "mae": 2.9666, "max_error": 15.5543, "trees": 50, "nodes": 7532820, "size_mb": 517.3}
