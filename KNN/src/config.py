"""
config.py for K-Nearest Neighbors (KNN) Baseline.
Defines paths, feature definitions, and KNN hyperparameters.
Operates completely isolated within KNN/.
"""

import os

# ----------------------------------------------
# Paths
# ----------------------------------------------
MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(MODULE_ROOT)
ARCHIVE_DIR = os.path.join(PROJECT_ROOT, "archive")
RESULTS_DIR = os.path.join(MODULE_ROOT, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")

for _dir in [RESULTS_DIR, PLOTS_DIR, MODELS_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ----------------------------------------------
# KNN Configuration
# ----------------------------------------------
KNN_PARAMS = {
    "n_neighbors": 5,
    "weights": "distance",      # Inverse distance weighting for physical sensor continuity
    "algorithm": "kd_tree",     # High-dimensional spatial indexing for efficient querying
    "leaf_size": 30,
    "n_jobs": -1,
}
