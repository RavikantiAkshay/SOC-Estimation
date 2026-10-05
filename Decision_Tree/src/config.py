"""
config.py for Decision Tree Baseline.
Defines filesystem paths, feature definitions, and model parameters.
Operates completely isolated within Decision_Tree/.
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
# Decision Tree Configuration
# ----------------------------------------------
# Using criterion and min_samples_leaf matching Random Forest's base learners
DT_BASE_PARAMS = {
    "criterion": "squared_error",
    "min_samples_leaf": 5,
}

# 5 seeds for evaluating single-tree variance and stability
DT_SEEDS = [42, 142, 242, 342, 442]
