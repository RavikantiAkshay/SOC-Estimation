"""
config.py
=========
Central configuration for the 7-Feature Maximum-Improvement Random Forest model.
Features: 4 baseline observables + Throttle [%] + Motor Torque [Nm] + Velocity [km/h].
All paths, feature definitions, hyperparameters, and benchmarks are defined here.
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

# Output directories
for _dir in [RESULTS_DIR, PLOTS_DIR, MODELS_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ----------------------------------------------
# CSV Parsing
# ----------------------------------------------
CSV_SEPARATOR = ";"
CSV_ENCODING = "latin1"

# ----------------------------------------------
# Feature Definitions (7 Features Total - Maximum Improvement Set)
# ----------------------------------------------
# Baseline 4 features (electrochemistry & thermal environment)
FEATURE_VOLTAGE = "Battery Voltage [V]"
FEATURE_CURRENT = "Battery Current [A]"
FEATURE_BATT_TEMP = "Battery Temperature [°C]"
FEATURE_AMBIENT_TEMP = "Ambient Temperature [°C]"

BASELINE_FEATURES = [
    FEATURE_VOLTAGE,
    FEATURE_CURRENT,
    FEATURE_BATT_TEMP,
    FEATURE_AMBIENT_TEMP,
]

# Additional 3 high-impact powertrain & kinematic features
FEATURE_THROTTLE = "Throttle [%]"
FEATURE_TORQUE = "Motor Torque [Nm]"
FEATURE_VELOCITY = "Velocity [km/h]"

NEW_FEATURES = [
    FEATURE_THROTTLE,
    FEATURE_TORQUE,
    FEATURE_VELOCITY,
]

# Full 7-feature input set
ENHANCED_7_FEATURES = BASELINE_FEATURES + NEW_FEATURES

# Target variable (manufacturer-estimated SOC)
TARGET_SOC = "SoC [%]"

# ----------------------------------------------
# Train / Test Split (Identical 60/10 trip partition)
# ----------------------------------------------
NUM_TRAIN_TRIPS = 60
NUM_TEST_TRIPS = 10

# ----------------------------------------------
# Random Forest Hyperparameters
# ----------------------------------------------
RF_TREE_COUNTS = [25, 50, 75, 100]
NUM_REPEATS = 5
RANDOM_SEED = 42

RF_BASE_PARAMS = {
    "criterion": "squared_error",
    "max_features": "sqrt",  # sqrt(7) = 2.65 -> 2 or 3 features sampled per split
    "min_samples_leaf": 5,   # prevents overfitting to transient sensor noise
    "bootstrap": True,
    "n_jobs": -1,            # multi-threaded CPU parallelization
}

# ----------------------------------------------
# Baseline 4-Feature Benchmarks (For Direct Comparison)
# ----------------------------------------------
BASELINE_4F_BENCHMARKS = {
    "25": {"best_rmse": 5.8876, "best_mae": 4.3736, "best_max": 26.6956, "avg_rmse": 5.9886, "avg_mae": 4.4725},
    "50": {"best_rmse": 5.9166, "best_mae": 4.3922, "best_max": 26.4759, "avg_rmse": 5.9855, "avg_mae": 4.4676},
    "75": {"best_rmse": 5.9095, "best_mae": 4.4134, "best_max": 24.0654, "avg_rmse": 5.9833, "avg_mae": 4.4700},
    "100": {"best_rmse": 5.8979, "best_mae": 4.3853, "best_max": 24.0784, "avg_rmse": 5.9619, "avg_mae": 4.4478},
}
