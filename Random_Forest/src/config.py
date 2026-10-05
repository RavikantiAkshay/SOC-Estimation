"""
config.py
=========
Central configuration for the SOC Estimation project.
All paths, feature names, hyperparameters, and split definitions
are kept here so every other module imports from one place.
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

# Create output directories if they don't exist
for _dir in [RESULTS_DIR, PLOTS_DIR, MODELS_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ----------------------------------------------
# CSV Parsing
# ----------------------------------------------
CSV_SEPARATOR = ";"
CSV_ENCODING = "latin1"

# ----------------------------------------------
# Column Name Mapping
# ----------------------------------------------
# Raw column names as they appear in the CSVs (with quirks like
# trailing spaces, mismatched brackets, etc.).  We map them to
# clean, standardised names used throughout the codebase.

# The 4 input features used in the paper
FEATURE_VOLTAGE = "Battery Voltage [V]"
FEATURE_CURRENT = "Battery Current [A]"
FEATURE_BATT_TEMP = "Battery Temperature [°C]"    # raw has °C via latin1
FEATURE_AMBIENT_TEMP = "Ambient Temperature [°C]"

# Target variable (manufacturer-estimated SOC, NOT displayed SOC)
TARGET_SOC = "SoC [%]"

# Paper's exact 4 input features (order matters for consistency)
PAPER_FEATURES = [
    FEATURE_VOLTAGE,
    FEATURE_CURRENT,
    FEATURE_BATT_TEMP,
    FEATURE_AMBIENT_TEMP,
]

# ----------------------------------------------
# Train / Test Split  (Paper: Trips 1-60 train, 61-70 test)
# ----------------------------------------------
# Files are sorted alphabetically: TripA01..TripA32, TripB01..TripB38
# That gives 70 files total.  First 60 -> train, last 10 -> test.
NUM_TRAIN_TRIPS = 60
NUM_TEST_TRIPS = 10

# ----------------------------------------------
# Random Forest Hyperparameters
# ----------------------------------------------
# Tree counts evaluated in the paper
RF_TREE_COUNTS = [25, 50, 75, 100]

# Best tree count found in the paper
RF_BEST_TREES = 25

# Cross-validation
K_FOLDS = 5           # k-fold cross-validation
NUM_REPEATS = 5       # number of independent CV runs to average

# MATLAB TreeBagger defaults used in the paper:
# min_samples_leaf=5 (MATLAB MinLeafSize for regression), max_features='sqrt' (decorrelates trees)
RF_BASE_PARAMS = {
    "criterion": "squared_error",
    "max_features": "sqrt",     # sqrt(4) = 2 features sampled per split
    "min_samples_leaf": 5,      # MATLAB default MinLeafSize for regression
    "bootstrap": True,
    "n_jobs": -1,               # use all CPU cores
}

# Random seed for reproducibility across repeated runs
RANDOM_SEED = 42
