"""
config.py
=========
Central configuration for the 15-Feature Temporal & Dynamic Random Forest model (RF_Temporal).

Feature Architecture:
  - 4 Baseline Observables (Electrochemistry & Thermal)
  - 3 Powertrain & Kinematics (Throttle, Torque, Velocity)
  - 8 Temporal, Dynamic & Physics-Compensated Features:
      * dV_dt: Instantaneous voltage rate-of-change
      * dI_dt: Instantaneous current rate-of-change
      * V_mean_15s: 15-second rolling average of Pack Voltage (local smoother)
      * I_mean_15s: 15-second rolling average of Pack Current
      * V_std_15s: 15-second rolling standard deviation of Pack Voltage
      * V_mean_60s: 60-second rolling average of Pack Voltage (macro baseline anchor)
      * I_mean_60s: 60-second rolling average of Pack Current
      * V_est_ocv: Temperature-compensated Open-Circuit Voltage estimate [V - I * R(T)]
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
# Feature Definitions (15 Features Total)
# ----------------------------------------------
# 1. Baseline 4 Features
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

# 2. Powertrain & Kinematics (from RF_New)
FEATURE_THROTTLE = "Throttle [%]"
FEATURE_TORQUE = "Motor Torque [Nm]"
FEATURE_VELOCITY = "Velocity [km/h]"

POWERTRAIN_FEATURES = [
    FEATURE_THROTTLE,
    FEATURE_TORQUE,
    FEATURE_VELOCITY,
]

# 3. Temporal, Dynamic & Physics-Compensated Features
FEATURE_DV_DT = "dV_dt [V/s]"
FEATURE_DI_DT = "dI_dt [A/s]"
FEATURE_V_MEAN_15S = "V_mean_15s [V]"
FEATURE_I_MEAN_15S = "I_mean_15s [A]"
FEATURE_V_STD_15S = "V_std_15s [V]"
FEATURE_V_MEAN_60S = "V_mean_60s [V]"
FEATURE_I_MEAN_60S = "I_mean_60s [A]"
FEATURE_V_MEAN_180S = "V_mean_180s [V]"
FEATURE_V_SAG_60S = "V_sag_60s [V]"
FEATURE_POWER = "Power [kW]"
FEATURE_V_EST_OCV = "V_est_ocv [V]"
FEATURE_V_EST_OCV_FULL = "V_est_ocv_full [V]"

TEMPORAL_FEATURES = [
    FEATURE_DV_DT,
    FEATURE_DI_DT,
    FEATURE_V_MEAN_15S,
    FEATURE_I_MEAN_15S,
    FEATURE_V_STD_15S,
    FEATURE_V_MEAN_60S,
    FEATURE_I_MEAN_60S,
    FEATURE_V_MEAN_180S,
    FEATURE_V_SAG_60S,
    FEATURE_POWER,
    FEATURE_V_EST_OCV,
    FEATURE_V_EST_OCV_FULL,
]

# Full 19-feature input matrix
ALL_15_FEATURES = BASELINE_FEATURES + POWERTRAIN_FEATURES + TEMPORAL_FEATURES
ALL_FEATURES = ALL_15_FEATURES  # Alias for consistency
RF_FEATURE_NAMES = ALL_FEATURES

# Target variable (manufacturer-estimated SOC)
TARGET_SOC = "SoC [%]"
TIME_COL = "Time [s]"

# ----------------------------------------------
# Train / Test Partitioning Protocol
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
    "max_features": "sqrt",  # sqrt(15) = 3.87 -> 4 features per split
    "min_samples_leaf": 5,   # prevents overfitting on noise
    "bootstrap": True,
    "n_jobs": -1,
}

# ----------------------------------------------
# Benchmark References for Comparative Tracking
# ----------------------------------------------
BASELINE_4F_25T = {"rmse": 5.8876, "mae": 4.3736, "max_error": 26.6956}
ENHANCED_7F_50T = {"rmse": 4.5734, "mae": 3.3158, "max_error": 27.0113}
