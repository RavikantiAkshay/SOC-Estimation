"""
data_loader.py
==============
Data loading, preprocessing, temporal feature engineering, and train/test
partitioning for the 15-feature Temporal & Physics-Compensated Random Forest model.

Features engineered strictly PER TRIP to avoid temporal data leakage:
  1. Battery Voltage [V]
  2. Battery Current [A]
  3. Battery Temperature [°C]
  4. Ambient Temperature [°C]
  5. Throttle [%]
  6. Motor Torque [Nm]
  7. Velocity [km/h]
  8. dV_dt [V/s]          - Instantaneous voltage derivative
  9. dI_dt [A/s]          - Instantaneous current derivative
 10. V_mean_15s [V]       - 15s rolling average voltage (local noise filter)
 11. I_mean_15s [A]       - 15s rolling average current
 12. V_std_15s [V]        - 15s rolling voltage standard deviation
 13. V_mean_60s [V]       - 60s rolling average voltage (macro baseline anchor)
 14. I_mean_60s [A]       - 60s rolling average current
 15. V_est_ocv [V]        - Arrhenius temperature-compensated Open-Circuit Voltage [V - I * R(T)]
"""

import os
import glob
import numpy as np
import pandas as pd

from config import (
    ARCHIVE_DIR,
    CSV_SEPARATOR,
    CSV_ENCODING,
    BASELINE_FEATURES,
    POWERTRAIN_FEATURES,
    ALL_15_FEATURES,
    TARGET_SOC,
    TIME_COL,
    NUM_TRAIN_TRIPS,
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
    FEATURE_VOLTAGE,
    FEATURE_CURRENT,
    FEATURE_BATT_TEMP,
    FEATURE_AMBIENT_TEMP,
)


def _clean_column_name(col: str) -> str:
    """Standardize raw CSV headers."""
    col = col.strip()
    col = col.replace("[%)", "[%]")
    col = col.replace("]]]", "]")
    return col


def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column headers and drop any unnamed columns."""
    df.columns = [_clean_column_name(c) for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df


def load_single_trip(filepath: str) -> pd.DataFrame:
    """Load one trip CSV, clean headers, and return DataFrame."""
    df = pd.read_csv(filepath, sep=CSV_SEPARATOR, encoding=CSV_ENCODING)
    df = _standardise_columns(df)
    return df


def get_sorted_trip_files() -> list[str]:
    """Return all 70 CSV file paths in sorted order."""
    files = sorted(glob.glob(os.path.join(ARCHIVE_DIR, "Trip*.csv")))
    if len(files) != 70:
        raise FileNotFoundError(
            f"Expected 70 trip CSV files in {ARCHIVE_DIR}, found {len(files)}"
        )
    return files


def load_all_trips() -> tuple[list[pd.DataFrame], list[str]]:
    """Load all 70 trip files without row dropping."""
    files = get_sorted_trip_files()
    trip_dfs = []
    trip_names = []

    for fpath in files:
        df = load_single_trip(fpath)
        name = os.path.splitext(os.path.basename(fpath))[0]
        trip_dfs.append(df)
        trip_names.append(name)

    return trip_dfs, trip_names


def _find_matching_column(df: pd.DataFrame, target_name: str) -> str:
    """Match column name handling degree symbols and casing."""
    def ascii_skeleton(s: str) -> str:
        return s.encode("ascii", "ignore").decode("ascii").strip().lower()

    target_skel = ascii_skeleton(target_name)

    # 1. Exact skeleton match
    for col in df.columns:
        if ascii_skeleton(col) == target_skel:
            return col

    # 2. Substring match
    base_target = target_name.split("[")[0].strip().lower()
    for col in df.columns:
        if base_target in col.lower():
            if "max" not in col.lower() and "min" not in col.lower() and "displayed" not in col.lower():
                return col

    raise KeyError(f"Could not find column matching '{target_name}' in columns: {list(df.columns)}")


def engineer_trip_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract 7 base features, time, and target, clean decimals, and compute
    the 8 temporal and physics-compensated features strictly within this single trip.
    """
    raw_7 = BASELINE_FEATURES + POWERTRAIN_FEATURES
    resolved_7 = [_find_matching_column(df, f) for f in raw_7]
    resolved_target = _find_matching_column(df, TARGET_SOC)
    resolved_time = _find_matching_column(df, TIME_COL)

    cols_needed = resolved_7 + [resolved_target, resolved_time]
    subset = df[cols_needed].copy()

    # Standardize column names
    rename_map = {old: new for old, new in zip(resolved_7, raw_7)}
    rename_map[resolved_target] = TARGET_SOC
    rename_map[resolved_time] = TIME_COL
    subset.rename(columns=rename_map, inplace=True)

    # Numeric conversion
    for col in subset.columns:
        if subset[col].dtype == object:
            subset[col] = subset[col].astype(str).str.replace(",", ".")
        subset[col] = pd.to_numeric(subset[col], errors="coerce")

    # Drop NaNs before rolling computations
    subset.dropna(inplace=True)
    subset.reset_index(drop=True, inplace=True)

    # Compute time delta (seconds)
    dt = subset[TIME_COL].diff().fillna(1.0)
    dt = dt.replace(0, 1.0)  # avoid division by zero

    # 1. Derivatives (clipped to avoid single-timestep sensor noise spikes)
    dV = subset[FEATURE_VOLTAGE].diff().fillna(0.0)
    dI = subset[FEATURE_CURRENT].diff().fillna(0.0)
    subset[FEATURE_DV_DT] = (dV / dt).clip(-20.0, 20.0)
    subset[FEATURE_DI_DT] = (dI / dt).clip(-100.0, 100.0)

    # 2. Local 15-second rolling statistics (filters momentary pedal blips)
    subset[FEATURE_V_MEAN_15S] = subset[FEATURE_VOLTAGE].rolling(window=15, min_periods=1).mean()
    subset[FEATURE_I_MEAN_15S] = subset[FEATURE_CURRENT].rolling(window=15, min_periods=1).mean()
    subset[FEATURE_V_STD_15S] = subset[FEATURE_VOLTAGE].rolling(window=15, min_periods=1).std().fillna(0.0)

    # 3. Macro 60-second rolling statistics (retains true baseline resting potential during hard acceleration)
    subset[FEATURE_V_MEAN_60S] = subset[FEATURE_VOLTAGE].rolling(window=60, min_periods=1).mean()
    subset[FEATURE_I_MEAN_60S] = subset[FEATURE_CURRENT].rolling(window=60, min_periods=1).mean()

    # 4. Ultra-Macro 180-second rolling statistics (3-minute baseline anchor that resists 60s load pulls)
    subset[FEATURE_V_MEAN_180S] = subset[FEATURE_VOLTAGE].rolling(window=180, min_periods=1).mean()

    # 5. Dynamic voltage sag relative to 60s macro baseline
    subset[FEATURE_V_SAG_60S] = subset[FEATURE_VOLTAGE] - subset[FEATURE_V_MEAN_60S]

    # 6. Instantaneous electrical tractive power (kW)
    subset[FEATURE_POWER] = (subset[FEATURE_VOLTAGE] * subset[FEATURE_CURRENT]) / 1000.0

    # 7. Physics-informed calibrated Arrhenius electro-thermal resistance compensation
    # Empirical R_0: 0.055 Ohm at 25°C, scaling with slope 0.060 for realistic sub-zero impedance
    r_ohmic = 0.055 * np.exp(-0.060 * (subset[FEATURE_BATT_TEMP] - 25.0))
    r_pol = 0.040 * np.exp(-0.045 * (subset[FEATURE_BATT_TEMP] - 25.0))
    subset[FEATURE_V_EST_OCV] = subset[FEATURE_VOLTAGE] - (subset[FEATURE_CURRENT] * r_ohmic)
    subset[FEATURE_V_EST_OCV_FULL] = subset[FEATURE_VOLTAGE] - (subset[FEATURE_CURRENT] * r_ohmic) - (subset[FEATURE_I_MEAN_15S] * r_pol)

    return subset


def compute_sample_weights(train_df: pd.DataFrame) -> np.ndarray:
    """
    Compute non-uniform sample weights prioritizing extreme electro-thermal operating regimes:
      - Cold battery electrolyte (Batt Temp <= 5°C): 2.0x weight
      - Heavy discharge transient (Current < -80 A): 2.0x weight
      - Severe acceleration voltage-collapse outliers (Current < -100 A and Voltage < 335 V): 3.0x weight
    """
    weights = np.ones(len(train_df), dtype=np.float32)
    weights[train_df[FEATURE_BATT_TEMP] <= 5.0] *= 2.0
    weights[train_df[FEATURE_CURRENT] < -80.0] *= 2.0
    outlier_mask = (train_df[FEATURE_CURRENT] < -100.0) & (train_df[FEATURE_VOLTAGE] < 335.0)
    weights[outlier_mask] *= 3.0

    return weights


def prepare_train_test() -> dict:
    """
    Process all 70 trips and partition into 60 train and 10 test trips.
    """
    trip_dfs, trip_names = load_all_trips()

    train_frames = []
    test_frames = []
    train_sizes = []
    test_sizes = []

    for i, (df, name) in enumerate(zip(trip_dfs, trip_names)):
        clean = engineer_trip_features(df)

        if i < NUM_TRAIN_TRIPS:
            train_frames.append(clean)
            train_sizes.append(len(clean))
        else:
            test_frames.append(clean)
            test_sizes.append(len(clean))

    train_df = pd.concat(train_frames, ignore_index=True)
    test_df = pd.concat(test_frames, ignore_index=True)

    X_train = train_df[ALL_15_FEATURES].values
    y_train = train_df[TARGET_SOC].values
    X_test = test_df[ALL_15_FEATURES].values
    y_test = test_df[TARGET_SOC].values

    sample_weights = compute_sample_weights(train_df)

    print(f"Training set  : {X_train.shape[0]:>10,} instances from {NUM_TRAIN_TRIPS} trips")
    print(f"Testing set   : {X_test.shape[0]:>10,} instances from {len(trip_names) - NUM_TRAIN_TRIPS} trips")
    print(f"Features (15) : {ALL_15_FEATURES}")
    print(f"Target        : {TARGET_SOC}")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "sample_weights": sample_weights,
        "feature_names": ALL_15_FEATURES,
        "train_trips": trip_names[:NUM_TRAIN_TRIPS],
        "test_trips": trip_names[NUM_TRAIN_TRIPS:],
        "train_sizes": train_sizes,
        "test_sizes": test_sizes,
    }


if __name__ == "__main__":
    data = prepare_train_test()
    print(f"X_train shape: {data['X_train'].shape}")
    print(f"X_test shape : {data['X_test'].shape}")
