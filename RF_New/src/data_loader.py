"""
data_loader.py
==============
Data loading, cleaning, and train/test partitioning for the 7-feature
maximum-improvement Random Forest model.

Dataset specification:
    - 70 CSV files (semicolon-delimited, Latin-1 encoded)
    - Trips 1-60 -> training  (945,026 clean instances)
    - Trips 61-70 -> testing  (118,974 clean instances)
    - 7 input features:
        1. Battery Voltage [V]
        2. Battery Current [A]
        3. Battery Temperature [°C]
        4. Ambient Temperature [°C]
        5. Throttle [%]
        6. Motor Torque [Nm]
        7. Velocity [km/h]
    - 1 target: SoC [%] (manufacturer-estimated ground truth)
"""

import os
import glob
import numpy as np
import pandas as pd

from config import (
    ARCHIVE_DIR,
    CSV_SEPARATOR,
    CSV_ENCODING,
    ENHANCED_7_FEATURES,
    TARGET_SOC,
    NUM_TRAIN_TRIPS,
)


def _clean_column_name(col: str) -> str:
    """
    Standardise raw CSV headers:
      - Strip leading and trailing whitespace
      - Fix mismatched brackets: 'max. SoC [%)' -> 'max. SoC [%]'
      - Fix triple bracket typo: 'Velocity [km/h]]]' -> 'Velocity [km/h]'
    """
    col = col.strip()
    col = col.replace("[%)", "[%]")
    col = col.replace("]]]", "]")
    return col


def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column headers and drop any unnamed columns."""
    df.columns = [_clean_column_name(c) for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df


def load_single_trip(filepath: str) -> pd.DataFrame:
    """Load one trip CSV, clean column headers, and return full DataFrame."""
    df = pd.read_csv(filepath, sep=CSV_SEPARATOR, encoding=CSV_ENCODING)
    df = _standardise_columns(df)
    return df


def get_sorted_trip_files() -> list[str]:
    """Return all 70 CSV file paths in sorted alphabetical order (TripA01..TripB38)."""
    files = sorted(glob.glob(os.path.join(ARCHIVE_DIR, "Trip*.csv")))
    if len(files) != 70:
        raise FileNotFoundError(
            f"Expected 70 trip CSV files in {ARCHIVE_DIR}, found {len(files)}"
        )
    return files


def load_all_trips() -> tuple[list[pd.DataFrame], list[str]]:
    """Load all 70 trip files without dropping rows."""
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
    """
    Resolve matching column name handling degree symbols and whitespace quirks.
    """
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
            if "max" not in col.lower() and "min" not in col.lower():
                return col

    raise KeyError(
        f"Could not find column matching '{target_name}' in columns: {list(df.columns)}"
    )


def extract_features_target(
    df: pd.DataFrame,
    feature_names: list[str] = ENHANCED_7_FEATURES,
    target_name: str = TARGET_SOC,
) -> pd.DataFrame:
    """
    Extract the specified 7 features + target, clean string decimals, and drop NaNs.
    """
    resolved_features = [_find_matching_column(df, f) for f in feature_names]
    resolved_target = _find_matching_column(df, target_name)

    cols_needed = resolved_features + [resolved_target]
    subset = df[cols_needed].copy()

    # Standardise column names
    rename_map = {old: new for old, new in zip(resolved_features, feature_names)}
    rename_map[resolved_target] = target_name
    subset.rename(columns=rename_map, inplace=True)

    # Coerce numeric values (handling comma decimals if present)
    for col in subset.columns:
        if subset[col].dtype == object:
            subset[col] = subset[col].astype(str).str.replace(",", ".")
        subset[col] = pd.to_numeric(subset[col], errors="coerce")

    # Drop any row containing NaN in any of the 7 features or target
    subset.dropna(inplace=True)
    subset.reset_index(drop=True, inplace=True)
    return subset


def prepare_train_test(
    feature_names: list[str] = ENHANCED_7_FEATURES,
    target_name: str = TARGET_SOC,
) -> dict:
    """
    Partition 70 trips into 60 training and 10 testing trips with 7 features.
    """
    trip_dfs, trip_names = load_all_trips()

    train_frames = []
    test_frames = []
    train_sizes = []
    test_sizes = []

    for i, (df, name) in enumerate(zip(trip_dfs, trip_names)):
        clean = extract_features_target(df, feature_names, target_name)

        if i < NUM_TRAIN_TRIPS:
            train_frames.append(clean)
            train_sizes.append(len(clean))
        else:
            test_frames.append(clean)
            test_sizes.append(len(clean))

    train_df = pd.concat(train_frames, ignore_index=True)
    test_df = pd.concat(test_frames, ignore_index=True)

    X_train = train_df[feature_names].values
    y_train = train_df[target_name].values
    X_test = test_df[feature_names].values
    y_test = test_df[target_name].values

    print(f"Training set : {X_train.shape[0]:>10,} instances from {NUM_TRAIN_TRIPS} trips")
    print(f"Testing set  : {X_test.shape[0]:>10,} instances from {len(trip_names) - NUM_TRAIN_TRIPS} trips")
    print(f"Features (7) : {feature_names}")
    print(f"Target       : {target_name}")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_names,
        "train_trips": trip_names[:NUM_TRAIN_TRIPS],
        "test_trips": trip_names[NUM_TRAIN_TRIPS:],
        "train_sizes": train_sizes,
        "test_sizes": test_sizes,
    }


if __name__ == "__main__":
    data = prepare_train_test()
    print(f"X_train shape: {data['X_train'].shape}")
    print(f"X_test shape : {data['X_test'].shape}")
