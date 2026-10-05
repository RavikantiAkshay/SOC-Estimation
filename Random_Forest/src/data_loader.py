"""
data_loader.py
==============
Handles all data loading, column cleaning, NaN removal, and
train/test splitting exactly as described in the paper.

Paper spec:
    - 70 CSV files (semicolon-delimited, latin1 encoded)
    - Trips 1-60 -> training  (945,027 clean instances)
    - Trips 61-70 -> testing  (118,974 clean instances)
    - 4 input features: Voltage, Current, Battery Temp, Ambient Temp
    - 1 target: SoC [%]  (manufacturer-estimated, not displayed)
    - Preprocessing: drop rows where any of these 5 columns is NaN
"""

import os
import glob
import numpy as np
import pandas as pd

from config import (
    ARCHIVE_DIR,
    CSV_SEPARATOR,
    CSV_ENCODING,
    PAPER_FEATURES,
    TARGET_SOC,
    NUM_TRAIN_TRIPS,
)


# ----------------------------------------------
# Column name normalisation
# ----------------------------------------------
def _clean_column_name(col: str) -> str:
    """
    Fix known quirks in the raw CSV headers:
      - Strip leading/trailing whitespace
      - Replace mismatched bracket  'max. SoC [%)'  ->  'max. SoC [%]'
      - Fix triple-bracket typo     'Velocity [km/h]]]'  ->  'Velocity [km/h]'
    """
    col = col.strip()
    col = col.replace("[%)", "[%]")
    col = col.replace("]]]", "]")
    return col


def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply column name fixes and drop any 'Unnamed' columns."""
    df.columns = [_clean_column_name(c) for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df


# ----------------------------------------------
# Single-file loader
# ----------------------------------------------
def load_single_trip(filepath: str) -> pd.DataFrame:
    """
    Read one trip CSV, clean column names, and return the full
    DataFrame (no filtering or NaN removal yet).
    """
    df = pd.read_csv(filepath, sep=CSV_SEPARATOR, encoding=CSV_ENCODING)
    df = _standardise_columns(df)
    return df


# ----------------------------------------------
# Full dataset loader
# ----------------------------------------------
def get_sorted_trip_files() -> list[str]:
    """
    Return all 70 CSV file paths sorted alphabetically.
    Alphabetical order gives: TripA01..TripA32, TripB01..TripB38.
    """
    files = sorted(glob.glob(os.path.join(ARCHIVE_DIR, "Trip*.csv")))
    if len(files) != 70:
        raise FileNotFoundError(
            f"Expected 70 trip CSV files in {ARCHIVE_DIR}, found {len(files)}"
        )
    return files


def load_all_trips() -> tuple[list[pd.DataFrame], list[str]]:
    """
    Load all 70 trip CSVs.

    Returns
    -------
    trip_dfs : list[pd.DataFrame]
        One DataFrame per trip, columns standardised but NO rows dropped.
    trip_names : list[str]
        Trip identifiers (e.g. 'TripA01', 'TripB38').
    """
    files = get_sorted_trip_files()
    trip_dfs = []
    trip_names = []

    for fpath in files:
        df = load_single_trip(fpath)
        name = os.path.splitext(os.path.basename(fpath))[0]
        trip_dfs.append(df)
        trip_names.append(name)

    return trip_dfs, trip_names


# ----------------------------------------------
# Feature / target extraction + NaN cleaning
# ----------------------------------------------
def _find_matching_column(df: pd.DataFrame, target_name: str) -> str:
    """
    Find the actual column in `df` that matches `target_name`.

    Some CSVs encode the degree symbol differently depending on the
    editor that generated them (°C vs direct byte).  We do a fuzzy
    match: strip non-ASCII and compare the ASCII skeleton.
    """
    def ascii_skeleton(s: str) -> str:
        return s.encode("ascii", "ignore").decode("ascii").strip().lower()

    target_skel = ascii_skeleton(target_name)

    for col in df.columns:
        if ascii_skeleton(col) == target_skel:
            return col

    # Fallback: check if target_name substring is contained
    for col in df.columns:
        if target_name.split("[")[0].strip().lower() in col.lower():
            # Avoid matching 'max. Battery Temperature' when we want 'Battery Temperature'
            if "max" not in col.lower() and "min" not in col.lower():
                return col

    raise KeyError(
        f"Could not find column matching '{target_name}' in columns: {list(df.columns)}"
    )


def extract_features_target(
    df: pd.DataFrame,
    feature_names: list[str] = PAPER_FEATURES,
    target_name: str = TARGET_SOC,
) -> pd.DataFrame:
    """
    From a full trip DataFrame, extract only the required feature
    columns and target column, then drop any rows with NaN in those
    columns.

    Returns a clean DataFrame with columns:
        [feature_1, feature_2, ..., feature_n, target]
    """
    # Resolve actual column names (handles encoding quirks)
    resolved_features = [_find_matching_column(df, f) for f in feature_names]
    resolved_target = _find_matching_column(df, target_name)

    cols_needed = resolved_features + [resolved_target]
    subset = df[cols_needed].copy()

    # Rename to standardised names for downstream consistency
    rename_map = {old: new for old, new in zip(resolved_features, feature_names)}
    rename_map[resolved_target] = target_name
    subset.rename(columns=rename_map, inplace=True)

    # Drop rows with any NaN in the selected columns
    subset.dropna(inplace=True)

    # Ensure all columns are numeric
    for col in subset.columns:
        subset[col] = pd.to_numeric(subset[col], errors="coerce")
    subset.dropna(inplace=True)

    subset.reset_index(drop=True, inplace=True)
    return subset


# ----------------------------------------------
# Train / test split by trip index
# ----------------------------------------------
def prepare_train_test(
    feature_names: list[str] = PAPER_FEATURES,
    target_name: str = TARGET_SOC,
) -> dict:
    """
    Load all 70 trips, extract features/target, clean NaNs, and
    split into training (trips 1-60) and testing (trips 61-70).

    Returns
    -------
    dict with keys:
        'X_train'      : np.ndarray  (N_train, n_features)
        'y_train'      : np.ndarray  (N_train,)
        'X_test'       : np.ndarray  (N_test, n_features)
        'y_test'       : np.ndarray  (N_test,)
        'feature_names': list[str]
        'train_trips'  : list[str]   trip names used for training
        'test_trips'   : list[str]   trip names used for testing
        'train_sizes'  : list[int]   number of clean rows per training trip
        'test_sizes'   : list[int]   number of clean rows per testing trip
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
    print(f"Features     : {feature_names}")
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


# ----------------------------------------------
# Quick sanity check when run directly
# ----------------------------------------------
if __name__ == "__main__":
    data = prepare_train_test()
    print(f"\nX_train shape : {data['X_train'].shape}")
    print(f"y_train range : [{data['y_train'].min():.2f}, {data['y_train'].max():.2f}]")
    print(f"X_test shape  : {data['X_test'].shape}")
    print(f"y_test range  : [{data['y_test'].min():.2f}, {data['y_test'].max():.2f}]")
