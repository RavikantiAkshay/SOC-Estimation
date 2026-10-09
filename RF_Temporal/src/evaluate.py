"""
evaluate.py
===========
Evaluation metrics computation for the 12-feature Temporal SOC estimation model:
    1. RMSE      — Root Mean Square Error (%)
    2. MAE       — Mean Absolute Error (%)
    3. MAX ERROR — Maximum peak absolute residual (%)
    4. STD DEV   — Standard deviation of residuals (%)
"""

import numpy as np


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Square Error in SOC %."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error in SOC %."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Maximum absolute residual in SOC %."""
    return float(np.max(np.abs(y_true - y_pred)))


def compute_std_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Standard deviation of residuals in SOC %."""
    errors = y_true - y_pred
    return float(np.std(errors, ddof=0))


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all 4 standard regression metrics in a single call."""
    return {
        "RMSE": compute_rmse(y_true, y_pred),
        "MAE": compute_mae(y_true, y_pred),
        "MAX_ERROR": compute_max_error(y_true, y_pred),
        "STD_DEV": compute_std_error(y_true, y_pred),
    }


def print_metrics(metrics: dict, label: str = "") -> None:
    """Pretty-print metrics formatted as percentages."""
    header = f"  Metrics: {label}  " if label else "  Metrics  "
    print(f"\n{'=' * 50}")
    print(f"{header:^50}")
    print(f"{'=' * 50}")
    print(f"  {'RMSE':<15} : {metrics['RMSE']:.4f} %")
    print(f"  {'MAE':<15} : {metrics['MAE']:.4f} %")
    print(f"  {'MAX ERROR':<15} : {metrics['MAX_ERROR']:.4f} %")
    print(f"  {'STD DEV':<15} : {metrics['STD_DEV']:.4f} %")
    print(f"{'=' * 50}\n")
