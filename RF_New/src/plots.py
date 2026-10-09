"""
plots.py
========
Diagnostic visualization suite for the 7-feature maximum-improvement Random Forest model.
Generates publication-quality charts saved to results/plots/.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for server execution
import matplotlib.pyplot as plt

from config import PLOTS_DIR, BASELINE_4F_BENCHMARKS

# Consistent style tokens
plt.rcParams.update({
    "figure.figsize": (14, 5),
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})


def plot_soc_estimation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_trees: int,
    save: bool = True,
) -> None:
    """Actual SOC vs 7-Feature Predicted SOC across test instances."""
    fig, ax = plt.subplots(figsize=(14, 5))
    instances = np.arange(len(y_true))

    ax.plot(instances, y_true, color="#1e40af", linewidth=0.9, alpha=0.95, label="Actual SOC (BMS Ground Truth)")
    ax.plot(instances, y_pred, color="#059669", linewidth=0.7, alpha=0.85, linestyle="--",
            label=f"Predicted by 7-Feature RF ({n_trees} trees)")

    ax.set_xlabel("Test Instances (Seconds)", fontweight="bold")
    ax.set_ylabel("SOC (%)", fontweight="bold")
    ax.set_title(f"SOC Estimation by Enhanced 7-Feature Random Forest ({n_trees} Trees) — Test Data",
                 fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, "soc_estimation_rf.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_prediction_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_trees: int,
    save: bool = True,
) -> None:
    """Instantaneous prediction error across test set."""
    fig, ax = plt.subplots(figsize=(14, 4.5))
    errors = y_true - y_pred
    instances = np.arange(len(errors))

    ax.plot(instances, errors, color="#dc2626", linewidth=0.5, alpha=0.75)
    ax.axhline(y=0, color="#0f172a", linewidth=0.9, linestyle="-")
    ax.fill_between(instances, errors, 0, alpha=0.15, color="#dc2626")

    ax.set_xlabel("Test Instances (Seconds)", fontweight="bold")
    ax.set_ylabel("Error (Actual − Predicted %)", fontweight="bold")
    ax.set_title(f"Prediction Error Profile — Enhanced 7-Feature RF ({n_trees} Trees)", fontweight="bold")

    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, "prediction_error_rf.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_feature_importance(
    model,
    feature_names: list[str],
    save: bool = True,
) -> None:
    """Horizontal bar chart showing relative feature importances across all 7 features."""
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Distinct colors for baseline vs newly added features
    colors = []
    baseline_set = {"Battery Voltage [V]", "Battery Current [A]", "Battery Temperature [°C]", "Ambient Temperature [°C]"}
    for idx in sorted_idx:
        feat = feature_names[idx]
        colors.append("#2563eb" if feat in baseline_set else "#059669")

    bars = ax.barh(range(len(sorted_idx)), importances[sorted_idx], color=colors, height=0.6, edgecolor="#0f172a", linewidth=0.5)

    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontweight="bold")
    ax.set_xlabel("MDI Feature Importance Score", fontweight="bold")
    ax.set_title("7-Feature Random Forest Relative Importance Ranking", fontweight="bold")

    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w * 100:.2f}%", xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=9.5, fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2563eb", edgecolor="#0f172a", label="Baseline Observable"),
        Patch(facecolor="#059669", edgecolor="#0f172a", label="Added Powertrain / Kinematic Feature"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9)
    ax.set_xlim(0, max(importances) * 1.15)

    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, "feature_importance_rf.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_tree_count_comparison(
    cv_results: dict,
    save: bool = True,
) -> None:
    """Comparative bar charts for RMSE and MAE across tree counts."""
    tree_counts = sorted(cv_results.keys())
    x = np.arange(len(tree_counts))
    width = 0.35

    best_rmse = [cv_results[n]["best_metrics"]["RMSE"] for n in tree_counts]
    avg_rmse = [cv_results[n]["avg_metrics"]["RMSE"] for n in tree_counts]
    best_mae = [cv_results[n]["best_metrics"]["MAE"] for n in tree_counts]
    avg_mae = [cv_results[n]["avg_metrics"]["MAE"] for n in tree_counts]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # RMSE plot
    b1 = ax1.bar(x - width/2, best_rmse, width, label="Best Run RMSE", color="#059669", edgecolor="#0f172a", linewidth=0.5)
    b2 = ax1.bar(x + width/2, avg_rmse, width, label="Average RMSE", color="#6ee7b7", edgecolor="#0f172a", linewidth=0.5)
    ax1.set_xlabel("Number of Trees", fontweight="bold")
    ax1.set_ylabel("RMSE (%)", fontweight="bold")
    ax1.set_title("Test RMSE across Tree Counts (7 Features)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(t) for t in tree_counts])
    ax1.legend(loc="upper right")
    ax1.set_ylim(0, max(avg_rmse) * 1.25)
    for bar in b1:
        ax1.annotate(f"{bar.get_height():.2f}%", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for bar in b2:
        ax1.annotate(f"{bar.get_height():.2f}%", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    # MAE plot
    b3 = ax2.bar(x - width/2, best_mae, width, label="Best Run MAE", color="#2563eb", edgecolor="#0f172a", linewidth=0.5)
    b4 = ax2.bar(x + width/2, avg_mae, width, label="Average MAE", color="#93c5fd", edgecolor="#0f172a", linewidth=0.5)
    ax2.set_xlabel("Number of Trees", fontweight="bold")
    ax2.set_ylabel("MAE (%)", fontweight="bold")
    ax2.set_title("Test MAE across Tree Counts (7 Features)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(t) for t in tree_counts])
    ax2.legend(loc="upper right")
    ax2.set_ylim(0, max(avg_mae) * 1.25)
    for bar in b3:
        ax2.annotate(f"{bar.get_height():.2f}%", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for bar in b4:
        ax2.annotate(f"{bar.get_height():.2f}%", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, "tree_count_comparison.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_error_by_soc_range(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_trees: int,
    save: bool = True,
) -> None:
    """Evaluates error across discrete SOC ranges (0-20%, 20-40%, etc.)."""
    ranges = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    labels = [f"{lo}-{hi}%" for lo, hi in ranges]
    errors = y_true - y_pred

    rmse_per_range = []
    mae_per_range = []
    box_data = []

    for lo, hi in ranges:
        mask = (y_true >= lo) & (y_true < hi) if hi < 100 else (y_true >= lo) & (y_true <= hi)
        errs = errors[mask]
        if len(errs) > 0:
            rmse_per_range.append(np.sqrt(np.mean(errs**2)))
            mae_per_range.append(np.mean(np.abs(errs)))
            box_data.append(errs)
        else:
            rmse_per_range.append(0.0)
            mae_per_range.append(0.0)
            box_data.append(np.array([0.0]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(labels))
    w = 0.35

    ax1.bar(x - w/2, rmse_per_range, w, label="RMSE", color="#059669", edgecolor="#0f172a", linewidth=0.5)
    ax1.bar(x + w/2, mae_per_range, w, label="MAE", color="#2563eb", edgecolor="#0f172a", linewidth=0.5)
    ax1.set_xlabel("SOC Operational Range (%)", fontweight="bold")
    ax1.set_ylabel("Error (%)", fontweight="bold")
    ax1.set_title("RMSE & MAE across Operational SOC Intervals", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend(loc="upper right")

    ax2.boxplot(box_data, tick_labels=labels, showfliers=False, patch_artist=True,
                boxprops=dict(facecolor="#d1fae5", color="#065f46"),
                medianprops=dict(color="#047857", linewidth=2))
    ax2.axhline(0, color="#0f172a", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("SOC Operational Range (%)", fontweight="bold")
    ax2.set_ylabel("Residual (Actual − Predicted %)", fontweight="bold")
    ax2.set_title("Residual Distribution (Interquartile Range)", fontweight="bold")

    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, "error_by_soc_range.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_trees: int,
    save: bool = True,
) -> None:
    """Histogram of prediction residuals with mean and std lines."""
    errors = y_true - y_pred
    mean_err = np.mean(errors)
    std_err = np.std(errors)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(errors, bins=100, color="#10b981", edgecolor="#047857", alpha=0.75, density=True)
    ax.axvline(0, color="#0f172a", linestyle="-", linewidth=1.2, label="Zero Error Line")
    ax.axvline(mean_err, color="#dc2626", linestyle="--", linewidth=1.2, label=f"Mean Error: {mean_err:+.3f}%")

    ax.set_xlabel("Prediction Residual (SOC %)", fontweight="bold")
    ax.set_ylabel("Probability Density", fontweight="bold")
    ax.set_title(f"Error Distribution — 7-Feature RF ({n_trees} Trees) [Std: {std_err:.3f}%]", fontweight="bold")
    ax.legend(loc="upper right")

    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, "error_distribution.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_per_trip_error(
    test_trips: list[str],
    test_sizes: list[int],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_trees: int,
    save: bool = True,
) -> None:
    """Individual trip generalization performance across test trips."""
    trip_rmse = []
    trip_mae = []
    offset = 0

    for sz in test_sizes:
        yt = y_true[offset : offset + sz]
        yp = y_pred[offset : offset + sz]
        trip_rmse.append(np.sqrt(np.mean((yt - yp)**2)))
        trip_mae.append(np.mean(np.abs(yt - yp)))
        offset += sz

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(test_trips))

    b1 = ax1.bar(x, trip_rmse, color="#059669", edgecolor="#0f172a", linewidth=0.5)
    ax1.axhline(np.mean(trip_rmse), color="#dc2626", linestyle="--", label=f"Mean: {np.mean(trip_rmse):.2f}%")
    ax1.set_xticks(x)
    ax1.set_xticklabels(test_trips, rotation=35, ha="right")
    ax1.set_ylabel("RMSE (%)", fontweight="bold")
    ax1.set_title("Test RMSE per Trip (7 Features)", fontweight="bold")
    ax1.legend(loc="upper right")
    for bar in b1:
        ax1.annotate(f"{bar.get_height():.2f}%", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    b2 = ax2.bar(x, trip_mae, color="#2563eb", edgecolor="#0f172a", linewidth=0.5)
    ax2.axhline(np.mean(trip_mae), color="#dc2626", linestyle="--", label=f"Mean: {np.mean(trip_mae):.2f}%")
    ax2.set_xticks(x)
    ax2.set_xticklabels(test_trips, rotation=35, ha="right")
    ax2.set_ylabel("MAE (%)", fontweight="bold")
    ax2.set_title("Test MAE per Trip (7 Features)", fontweight="bold")
    ax2.legend(loc="upper right")
    for bar in b2:
        ax2.annotate(f"{bar.get_height():.2f}%", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, "per_trip_error.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_comparison_4f_vs_7f(
    best_7f_metrics: dict,
    best_n: int,
    save: bool = True,
) -> None:
    """Direct Head-to-Head Comparison: 4-Feature Baseline vs 7-Feature Enhanced."""
    base_m = BASELINE_4F_BENCHMARKS.get(str(best_n), BASELINE_4F_BENCHMARKS["25"])
    
    metrics_labels = ["RMSE", "MAE", "MAX ERROR"]
    base_vals = [base_m["best_rmse"], base_m["best_mae"], base_m["best_max"]]
    enh_vals = [best_7f_metrics["RMSE"], best_7f_metrics["MAE"], best_7f_metrics["MAX_ERROR"]]

    x = np.arange(len(metrics_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))
    b1 = ax.bar(x - width/2, base_vals, width, label="Baseline (4 Features)", color="#3b82f6", edgecolor="#0f172a", linewidth=0.6)
    b2 = ax.bar(x + width/2, enh_vals, width, label="Enhanced (7 Features)", color="#10b981", edgecolor="#0f172a", linewidth=0.6)

    ax.set_ylabel("Error Metric (%)", fontweight="bold")
    ax.set_title(f"Head-to-Head: 4-Feature Baseline vs 7-Feature Enhanced RF ({best_n} Trees)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_labels, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, max(base_vals + enh_vals) * 1.25)

    for bar in b1:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    for bar in b2:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, "rf_4f_vs_7f_comparison.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def generate_all_plots(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    model,
    feature_names: list[str],
    cv_results: dict,
    best_n: int,
    best_metrics: dict,
    test_trips: list[str] = None,
    test_sizes: list[int] = None,
) -> None:
    """Generate all 8 visualization artifacts in a single coordinated pipeline."""
    print("\n  Generating 8 diagnostic plots for RF_New...")
    plot_soc_estimation(y_test, y_pred, best_n)
    plot_prediction_error(y_test, y_pred, best_n)
    plot_feature_importance(model, feature_names)
    plot_tree_count_comparison(cv_results)
    plot_error_by_soc_range(y_test, y_pred, best_n)
    plot_error_distribution(y_test, y_pred, best_n)
    if test_trips is not None and test_sizes is not None:
        plot_per_trip_error(test_trips, test_sizes, y_test, y_pred, best_n)
    plot_comparison_4f_vs_7f(best_metrics, best_n)
    print(f"  [OK] All plots saved to: {PLOTS_DIR}")
