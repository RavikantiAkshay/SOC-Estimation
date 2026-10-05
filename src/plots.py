"""
plots.py
========
Generates all visualisation plots for the SOC estimation project.

Replicates the paper's key figures:
    - Fig. 3(a): SOC estimation curve (Actual vs. Predicted) on test data
    - Fig. 3(b): Error between actual and predicted SOC
    - Feature importance bar chart (RF built-in)
    - Tree count comparison bar chart (replicating Table 2 visually)
    - k-fold CV results visualisation
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving
import matplotlib.pyplot as plt

from config import PLOTS_DIR


# ----------------------------------------------
# Style defaults
# ----------------------------------------------
plt.rcParams.update({
    "figure.figsize": (14, 5),
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})


# ----------------------------------------------
# Plot 1: SOC Estimation — Actual vs Predicted
#         (Replicates Paper Fig. 3a)
# ----------------------------------------------
def plot_soc_estimation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_trees: int = 25,
    save: bool = True,
) -> None:
    """
    Plot actual SOC vs RF-predicted SOC over test instances.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    instances = np.arange(len(y_true))

    ax.plot(instances, y_true, color="#1f77b4", linewidth=0.8,
            alpha=0.9, label="Actual SOC")
    ax.plot(instances, y_pred, color="#ff7f0e", linewidth=0.6,
            alpha=0.8, linestyle="--", label=f"Predicted by RF ({n_trees} trees)")

    ax.set_xlabel("Instances (s)")
    ax.set_ylabel("SOC (%)")
    ax.set_title(f"SOC Estimation by Random Forest ({n_trees} trees) — Test Data")
    ax.legend(loc="upper right")

    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, "soc_estimation_rf.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# ----------------------------------------------
# Plot 2: Prediction Error
#         (Replicates Paper Fig. 3b)
# ----------------------------------------------
def plot_prediction_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_trees: int = 25,
    save: bool = True,
) -> None:
    """
    Plot the error (actual − predicted) across test instances.
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    errors = y_true - y_pred
    instances = np.arange(len(errors))

    ax.plot(instances, errors, color="#d62728", linewidth=0.5, alpha=0.7)
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")

    ax.fill_between(instances, errors, 0, alpha=0.15, color="#d62728")

    ax.set_xlabel("Instances (s)")
    ax.set_ylabel("Error (SOC %)")
    ax.set_title(f"Prediction Error — RF ({n_trees} trees)")

    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, "prediction_error_rf.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# ----------------------------------------------
# Plot 3: Feature Importance (RF built-in)
# ----------------------------------------------
def plot_feature_importance(
    model,
    feature_names: list[str],
    save: bool = True,
) -> None:
    """
    Bar chart of Random Forest feature importances (MDI / Gini).
    """
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728",
              "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

    bars = ax.bar(
        range(len(importances)),
        importances[sorted_idx],
        color=[colors[i % len(colors)] for i in range(len(importances))],
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(
        [feature_names[i] for i in sorted_idx],
        rotation=25,
        ha="right",
    )
    ax.set_ylabel("Feature Importance (MDI)")
    ax.set_title("Random Forest — Feature Importance")

    # Add value labels on bars
    for bar, val in zip(bars, importances[sorted_idx]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, "feature_importance_rf.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# ----------------------------------------------
# Plot 4: Tree Count Comparison
#         (Visual version of Paper Table 2)
# ----------------------------------------------
def plot_tree_count_comparison(
    cv_results: dict,
    save: bool = True,
) -> None:
    """
    Two-panel comparison chart (RMSE and MAE) across different tree counts
    (25, 50, 75, 100) with scaled axes and exact value annotations so differences
    are clearly distinguishable (replicating Paper Table 2).
    """
    # Support both int and str keys
    raw_keys = list(cv_results.keys())
    tree_counts = sorted([int(k) for k in raw_keys])

    def get_val(n, group, metric):
        item = cv_results.get(n) or cv_results.get(str(n))
        return item[group][metric]

    best_rmse = [get_val(n, "best_metrics", "RMSE") for n in tree_counts]
    avg_rmse = [get_val(n, "avg_metrics", "RMSE") for n in tree_counts]
    best_mae = [get_val(n, "best_metrics", "MAE") for n in tree_counts]
    avg_mae = [get_val(n, "avg_metrics", "MAE") for n in tree_counts]

    # Paper Table 2 benchmarks
    paper_best_rmse = 5.9028
    paper_best_mae = 4.4321

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(tree_counts))
    width = 0.35

    # --- Subplot 1: RMSE ---
    rects1 = ax1.bar(x - width/2, best_rmse, width, label="Ours: Best RMSE", color="#1f77b4", edgecolor="black", linewidth=0.5)
    rects2 = ax1.bar(x + width/2, avg_rmse, width, label="Ours: Avg RMSE", color="#aec7e8", edgecolor="black", linewidth=0.5)
    ax1.axhline(paper_best_rmse, color="#d62728", linestyle="--", linewidth=1.2, label=f"Paper Best: {paper_best_rmse}%")

    # Zoom y-axis to show differences clearly
    min_rmse = min(best_rmse + avg_rmse)
    max_rmse = max(best_rmse + avg_rmse)
    ax1.set_ylim(min_rmse - 0.04, max_rmse + 0.05)

    for i, r in enumerate(rects1):
        h = r.get_height()
        label = f"★ {h:.4f}%\n(Best)" if i == 0 else f"{h:.4f}%"
        color = "#155724" if i == 0 else "black"
        ax1.annotate(label, xy=(r.get_x() + r.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                     fontsize=8.5, fontweight="bold", color=color)
    for r in rects2:
        h = r.get_height()
        ax1.annotate(f"{h:.4f}%", xy=(r.get_x() + r.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8, color="#555555")

    ax1.set_xlabel("Number of Trees", fontweight="bold")
    ax1.set_ylabel("RMSE (%)", fontweight="bold")
    ax1.set_title("Root Mean Square Error (RMSE) vs. Tree Count", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{n} Trees" for n in tree_counts])
    ax1.legend(loc="upper right", framealpha=0.9)

    # --- Subplot 2: MAE ---
    rects3 = ax2.bar(x - width/2, best_mae, width, label="Ours: Best MAE", color="#ff7f0e", edgecolor="black", linewidth=0.5)
    rects4 = ax2.bar(x + width/2, avg_mae, width, label="Ours: Avg MAE", color="#ffbb78", edgecolor="black", linewidth=0.5)
    ax2.axhline(paper_best_mae, color="#d62728", linestyle="--", linewidth=1.2, label=f"Paper Best: {paper_best_mae}%")

    min_mae = min(best_mae + avg_mae)
    max_mae = max(best_mae + avg_mae)
    ax2.set_ylim(min_mae - 0.04, max_mae + 0.05)

    for i, r in enumerate(rects3):
        h = r.get_height()
        label = f"★ {h:.4f}%\n(Best)" if i == 0 else f"{h:.4f}%"
        color = "#8a3b00" if i == 0 else "black"
        ax2.annotate(label, xy=(r.get_x() + r.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                     fontsize=8.5, fontweight="bold", color=color)
    for r in rects4:
        h = r.get_height()
        ax2.annotate(f"{h:.4f}%", xy=(r.get_x() + r.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8, color="#555555")

    ax2.set_xlabel("Number of Trees", fontweight="bold")
    ax2.set_ylabel("MAE (%)", fontweight="bold")
    ax2.set_title("Mean Absolute Error (MAE) vs. Tree Count", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{n} Trees" for n in tree_counts])
    ax2.legend(loc="upper right", framealpha=0.9)

    fig.suptitle("Random Forest Performance Comparison Across Tree Counts (Paper Table 2 Replication)\n(Y-axes scaled to clearly show differences between configurations)",
                 fontsize=12, fontweight="bold", y=1.02)

    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, "tree_count_comparison.png")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"  Saved: {path}")
    plt.close(fig)


# ----------------------------------------------
# Plot 5: Error by SOC Operational Range
# ----------------------------------------------
def plot_error_by_soc_range(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_trees: int = 25,
    save: bool = True,
) -> None:
    """
    Evaluates RMSE and MAE across discrete SOC operational ranges:
    [0-20%, 20-40%, 40-60%, 60-80%, 80-100%].
    Reveals non-linear accuracy patterns across battery capacity regions.
    """
    bins = [0, 20, 40, 60, 80, 100]
    bin_labels = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]

    bin_rmse = []
    bin_mae = []
    bin_counts = []
    active_labels = []
    bin_errors = []

    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        mask = (y_true >= low) & (y_true < high) if i < len(bins) - 2 else (y_true >= low) & (y_true <= high)
        count = int(np.sum(mask))
        if count > 0:
            err = y_true[mask] - y_pred[mask]
            r = float(np.sqrt(np.mean(err ** 2)))
            m = float(np.mean(np.abs(err)))
            bin_rmse.append(r)
            bin_mae.append(m)
            bin_counts.append(count)
            active_labels.append(bin_labels[i])
            bin_errors.append(err)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

    # Subplot 1: Grouped bar chart of RMSE & MAE by SOC Range
    x = np.arange(len(active_labels))
    width = 0.35

    rects1 = ax1.bar(x - width/2, bin_rmse, width, label="RMSE (%)", color="#1f77b4", edgecolor="black", linewidth=0.5)
    rects2 = ax1.bar(x + width/2, bin_mae, width, label="MAE (%)", color="#ff7f0e", edgecolor="black", linewidth=0.5)

    for i, r in enumerate(rects1):
        h = r.get_height()
        ax1.annotate(f"{h:.2f}%\n(N={bin_counts[i]:,})", xy=(r.get_x() + r.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for r in rects2:
        h = r.get_height()
        ax1.annotate(f"{h:.2f}%", xy=(r.get_x() + r.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    ax1.set_xlabel("SOC Operational Range", fontweight="bold")
    ax1.set_ylabel("Error (%)", fontweight="bold")
    ax1.set_title("RMSE & MAE by SOC Operational Range", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(active_labels)
    ax1.set_ylim(0, max(bin_rmse) * 1.25)
    ax1.legend(loc="upper right")

    # Subplot 2: Boxplot of residual distribution per SOC bin
    bp = ax2.boxplot(bin_errors, tick_labels=active_labels, patch_artist=True,
                     showfliers=False, medianprops=dict(color="black", linewidth=1.5))
    colors = ["#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5"]
    for patch, color in zip(bp["boxes"], colors[:len(active_labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax2.axhline(0, color="red", linestyle="--", linewidth=1, label="Zero Error")
    ax2.set_xlabel("SOC Operational Range", fontweight="bold")
    ax2.set_ylabel("Residual Error (Actual − Predicted %)", fontweight="bold")
    ax2.set_title("Error Spread & Distribution by SOC Range", fontweight="bold")
    ax2.legend(loc="upper right")

    fig.suptitle(f"Prediction Accuracy Across Battery SOC Operational Ranges — RF ({n_trees} trees)",
                 fontsize=13, fontweight="bold", y=0.98)

    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, "error_by_soc_range.png")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"  Saved: {path}")
    plt.close(fig)


# ----------------------------------------------
# Plot 6: Error Distribution Histogram
# ----------------------------------------------
def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_trees: int = 25,
    save: bool = True,
) -> None:
    """
    Histogram of prediction errors to visualise bias and spread.
    """
    errors = y_true - y_pred

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(errors, bins=100, color="#2ca02c", alpha=0.7, edgecolor="white",
            linewidth=0.3)

    ax.axvline(x=0, color="red", linewidth=1.2, linestyle="--", label="Zero error")
    ax.axvline(x=np.mean(errors), color="blue", linewidth=1.2,
               linestyle="--", label=f"Mean error = {np.mean(errors):.3f}%")

    ax.set_xlabel("Prediction Error (SOC %)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Error Distribution — RF ({n_trees} trees)")
    ax.legend()

    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, "error_distribution.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# ----------------------------------------------
# Plot 7: Per-Trip Error Breakdown (10 Test Trips)
# ----------------------------------------------
def plot_per_trip_error(
    test_trips: list[str],
    test_sizes: list[int],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_trees: int = 25,
    save: bool = True,
) -> None:
    """
    Evaluates individual RMSE and MAE across each of the 10 unseen test trips
    (TripB29 through TripB38). Demonstrates cross-trip robustness under diverse driving dynamics.
    """
    trip_rmse = []
    trip_mae = []

    start_idx = 0
    for size in test_sizes:
        end_idx = start_idx + size
        trip_y_true = y_true[start_idx:end_idx]
        trip_y_pred = y_pred[start_idx:end_idx]

        r = float(np.sqrt(np.mean((trip_y_true - trip_y_pred) ** 2)))
        m = float(np.mean(np.abs(trip_y_true - trip_y_pred)))

        trip_rmse.append(r)
        trip_mae.append(m)
        start_idx = end_idx

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(test_trips))

    avg_rmse = float(np.mean(trip_rmse))
    avg_mae = float(np.mean(trip_mae))

    # Subplot 1: Per-trip RMSE
    bars1 = ax1.bar(x, trip_rmse, color="#2b5c8f", edgecolor="black", linewidth=0.5, alpha=0.9)
    ax1.axhline(avg_rmse, color="#d62728", linestyle="--", linewidth=1.2, label=f"Mean RMSE: {avg_rmse:.2f}%")

    for i, b in enumerate(bars1):
        h = b.get_height()
        ax1.annotate(f"{h:.2f}%", xy=(b.get_x() + b.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    ax1.set_xlabel("Test Trip Identifier", fontweight="bold")
    ax1.set_ylabel("RMSE (%)", fontweight="bold")
    ax1.set_title("Test Set RMSE per Individual Trip", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(test_trips, rotation=35, ha="right")
    ax1.set_ylim(0, max(trip_rmse) * 1.25)
    ax1.legend(loc="upper right")

    # Subplot 2: Per-trip MAE
    bars2 = ax2.bar(x, trip_mae, color="#e67e22", edgecolor="black", linewidth=0.5, alpha=0.9)
    ax2.axhline(avg_mae, color="#d62728", linestyle="--", linewidth=1.2, label=f"Mean MAE: {avg_mae:.2f}%")

    for i, b in enumerate(bars2):
        h = b.get_height()
        ax2.annotate(f"{h:.2f}%", xy=(b.get_x() + b.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    ax2.set_xlabel("Test Trip Identifier", fontweight="bold")
    ax2.set_ylabel("MAE (%)", fontweight="bold")
    ax2.set_title("Test Set MAE per Individual Trip", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(test_trips, rotation=35, ha="right")
    ax2.set_ylim(0, max(trip_mae) * 1.25)
    ax2.legend(loc="upper right")

    fig.suptitle(f"Individual Trip Generalization Performance across 10 Test Trips — RF ({n_trees} trees)",
                 fontsize=13, fontweight="bold", y=0.98)

    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, "per_trip_error.png")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"  Saved: {path}")
    plt.close(fig)


# ----------------------------------------------
# Generate all plots in one call
# ----------------------------------------------
def generate_all_plots(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    model,
    feature_names: list[str],
    cv_results: dict = None,
    n_trees: int = 25,
    test_trips: list[str] = None,
    test_sizes: list[int] = None,
) -> None:
    """Convenience function to generate all plots at once."""
    print("\n  Generating plots...")

    plot_soc_estimation(y_test, y_pred, n_trees)
    plot_prediction_error(y_test, y_pred, n_trees)
    plot_feature_importance(model, feature_names)

    if cv_results is not None:
        plot_tree_count_comparison(cv_results)

    plot_error_by_soc_range(y_test, y_pred, n_trees)
    plot_error_distribution(y_test, y_pred, n_trees)

    if test_trips is not None and test_sizes is not None:
        plot_per_trip_error(test_trips, test_sizes, y_test, y_pred, n_trees)

    print("  [OK] All plots saved to:", PLOTS_DIR)
