"""
plots_dt.py
===========
Dedicated visualization suite for Decision Tree baseline.
Generates publication-quality diagnostic plots and head-to-head comparison against Random Forest.
All figures are saved strictly within Decision_Tree/results/plots/.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#CBD5E1"
plt.rcParams["axes.linewidth"] = 0.8


def plot_soc_estimation_dt(y_true, y_pred, output_path):
    """Plots actual vs. Decision Tree predicted SOC over test instances."""
    fig, ax = plt.subplots(figsize=(14, 5), dpi=300)
    indices = np.arange(len(y_true))

    ax.plot(indices, y_true, label="Actual SOC (Manufacturer)", color="#1E3A8A", linewidth=1.2, alpha=0.9)
    ax.plot(indices, y_pred, label="Predicted SOC (Decision Tree - 1 Tree)", color="#D97706", linewidth=1.0, linestyle="--", alpha=0.85)

    ax.set_title("SOC Estimation: Actual vs. Single Decision Tree Baseline (Test Set)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Test Set Sample Index (Continuous Time at 1 Hz)", fontsize=11, labelpad=8)
    ax.set_ylabel("Battery State of Charge [%]", fontsize=11, labelpad=8)
    ax.set_ylim(-5, 105)
    ax.grid(True, linestyle="--", alpha=0.4, color="#94A3B8")
    ax.legend(loc="upper right", framealpha=0.95, edgecolor="#CBD5E1")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_prediction_error_dt(y_true, y_pred, output_path):
    """Plots instantaneous residual error for the Decision Tree model."""
    errors = y_true - y_pred
    fig, ax = plt.subplots(figsize=(14, 4.5), dpi=300)
    indices = np.arange(len(errors))

    ax.plot(indices, errors, color="#B45309", linewidth=0.8, alpha=0.8, label="Decision Tree Residual ($y_{true} - y_{pred}$)")
    ax.axhline(0, color="#1E293B", linestyle="-", linewidth=1.0, alpha=0.9, label="Zero Error Reference")

    mae = float(np.mean(np.abs(errors)))
    ax.axhline(mae, color="#DC2626", linestyle=":", linewidth=1.2, label=f"+MAE ({mae:.2f}%)")
    ax.axhline(-mae, color="#DC2626", linestyle=":", linewidth=1.2, label=f"-MAE (-{mae:.2f}%)")

    ax.set_title("Instantaneous Prediction Residuals: Single Decision Tree", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Test Set Sample Index", fontsize=11, labelpad=8)
    ax.set_ylabel("Error (Actual - Predicted) [%]", fontsize=11, labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.4, color="#94A3B8")
    ax.legend(loc="upper right", framealpha=0.95, edgecolor="#CBD5E1")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_feature_importance_dt(importances_dict, output_path):
    """Visualizes MDI / Gini feature importances for the single tree."""
    features = list(importances_dict.keys())
    values = [importances_dict[f] for f in features]

    # Sort descending
    sorted_indices = np.argsort(values)
    features_sorted = [features[i] for i in sorted_indices]
    values_sorted = [values[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    bars = ax.barh(features_sorted, values_sorted, color="#D97706", edgecolor="#B45309", height=0.55, alpha=0.85)

    for bar, val in zip(bars, values_sorted):
        ax.annotate(f"{val:.4f} ({val * 100:.1f}%)", xy=(val, bar.get_y() + bar.get_height() / 2),
                    xytext=(6, 0), textcoords="offset points", ha="left", va="center", fontsize=9.5, fontweight="bold", color="#78350F")

    ax.set_title("Feature Importance: Single Decision Tree (MDI / Gini)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Relative Importance Score", fontsize=11, labelpad=8)
    ax.set_xlim(0, max(values_sorted) * 1.25)
    ax.grid(True, linestyle="--", alpha=0.4, color="#94A3B8", axis="x")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_per_trip_error_dt(test_trips, test_sizes, y_true, y_pred, output_path):
    """Evaluates and visualizes individual RMSE and MAE across each unseen test trip."""
    trip_names = []
    trip_rmses = []
    trip_maes = []

    start = 0
    for name, size in zip(test_trips, test_sizes):
        end = start + size
        yt = y_true[start:end]
        yp = y_pred[start:end]
        err = yt - yp
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))

        trip_names.append(name.replace(".csv", ""))
        trip_rmses.append(rmse)
        trip_maes.append(mae)
        start = end

    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=300)
    x = np.arange(len(trip_names))
    width = 0.36

    rects1 = ax.bar(x - width / 2, trip_rmses, width, label="Trip RMSE [%]", color="#D97706", edgecolor="#B45309", alpha=0.85)
    rects2 = ax.bar(x + width / 2, trip_maes, width, label="Trip MAE [%]", color="#F59E0B", edgecolor="#D97706", alpha=0.85)

    for r in rects1:
        h = r.get_height()
        ax.annotate(f"{h:.2f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#78350F")

    for r in rects2:
        h = r.get_height()
        ax.annotate(f"{h:.2f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#B45309")

    overall_rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ax.axhline(overall_rmse, color="#B45309", linestyle="--", linewidth=1.2, label=f"Overall Test RMSE ({overall_rmse:.2f}%)")

    ax.set_title("Per-Trip Generalization Error: Decision Tree (10 Unseen Test Trips)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Unseen Test Trip Identifier", fontsize=11, labelpad=8)
    ax.set_ylabel("Error Metric [%]", fontsize=11, labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(trip_names, rotation=35, ha="right", fontsize=9.5)
    ax.set_ylim(0, max(trip_rmses) * 1.25)
    ax.grid(True, linestyle="--", alpha=0.4, color="#94A3B8", axis="y")
    ax.legend(loc="upper right", framealpha=0.95, edgecolor="#CBD5E1")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_rf_vs_dt_comparison(rf_metrics, dt_metrics, output_path):
    """Side-by-side grouped bar chart directly comparing Random Forest (25 trees) vs Decision Tree (1 tree)."""
    metrics_to_compare = ["RMSE", "MAE", "MAX_ERROR", "STD_DEV"]
    metric_labels = ["RMSE [%]", "MAE [%]", "Max Error [%]", "Std Dev [%]"]

    rf_vals = [rf_metrics[m] for m in metrics_to_compare]
    dt_vals = [dt_metrics[m] for m in metrics_to_compare]

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    x = np.arange(len(metrics_to_compare))
    width = 0.35

    rects_rf = ax.bar(x - width / 2, rf_vals, width, label="Random Forest (25 Trees - Ensemble)", color="#10B981", edgecolor="#059669", alpha=0.9)
    rects_dt = ax.bar(x + width / 2, dt_vals, width, label="Decision Tree (1 Tree - Single Learner)", color="#D97706", edgecolor="#B45309", alpha=0.9)

    for r in rects_rf:
        h = r.get_height()
        ax.annotate(f"{h:.2f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#065F46")

    for r in rects_dt:
        h = r.get_height()
        ax.annotate(f"{h:.2f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#78350F")

    ax.set_title("Ensemble Gain: Random Forest (25 Trees) vs. Single Decision Tree (1 Tree)", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Evaluation Metric", fontsize=11, labelpad=8)
    ax.set_ylabel("Error Value [%]", fontsize=11, labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(max(rf_vals), max(dt_vals)) * 1.18)
    ax.grid(True, linestyle="--", alpha=0.4, color="#94A3B8", axis="y")
    ax.legend(loc="upper left", framealpha=0.95, edgecolor="#CBD5E1")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def generate_all_dt_plots(y_true, y_pred, feature_importances, test_trips, test_sizes, rf_metrics, dt_metrics, plots_dir):
    """Generates all 5 Decision Tree plots."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Generating Decision Tree Plots ---")
    plot_soc_estimation_dt(y_true, y_pred, plots_dir / "soc_estimation_dt.png")
    plot_prediction_error_dt(y_true, y_pred, plots_dir / "prediction_error_dt.png")
    plot_feature_importance_dt(feature_importances, plots_dir / "feature_importance_dt.png")
    plot_per_trip_error_dt(test_trips, test_sizes, y_true, y_pred, plots_dir / "per_trip_error_dt.png")
    plot_rf_vs_dt_comparison(rf_metrics, dt_metrics, plots_dir / "rf_vs_dt_comparison.png")
    print(f"All 5 Decision Tree plots saved to: {plots_dir}")
