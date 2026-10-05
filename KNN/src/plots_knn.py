"""
plots_knn.py
============
Dedicated visualization suite for K-Nearest Neighbors (KNN) baseline.
Generates publication-quality diagnostic plots and head-to-head comparison against Random Forest.
All figures are saved strictly within KNN/results/plots/.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#CBD5E1"
plt.rcParams["axes.linewidth"] = 0.8


def plot_soc_estimation_knn(y_true, y_pred, output_path):
    """Plots actual vs. KNN predicted SOC over test instances."""
    fig, ax = plt.subplots(figsize=(14, 5), dpi=300)
    indices = np.arange(len(y_true))

    ax.plot(indices, y_true, label="Actual SOC (Manufacturer)", color="#1E3A8A", linewidth=1.2, alpha=0.9)
    ax.plot(indices, y_pred, label="Predicted SOC (KNN - k=5)", color="#8B5CF6", linewidth=1.0, linestyle="--", alpha=0.85)

    ax.set_title("SOC Estimation: Actual vs. K-Nearest Neighbors Baseline (k=5)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Test Set Sample Index (Continuous Time at 1 Hz)", fontsize=11, labelpad=8)
    ax.set_ylabel("Battery State of Charge [%]", fontsize=11, labelpad=8)
    ax.set_ylim(-5, 105)
    ax.grid(True, linestyle="--", alpha=0.4, color="#94A3B8")
    ax.legend(loc="upper right", framealpha=0.95, edgecolor="#CBD5E1")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_prediction_error_knn(y_true, y_pred, output_path):
    """Plots instantaneous residual error for the KNN model."""
    errors = y_true - y_pred
    fig, ax = plt.subplots(figsize=(14, 4.5), dpi=300)
    indices = np.arange(len(errors))

    ax.plot(indices, errors, color="#7C3AED", linewidth=0.8, alpha=0.8, label="KNN Residual ($y_{true} - y_{pred}$)")
    ax.axhline(0, color="#1E293B", linestyle="-", linewidth=1.0, alpha=0.9, label="Zero Error Reference")

    mae = float(np.mean(np.abs(errors)))
    ax.axhline(mae, color="#F59E0B", linestyle=":", linewidth=1.2, label=f"+MAE ({mae:.2f}%)")
    ax.axhline(-mae, color="#F59E0B", linestyle=":", linewidth=1.2, label=f"-MAE (-{mae:.2f}%)")

    ax.set_title("Instantaneous Prediction Residuals: KNN (k=5)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Test Set Sample Index", fontsize=11, labelpad=8)
    ax.set_ylabel("Error (Actual - Predicted) [%]", fontsize=11, labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.4, color="#94A3B8")
    ax.legend(loc="upper right", framealpha=0.95, edgecolor="#CBD5E1")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_error_distribution_knn(y_true, y_pred, output_path):
    """Plots error distribution histogram and normal density curve for KNN."""
    errors = y_true - y_pred
    mean_err = float(np.mean(errors))
    std_err = float(np.std(errors))

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    n_bins = 100
    counts, bins, patches = ax.hist(errors, bins=n_bins, density=True, color="#C4B5FD", edgecolor="#7C3AED", alpha=0.6, label="Residual Distribution")

    x_range = np.linspace(errors.min(), errors.max(), 500)
    gauss = (1 / (std_err * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean_err) / std_err) ** 2)
    ax.plot(x_range, gauss, color="#6D28D9", linewidth=2.0, label=f"Normal Fit (mean={mean_err:.2f}%, std={std_err:.2f}%)")

    ax.axvline(0, color="#1E293B", linestyle="--", linewidth=1.2, label="Zero Error Line")
    ax.axvline(mean_err, color="#2563EB", linestyle=":", linewidth=1.2, label=f"Mean Error ({mean_err:.2f}%)")

    ax.set_title("Residual Error Distribution: KNN Baseline (k=5)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Prediction Error (Actual - Predicted) [%]", fontsize=11, labelpad=8)
    ax.set_ylabel("Probability Density", fontsize=11, labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.4, color="#94A3B8")
    ax.legend(loc="upper right", framealpha=0.95, edgecolor="#CBD5E1")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_per_trip_error_knn(test_trips, test_sizes, y_true, y_pred, output_path):
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

    rects1 = ax.bar(x - width / 2, trip_rmses, width, label="Trip RMSE [%]", color="#8B5CF6", edgecolor="#6D28D9", alpha=0.85)
    rects2 = ax.bar(x + width / 2, trip_maes, width, label="Trip MAE [%]", color="#A78BFA", edgecolor="#7C3AED", alpha=0.85)

    for r in rects1:
        h = r.get_height()
        ax.annotate(f"{h:.2f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#5B21B6")

    for r in rects2:
        h = r.get_height()
        ax.annotate(f"{h:.2f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#6D28D9")

    overall_rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ax.axhline(overall_rmse, color="#6D28D9", linestyle="--", linewidth=1.2, label=f"Overall Test RMSE ({overall_rmse:.2f}%)")

    ax.set_title("Per-Trip Generalization Error: KNN (10 Unseen Test Trips)", fontsize=13, fontweight="bold", pad=12)
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


def plot_rf_vs_knn_comparison(rf_metrics, knn_metrics, output_path):
    """Side-by-side grouped bar chart directly comparing Random Forest (25 trees) vs KNN (k=5)."""
    metrics_to_compare = ["RMSE", "MAE", "MAX_ERROR", "STD_DEV"]
    metric_labels = ["RMSE [%]", "MAE [%]", "Max Error [%]", "Std Dev [%]"]

    rf_vals = [rf_metrics[m] for m in metrics_to_compare]
    knn_vals = [knn_metrics[m] for m in metrics_to_compare]

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    x = np.arange(len(metrics_to_compare))
    width = 0.35

    rects_rf = ax.bar(x - width / 2, rf_vals, width, label="Random Forest (25 Trees - Ensemble)", color="#10B981", edgecolor="#059669", alpha=0.9)
    rects_knn = ax.bar(x + width / 2, knn_vals, width, label="K-Nearest Neighbors (k=5, Distance-Weighted)", color="#8B5CF6", edgecolor="#6D28D9", alpha=0.9)

    for r in rects_rf:
        h = r.get_height()
        ax.annotate(f"{h:.2f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#065F46")

    for r in rects_knn:
        h = r.get_height()
        ax.annotate(f"{h:.2f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#5B21B6")

    ax.set_title("Benchmark Comparison: Random Forest (25 Trees) vs. K-Nearest Neighbors (k=5)", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Evaluation Metric", fontsize=11, labelpad=8)
    ax.set_ylabel("Error Value [%]", fontsize=11, labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(max(rf_vals), max(knn_vals)) * 1.18)
    ax.grid(True, linestyle="--", alpha=0.4, color="#94A3B8", axis="y")
    ax.legend(loc="upper left", framealpha=0.95, edgecolor="#CBD5E1")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def generate_all_knn_plots(y_true, y_pred, test_trips, test_sizes, rf_metrics, knn_metrics, plots_dir):
    """Generates all 5 KNN diagnostic and comparative plots."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Generating KNN Diagnostic Plots ---")
    plot_soc_estimation_knn(y_true, y_pred, plots_dir / "soc_estimation_knn.png")
    plot_prediction_error_knn(y_true, y_pred, plots_dir / "prediction_error_knn.png")
    plot_error_distribution_knn(y_true, y_pred, plots_dir / "error_distribution_knn.png")
    plot_per_trip_error_knn(test_trips, test_sizes, y_true, y_pred, plots_dir / "per_trip_error_knn.png")
    plot_rf_vs_knn_comparison(rf_metrics, knn_metrics, plots_dir / "rf_vs_knn_comparison.png")
    print(f"All 5 KNN plots saved to: {plots_dir}")
