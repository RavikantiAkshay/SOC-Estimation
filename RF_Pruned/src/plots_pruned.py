"""
plots_pruned.py
===============
Publication-quality diagnostic visualizations for Stage 4:
Random Forest Depth Pruning & Pareto Complexity Optimization.

Generates:
  1. pareto_frontier_tradeoff.png  - Complexity (Nodes) vs Accuracy (RMSE & MAX Error)
  2. depth_sweep_comparison.png    - Bar chart of RMSE, MAE, MAX across depths
  3. four_stage_evolution.png      - 4-Stage journey: Baseline -> Powertrain -> SOTA -> Pruned
  4. soc_estimation_pruned.png     - Actual vs Predicted SOC tracking curve
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Style standards
plt.rcParams.update({
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
    "font.family": "sans-serif",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_pareto_frontier(results: dict, best_depth_key: str, output_path: str):
    """
    Dual-panel Pareto Frontier curve showing Model Complexity (Node Count & C Flash MB)
    vs Prediction Errors (RMSE & Peak MAX Error).
    """
    depth_keys = [k for k in ["10", "14", "18", "22", "26", "unconstrained"] if k in results]
    nodes = [results[k]["total_nodes"] for k in depth_keys]
    rmses = [results[k]["rmse"] for k in depth_keys]
    max_errors = [results[k]["max_error"] for k in depth_keys]
    labels = [results[k]["depth_label"] for k in depth_keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8))
    fig.patch.set_facecolor("#ffffff")

    # Panel 1: Nodes vs RMSE
    ax1.plot(nodes, rmses, "o-", color="#0284c7", linewidth=2.5, markersize=8, label="Pareto Trajectory", zorder=3)
    ax1.set_xlabel("Model Complexity: Total Tree Nodes in Forest", fontsize=11, fontweight="bold", labelpad=8)
    ax1.set_ylabel("Test Set RMSE (%)", fontsize=11, fontweight="bold", labelpad=8)
    ax1.set_title("Pareto Frontier: Accuracy vs. Tree Complexity", fontsize=12, fontweight="bold", pad=12)
    ax1.grid(True, linestyle="--", alpha=0.5, zorder=1)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x/1e3):,}k"))

    # Highlight Optimal Knee
    best_nodes = results[best_depth_key]["total_nodes"]
    best_rmse = results[best_depth_key]["rmse"]
    ax1.scatter([best_nodes], [best_rmse], color="#dc2626", s=180, zorder=5, edgecolor="#991b1b", linewidth=2, label=f"Pareto Optimal (Depth {best_depth_key})")
    ax1.annotate(
        f"Pareto Sweet Spot\nDepth {best_depth_key} ({results[best_depth_key]['node_reduction_pct']:.0f}% node reduction)\nRMSE: {best_rmse:.2f}%",
        xy=(best_nodes, best_rmse),
        xytext=(best_nodes + 400000, best_rmse + 0.35),
        arrowprops=dict(facecolor="#dc2626", shrink=0.08, width=1.5, headwidth=7),
        fontsize=9.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fef2f2", edgecolor="#fca5a5", alpha=0.95),
    )
    for i, txt in enumerate(labels):
        ax1.annotate(f"d={txt}", (nodes[i], rmses[i]), textcoords="offset points", xytext=(-10, 10), ha="center", fontsize=8.5, color="#334155")
    ax1.legend(loc="upper right", framealpha=0.9)

    # Panel 2: Nodes vs Peak MAX Error
    ax2.plot(nodes, max_errors, "s-", color="#10b981", linewidth=2.5, markersize=8, label="Peak Error Curve", zorder=3)
    best_max = results[best_depth_key]["max_error"]
    ax2.scatter([best_nodes], [best_max], color="#dc2626", s=180, zorder=5, edgecolor="#991b1b", linewidth=2, label=f"Pareto Optimal (Depth {best_depth_key})")
    ax2.set_xlabel("Model Complexity: Total Tree Nodes in Forest", fontsize=11, fontweight="bold", labelpad=8)
    ax2.set_ylabel("Peak MAX Error (%)", fontsize=11, fontweight="bold", labelpad=8)
    ax2.set_title("Worst-Case Transient Error Bound vs. Complexity", fontsize=12, fontweight="bold", pad=12)
    ax2.grid(True, linestyle="--", alpha=0.5, zorder=1)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x/1e3):,}k"))
    for i, txt in enumerate(labels):
        ax2.annotate(f"d={txt}", (nodes[i], max_errors[i]), textcoords="offset points", xytext=(-10, 10), ha="center", fontsize=8.5, color="#334155")
    ax2.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_depth_sweep_comparison(results: dict, best_depth_key: str, output_path: str):
    """Grouped bar comparison of RMSE, MAE, and Node Reduction across candidate depths."""
    depth_keys = [k for k in ["10", "14", "18", "22", "26", "unconstrained"] if k in results]
    labels = ["Depth 10", "Depth 14", "Depth 18", "Depth 22", "Depth 26", "Unconstrained\n(Depth ~49)"]
    rmses = [results[k]["rmse"] for k in depth_keys]
    maes = [results[k]["mae"] for k in depth_keys]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("#ffffff")

    rects1 = ax1.bar(x - width/2, rmses, width, label="Test RMSE (%)", color="#3b82f6", edgecolor="#1d4ed8", alpha=0.9)
    rects2 = ax1.bar(x + width/2, maes, width, label="Test MAE (%)", color="#10b981", edgecolor="#047857", alpha=0.9)

    # Highlight optimal bar
    opt_idx = depth_keys.index(best_depth_key)
    rects1[opt_idx].set_color("#f59e0b")
    rects1[opt_idx].set_edgecolor("#b45309")
    rects2[opt_idx].set_color("#f97316")
    rects2[opt_idx].set_edgecolor("#c2410c")

    ax1.set_ylabel("Error Metric (%)", fontsize=11, fontweight="bold", labelpad=8)
    ax1.set_title("Random Forest Performance Across Pruned Tree Depths (50 Trees)", fontsize=13, fontweight="bold", pad=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9.5, fontweight="500")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.set_ylim(0, max(rmses) * 1.25)

    # Value labels
    for bar in rects1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.08, f"{yval:.2f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar in rects2:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.08, f"{yval:.2f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax1.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_four_stage_evolution(four_stages: dict, output_path: str):
    """
    Visualizes the complete 4-Stage Research Journey:
    Stage 1 (Baseline 4F) -> Stage 2 (Powertrain 7F) -> Stage 3 (SOTA Temporal 17F) -> Stage 4 (Pareto Pruned 17F).
    """
    stages = [
        "Stage 1\nBaseline RF\n(4F, 25T)",
        "Stage 2\nPowertrain RF\n(7F, 50T)",
        "Stage 3\nSOTA Temporal\n(17F, 50T, Depth~49)",
        "Stage 4\nPareto-Pruned\n(17F, 50T, Depth 18)",
    ]

    s1 = four_stages["stage_1_baseline_4f"]
    s2 = four_stages["stage_2_enhanced_7f"]
    s3 = four_stages["stage_3_temporal_unconstrained"]
    s4 = four_stages["stage_4_pareto_pruned"]

    rmses = [s1["rmse"], s2["rmse"], s3["rmse"], s4["rmse"]]
    max_errors = [s1["max_error"], s2["max_error"], s3["max_error"], s4["max_error"]]
    nodes_k = [s1["nodes"]/1000, s2["nodes"]/1000, s3["nodes"]/1000, s4["nodes"]/1000]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor("#ffffff")

    # Panel 1: Error Evolution
    x = np.arange(len(stages))
    width = 0.35
    ax1.bar(x - width/2, rmses, width, label="Test RMSE (%)", color=["#94a3b8", "#60a5fa", "#3b82f6", "#10b981"], edgecolor="#1e293b", alpha=0.9)
    ax1.bar(x + width/2, max_errors, width, label="Peak MAX Error (%)", color=["#fca5a5", "#f87171", "#ef4444", "#f59e0b"], edgecolor="#7f1d1d", alpha=0.9)
    ax1.set_ylabel("Error Metric (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Error Metric Evolution Across 4 Architectural Stages", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontsize=8.5, fontweight="600")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for i, v in enumerate(rmses):
        ax1.text(i - width/2, v + 0.5, f"{v:.2f}%", ha="center", fontsize=8.5, fontweight="bold")
    for i, v in enumerate(max_errors):
        ax1.text(i + width/2, v + 0.5, f"{v:.1f}%", ha="center", fontsize=8.5, fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.9)

    # Panel 2: Complexity Evolution (Total Nodes in thousands)
    colors_node = ["#94a3b8", "#64748b", "#dc2626", "#059669"]
    bars = ax2.bar(stages, nodes_k, color=colors_node, edgecolor="#0f172a", width=0.55, alpha=0.9)
    ax2.set_ylabel("Total Tree Nodes in Forest (Thousands)", fontsize=11, fontweight="bold")
    ax2.set_title("Model Complexity & Memory Footprint Evolution", fontsize=12, fontweight="bold", pad=12)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 150, f"{h:,.0f}k\n({(h*12/1024):.1f} MB)", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_soc_estimation_pruned(y_test: np.ndarray, y_pred: np.ndarray, output_path: str):
    """Actual vs Predicted SOC trajectory on unseen test set for optimal pruned model."""
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("#ffffff")

    sample_step = max(1, len(y_test) // 2500)
    idx = np.arange(0, len(y_test), sample_step)

    ax.plot(idx, y_test[idx], color="#0f172a", linewidth=1.8, label="Actual SOC (BMS Ground Truth)", alpha=0.85)
    ax.plot(idx, y_pred[idx], color="#059669", linewidth=1.5, linestyle="--", label="Pareto-Pruned RF Predicted SOC", alpha=0.95)

    ax.set_xlabel("Test Telemetry Index (Unseen Trips B29 to B38)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("State of Charge (%)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("Pareto-Pruned Random Forest: Actual vs. Predicted SOC (118,974 Unseen Samples)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.set_ylim(-2, 102)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

