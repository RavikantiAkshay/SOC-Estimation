# K-Nearest Neighbors (KNN) Baseline for SOC Estimation

This directory contains the independent, modular implementation of the **K-Nearest Neighbors (KNN)** baseline for State of Charge (SOC) estimation in Electric Vehicles.

---

## 🎯 Baseline Objectives

1. **Instance-Based Benchmark**: Compare Random Forest against a non-parametric, distance-based lazy learner.
2. **Feature Scaling Requirement**: Distance metrics ($L_2$ Euclidean norm) require normalized inputs ($V, I, T_{\text{batt}}, T_{\text{amb}}$) to prevent large-magnitude features (pack voltage ~350V) from dominating lower-magnitude features (temperatures ~20°C).
3. **BMS Feasibility Analysis**: Evaluate real-time execution feasibility on automotive electronic control units (ECUs).

---

## ⚙️ Configuration & Hyperparameters

- **Neighbors ($k$)**: `5`
- **Weighting**: Inverse distance (`weights='distance'`) to reward nearby states
- **Algorithm**: `KD-Tree` (high-dimensional spatial indexing)
- **Leaf Size**: `30`
- **Feature Preprocessing**: `StandardScaler` fitted exclusively on training set

---

## 📊 Benchmark Results (Head-to-Head Comparison)

Tested on **118,974 unseen test instances** across **10 real-world EV trips**:

| Metric | Random Forest (25 Trees) | K-Nearest Neighbors ($k=5$) | Decision Tree (Single Tree) | RF Improvement vs. KNN |
| :--- | :---: | :---: | :---: | :---: |
| **RMSE** | **`5.8876%`** | `7.0826%` | `7.0223%` | **RF is 16.87% better** |
| **MAE** | **`4.3736%`** | `5.4040%` | `5.2075%` | **RF is 19.07% better** |
| **Max Error** | **`26.6956%`** | `29.1223%` | `41.7567%` | **RF is 8.33% better** |
| **Std Dev** | **`5.8876%`** | `7.0668%` | `7.0162%` | **RF is 16.69% better** |

---

## 🔬 BMS Engineering Perspective: Why Random Forest Wins

While KNN achieves acceptable accuracy on static interpolations, it is **unsuitable for embedded Battery Management Systems (BMS)** for two primary reasons:

1. **Memory Footprint**:
   - **KNN**: Stores all 945,026 training vectors (4 features $\times$ 4 bytes $\approx$ 15+ MB raw float data plus tree indexing structures) in RAM. Automotive microcontrollers (e.g. Infineon AURIX, TI TMS570) typically feature only a few megabytes of flash and RAM.
   - **Random Forest**: Stores fixed, compact decision split rules (43.2 MB for 25 trees or <10 MB if pruned/quantized) requiring no raw sample storage.
2. **Inference Latency**:
   - **KNN**: Must perform spatial neighborhood search across thousands of partitions for every single time step ($1.18$s indexing, $2.04$s inference).
   - **Random Forest**: Evaluates simple binary conditional branches (`if x[0] <= threshold`) with predictable $O(\text{depth} \times \text{trees})$ execution time, executing within microseconds.

---

## 🚀 How to Run

To train, evaluate, and generate diagnostic plots for the KNN baseline:

```bash
python KNN/src/main.py
```

Generated artifacts:
- Model & Scaler artifact: `KNN/results/models/knn_model.joblib`
- Numerical metrics & comparisons: `KNN/results/results.json`
- Visualization figures: `KNN/results/plots/`
