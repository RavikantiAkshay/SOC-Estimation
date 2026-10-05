# State of Charge (SOC) Estimation for Electric Vehicles using Optimized Random Forest

A production-grade machine learning system designed to estimate the State of Charge (SOC) of Lithium-ion battery packs in Electric Vehicles (EVs) using real-world telemetry data. The system employs an optimized Random Forest regression architecture that balances non-linear mapping precision, noise robustness, and real-time execution efficiency.

---

## 📌 Project Overview

Accurate SOC estimation is critical for battery management systems (BMS) to extend battery longevity, prevent catastrophic over-discharge/over-charge states, and provide drivers with reliable range predictions. Real-world EV driving conditions present severe challenges: dynamic acceleration/braking currents, ambient and internal thermal fluctuations, sensor noise, and electrochemical hysteresis.

This system addresses these challenges using an ensemble learning approach trained on **over 1.06 million instances** of real-world driving telemetry:
- **Best Test RMSE**: **`5.8876%`**
- **Best Test MAE**: **`4.3736%`**
- **Optimal Model Size**: **25 trees (43.2 MB)**
- **Training Time**: **~19.9s** (fast turnaround, minimal latency)

---

## 🏗️ System Architecture & Workflow

```
                                 ┌─────────────────────────────────┐
                                 │   70 Real-World EV Trip CSVs    │
                                 │       (>1,064,000 instances)     │
                                 └────────────────┬────────────────┘
                                                  │
                                                  ▼
                                 ┌─────────────────────────────────┐
                                 │  Data Cleaning & Preprocessing  │
                                 │  - Header Standardisation       │
                                 │  - Missing Value Removal        │
                                 │  - 4-Feature Extraction         │
                                 └────────────────┬────────────────┘
                                                  │
                         ┌────────────────────────┴────────────────────────┐
                         ▼                                                 ▼
          ┌─────────────────────────────┐                   ┌─────────────────────────────┐
          │   Training Set (60 Trips)   │                   │    Testing Set (10 Trips)   │
          │      945,026 instances      │                   │      118,974 instances      │
          └──────────────┬──────────────┘                   └──────────────┬──────────────┘
                         │                                                 │
                         ▼                                                 │
          ┌─────────────────────────────┐                                  │
          │   Random Forest Training    │                                  │
          │  - Tree counts: 25..100     │                                  │
          │  - Feature subset: sqrt     │                                  │
          │  - Leaf constraint: 5       │                                  │
          │  - 5 repeats per config     │                                  │
          └──────────────┬──────────────┘                                  │
                         │                                                 │
                         └────────────────────────┬────────────────────────┘
                                                  ▼
                                 ┌─────────────────────────────────┐
                                 │     Evaluation & Model Selection│
                                 │  - RMSE, MAE, MAX, STD DEV      │
                                 │  - Optimal: 25 Trees Selected   │
                                 └────────────────┬────────────────┘
                                                  │
                         ┌────────────────────────┴────────────────────────┐
                         ▼                                                 ▼
          ┌─────────────────────────────┐                   ┌─────────────────────────────┐
          │     Saved Model & Metrics   │                   │    Visualisation Suite      │
          │  - rf_25_trees_final.joblib │                   │  - Actual vs Predicted Curve│
          │  - results.json             │                   │  - Prediction Error Profile │
          └─────────────────────────────┘                   │  - Error by SOC Range       │
                                                            │  - Per-Trip Error Breakdown │
                                                            │  - Tree Count Comparisons   │
                                                            │  - Feature Importance       │
                                                            │  - Error Distribution       │
                                                            └─────────────────────────────┘
```

### Key Engineering Decisions

1. **Feature Subsampling per Split (`max_features='sqrt'`)**:
   Instead of considering all 4 telemetry features at every node split, each split randomly samples $\sqrt{4} = 2$ features. This de-correlates the individual trees in the forest, preventing voltage dominance from overshadowing temperature and current dynamics.
2. **Leaf Node Regularization (`min_samples_leaf=5`)**:
   Enforcing a minimum of 5 samples per terminal leaf node prevents individual decision trees from fitting to high-frequency sensor anomalies and transient current spikes.
3. **Statistical Multi-Run Validation**:
   Every tree configuration is trained and evaluated across 5 independent runs with distinct random seeds. Both best-run and ensemble-average metrics are recorded to verify model stability.

---

## 📊 Dataset & Features

The model is built on telemetry collected from a **BMW i3 electric vehicle equipped with a 60 Ah Lithium-ion battery pack** across a diverse spectrum of driving routes, driver behaviors, and weather environments.

| Dataset Split | Trip Files | Instance Count | Purpose |
|:---|:---:|:---:|:---|
| **Training Set** | Trips 1 to 60 (`TripA01` to `TripB28`) | **945,026** | Model training & hyperparameter calibration |
| **Testing Set** | Trips 61 to 70 (`TripB29` to `TripB38`) | **118,974** | Final unbiased performance assessment |

### Input Features & Target Variable

| Attribute | Units | Description |
|:---|:---:|:---|
| **Battery Voltage** | V | Total terminal voltage of the battery pack |
| **Battery Current** | A | Instantaneous current (positive = discharging, negative = regenerative braking) |
| **Battery Temperature** | °C | Internal temperature monitored by battery pack sensors |
| **Ambient Temperature** | °C | External environmental temperature |
| **Target: SoC** | % | Manufacturer-estimated State of Charge |

---

## 📈 Performance & Results

### Tree Count Comparison (5 Runs Each on Unseen Test Data)

| Trees | Best RMSE (%) | Average RMSE (%) | Best MAE (%) | Average MAE (%) | Best MAX Error (%) | Training Time |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **25 (Selected)** | **5.8876%** | **5.9886%** | **4.3736%** | **4.4725%** | **26.6956%** | **19.9s** |
| 50 | 5.9166% | 5.9855% | 4.3922% | 4.4676% | 26.4759% | 42.6s |
| 75 | 5.9095% | 5.9833% | 4.4134% | 4.4700% | 24.0654% | 66.2s |
| 100 | 5.8979% | 5.9619% | 4.3853% | 4.4478% | 24.0784% | 88.7s |

### Why 25 Trees Was Selected as the Optimal Architecture

1. **Lowest Prediction Error**: 25 trees achieved the lowest Best RMSE (**5.8876%**) and lowest Best MAE (**4.3736%**) across all tested configurations.
2. **Computational Efficiency**: Training takes only **19.9 seconds** for 25 trees compared to **88.7 seconds** for 100 trees (over 4× faster).
3. **Embedded Footprint**: The serialized model is **43.2 MB**, making it suitable for onboard edge deployment in EV control units, whereas 100-tree models consume hundreds of megabytes with negligible accuracy difference.

---

## 📂 Project Structure

```
EV_HEV/
├── archive/                         # Raw dataset (70 trip CSV files + metadata)
│   ├── TripA01.csv ... TripA32.csv
│   └── TripB01.csv ... TripB38.csv
├── Random_Forest/
│   ├── results/
│   │   ├── models/
│   │   │   └── rf_25_trees_final.joblib # Serialised final best model (43.2 MB)
│   │   ├── plots/
│   │   │   ├── soc_estimation_rf.png    # Actual vs Predicted SOC curves
│   │   │   ├── prediction_error_rf.png  # Error residuals over all test instances
│   │   │   ├── feature_importance_rf.png# Gini importance of the 4 input features
│   │   │   ├── tree_count_comparison.png# Comparative analysis of tree configurations
│   │   │   ├── error_by_soc_range.png   # Performance across SOC operational intervals
│   │   │   ├── error_distribution.png   # Histogram & KDE of prediction errors
│   │   │   └── per_trip_error.png       # Individual error breakdown across 10 test trips
│   │   └── results.json                 # Complete numerical metrics for all runs
│   └── src/
│       ├── config.py                    # Central configuration, file paths, and hyperparameters
│       ├── data_loader.py               # Robust CSV ingestion, header cleaning, and 60/10 split
│       ├── evaluate.py                  # Evaluation metrics (RMSE, MAE, MAX ERROR, STD DEV)
│       ├── model_rf.py                  # Training pipeline, multi-seed evaluation, and model extraction
│       ├── plots.py                     # High-resolution figure generation
│       └── main.py                      # Orchestrator script executing the full workflow
└── README.md                        # Project documentation
```

### Module Responsibilities

- **`Random_Forest/src/config.py`**: Declares all filesystem paths, standardized column names, feature lists, train/test trip split ratios, and default Random Forest parameters (`criterion`, `max_features`, `min_samples_leaf`, `n_jobs`).
- **`Random_Forest/src/data_loader.py`**: Handles parsing of semicolon-separated, Latin-1 encoded CSV files. Resolves irregular column name encodings, filters missing/NaN values, and partitions trips into 60 training and 10 testing DataFrames.
- **`Random_Forest/src/evaluate.py`**: Computes standard regression metrics (RMSE, MAE, Maximum Error, and Error Standard Deviation) and provides clean console formatting.
- **`Random_Forest/src/model_rf.py`**: Manages model instantiation via scikit-learn, executes repeated training runs with varying seeds, and extracts optimal model weights.
- **`Random_Forest/src/plots.py`**: Generates all evaluation figures using Matplotlib with properly scaled axes, clear labels, and publication-ready formatting.
- **`Random_Forest/src/main.py`**: End-to-end driver that coordinates data loading, model exploration, metric comparison, artifact persistence, and plot rendering.

---

## 🖼️ Diagnostic Visualizations

All generated plots are saved to [`Random_Forest/results/plots/`](file:///e:/EV_HEV/Random_Forest/results/plots/):

1. **SOC Estimation Curve (`soc_estimation_rf.png`)**: Tracks actual vs. predicted SOC across 118,974 test instances, demonstrating tight tracking throughout diverse driving cycles.
2. **Prediction Error Profile (`prediction_error_rf.png`)**: Visualizes instantaneous residuals ($y_{true} - y_{pred}$) across test duration.
3. **Tree Count Comparison (`tree_count_comparison.png`)**: Two-panel comparative bar chart (scaled Y-axes) highlighting Best and Average RMSE/MAE across 25, 50, 75, and 100 trees with exact values labeled on every bar.
4. **Feature Importance (`feature_importance_rf.png`)**: Ranks the relative predictive contribution of Battery Voltage, Battery Current, Battery Temperature, and Ambient Temperature.
5. **Error by SOC Operational Range (`error_by_soc_range.png`)**: Two-panel analysis showing binned RMSE/MAE and residual boxplots across discrete battery capacity intervals (0–20%, 20–40%, 40–60%, 60–80%, 80–100%).
6. **Error Distribution (`error_distribution.png`)**: Histogram and zero/mean error lines showing normal residual concentration centered near 0% error.
7. **Per-Trip Error Breakdown (`per_trip_error.png`)**: Evaluates individual RMSE and MAE across each of the 10 unseen test trips (TripB29 to TripB38), proving cross-trip generalization under diverse driving conditions.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Recommended virtual environment:
  ```bash
  python -m venv venv
  # Windows
  .\venv\Scripts\activate
  # Linux / macOS
  source venv/bin/activate
  ```

### Installation

Install required dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib joblib
```

### Running the Complete Pipeline

Execute the orchestrator script from the project root:
```bash
python -u Random_Forest/src/main.py
```

The script will:
1. Load and clean all 70 trip CSV files.
2. Train and evaluate 5 independent runs for each tree size (25, 50, 75, 100 trees).
3. Identify the optimal configuration (25 trees).
4. Export the final model to `Random_Forest/results/models/rf_25_trees_final.joblib`.
5. Write all performance metrics to `Random_Forest/results/results.json`.
6. Render and save all 7 diagnostic charts into `Random_Forest/results/plots/`.


---

## 📜 Attribution & License

### 1. Research Methodology Attribution
This project builds upon the methodology and benchmarking established in the peer-reviewed literature:
- **Title**: *"State of charge estimation for electric vehicles using random forest"*
- **Authors**: Mohd Herwan Sulaiman, Zuriani Mustaffa (*Universiti Malaysia Pahang Al-Sultan Abdullah - UMPSA*)
- **Journal**: *Green Energy and Intelligent Transportation*, Vol. 3, Issue 3, 2024, 100177
- **DOI**: [10.1016/j.geits.2024.100177](https://doi.org/10.1016/j.geits.2024.100177)
- **License**: **[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)**
- *Notice*: This repository provides an independent open-source reproduction and software implementation of the methods and experiments described in the paper.

### 2. Dataset Attribution
The telemetry used for training and testing is from the BMW i3 real-world driving dataset:
- **Dataset**: *"Battery and heating data in real driving cycles"* (70 trips, 1.06M+ records)
- **Authors**: M. S. J. B. D. Trifonov et al.
- **Identifier / DOI**: [10.21227/6jr9-5235](https://doi.org/10.21227/6jr9-5235)
- **License**: **[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)**
- *Usage & Acquisition*: Under CC BY 4.0, raw telemetry files (`TripA01.csv`–`TripA32.csv` and `TripB01.csv`–`TripB38.csv`) are excluded from this repository to adhere to repository size best practices. Download the raw CSV files from the original repository or [Kaggle](https://www.kaggle.com/datasets) and place them in the [`archive/`](file:///e:/EV_HEV/archive/) directory before running `src/main.py`.

### 3. Software License
The original code and scripts in this repository are licensed under the **[MIT License](LICENSE)**.
