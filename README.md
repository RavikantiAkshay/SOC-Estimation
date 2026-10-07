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

### 🏆 Baseline Benchmarking: Random Forest vs. Baselines

To rigorously evaluate the necessity and advantages of the ensemble Random Forest, two distinct baselines were implemented under an identical experimental setup (same 60-trip training split of 945,026 instances and 10-trip test split of 118,974 instances):
1. **Decision Tree (Single Tree)**: Evaluates a single decision tree with identical tree regularization (`max_features='sqrt'`, `min_samples_leaf=5`) across 5 random seeds to measure the variance reduction and stabilization of ensemble bagging.
2. **K-Nearest Neighbors ($k=5$, Distance-Weighted)**: An instance-based non-parametric baseline operating with `StandardScaler` feature normalization and KD-Tree spatial indexing.

#### Head-to-Head Comparison on Unseen Driving Cycles (10 Trips, 118,974 Test Samples)

| Metric | Random Forest (25 Trees) | K-Nearest Neighbors ($k=5$) | Decision Tree (Single Tree) | RF Improvement vs. KNN | RF Improvement vs. DT |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RMSE** | **`5.8876%`** | `7.0826%` | `7.0223%` | **16.87% lower error** | **16.16% lower error** |
| **MAE** | **`4.3736%`** | `5.4040%` | `5.2075%` | **19.07% lower error** | **16.01% lower error** |
| **Max Error** | **`26.6956%`** | `29.1223%` | `41.7567%` | **8.33% lower error** | **36.07% lower error** |
| **Std Dev** | **`5.8876%`** | `7.0668%` | `7.0162%` | **16.69% lower variance** | **16.09% lower variance** |

#### BMS Engineering Takeaways: Why Random Forest Wins

1. **Superior Predictive Accuracy**:
   - Random Forest cuts root mean squared error by **`16.87%`** and mean absolute error by **`19.07%`** compared to KNN.
   - It cuts RMSE by **`16.16%`** and MAE by **`16.01%`** compared to a single Decision Tree.
2. **36% Reduction in Catastrophic Peak Error (vs. Decision Tree)**:
   - A single Decision Tree suffered a dangerous worst-case residual error of **`41.76%`** during sudden acceleration and regenerative dynamics. In a physical vehicle, this error magnitude could trigger premature shutdown or over-discharge.
   - By aggregating 25 decorrelated trees, Random Forest compresses maximum error to **`26.70%`** (a **`36.07%` reduction**).
3. **Automotive Microcontroller Constraints (vs. KNN)**:
   - **Memory Bottleneck**: KNN requires storing all 945,026 training vectors (~15+ MB raw float data plus tree indexing structures) in RAM. Automotive ECUs (e.g., Infineon AURIX, TI TMS570) typically feature only a few megabytes of total RAM. Random Forest requires zero training vector retention.
   - **Inference Latency**: KNN computes Euclidean distance searches across multi-dimensional partitions for every single prediction cycle. Random Forest traverses lightweight binary comparison trees (`if x <= threshold`) within microseconds.

---

## 📂 Project Structure

```
EV_HEV/
├── app/                             # Interactive BMS Digital Twin Dashboard
│   ├── server.py                    # Lightweight HTTP server & ML inference API
│   ├── index.html                   # Instrument cluster single-page UI
│   ├── style.css                    # Design system (ceramic & deep pine palette)
│   └── app.js                       # 60fps trip replayer & Canvas charting engine
├── archive/                         # Raw dataset (70 trip CSV files + metadata)
│   ├── TripA01.csv ... TripA32.csv
│   └── TripB01.csv ... TripB38.csv
├── Random_Forest/                   # Main proposed ensemble architecture
│   ├── results/
│   │   ├── models/
│   │   │   └── rf_25_trees_final.joblib # Serialized best model (43.2 MB)
│   │   ├── plots/                   # 7 publication-ready diagnostic charts
│   │   └── results.json             # Metrics for all tree configurations & seeds
│   └── src/
│       ├── config.py                # Hyperparameters, paths, and column standards
│       ├── data_loader.py           # Robust ingestion, cleaning, and train/test split
│       ├── evaluate.py              # Regression metrics (RMSE, MAE, Max Error, Std Dev)
│       ├── model_rf.py              # Multi-seed training and model selection
│       ├── plots.py                 # Figure generation suite
│       └── main.py                  # End-to-end execution pipeline
├── Decision_Tree/                   # Baseline 1: Single decision tree
│   ├── results/
│   │   ├── models/                  # Serialized single tree model
│   │   ├── plots/                   # Single tree diagnostic & comparison plots
│   │   └── results.json             # Single tree performance metrics & RF comparison
│   ├── src/
│   │   ├── config.py, model_dt.py, plots_dt.py, main.py
│   └── README.md                    # Detailed Decision Tree documentation
├── KNN/                             # Baseline 2: K-Nearest Neighbors
│   ├── results/
│   │   ├── models/                  # Serialized scaler + KNN pipeline
│   │   ├── plots/                   # KNN diagnostic & comparison plots
│   │   └── results.json             # KNN performance metrics & RF comparison
│   ├── src/
│   │   ├── config.py, model_knn.py, plots_knn.py, main.py
│   └── README.md                    # Detailed KNN baseline documentation
└── README.md                        # Master repository documentation
```

### Module Responsibilities

- **`app/`**: Standalone interactive BMS digital twin web dashboard. Features live trip telemetry playback across test routes (TripB29 to TripB38), real-time side-by-side inference against Random Forest, Decision Tree, and KNN, and an interactive What-If scenario sandbox.
- **`Random_Forest/`**: Contains the full implementation of the proposed 25-tree ensemble model, including hyperparameter grid exploration (25–100 trees), multi-seed evaluation, and extensive diagnostic visualizations.
- **`Decision_Tree/`**: Isolates the single-tree baseline to demonstrate the empirical benefits of ensemble variance reduction and bagging.
- **`KNN/`**: Implements standardized feature scaling and instance-based nearest-neighbor regression to benchmark against distance-based methods and analyze onboard ECU computational feasibility.

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

### Running the Baselines

Execute individual isolated baselines to reproduce the comparative benchmark:

- **Decision Tree (Single Tree)**:
  ```bash
  python Decision_Tree/src/main.py
  ```
- **K-Nearest Neighbors (KNN)**:
  ```bash
  python KNN/src/main.py
  ```

### Launching the Interactive Web Dashboard

To run the real-time BMS digital twin dashboard and trip replayer:
```bash
python app/server.py
```
Then open [`http://localhost:8000`](http://localhost:8000) in your browser.

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
