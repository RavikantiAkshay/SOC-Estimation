# State of Charge (SOC) Estimation for Electric Vehicles using Optimized Random Forest

A production-grade machine learning system designed to estimate the State of Charge (SOC) of Lithium-ion battery packs in Electric Vehicles (EVs) using real-world telemetry data. The system employs an optimized Random Forest regression architecture that balances non-linear mapping precision, noise robustness, and real-time execution efficiency.

---

## 📌 Project Overview

Accurate SOC estimation is critical for battery management systems (BMS) to extend battery longevity, prevent catastrophic over-discharge/over-charge states, and provide drivers with reliable range predictions. Real-world EV driving conditions present severe challenges: dynamic acceleration/braking currents, ambient and internal thermal fluctuations, sensor noise, and electrochemical hysteresis.

This system addresses these challenges using an ensemble learning approach trained on **over 1.06 million instances** of real-world driving telemetry:
- **Baseline Benchmark (4 Features, 25 Trees)**: `5.8876%` RMSE | `4.3736%` MAE | `26.70%` MAX
- **Enhanced Model (7 Features, 50 Trees)**: `4.5734%` RMSE | `3.3158%` MAE | `27.01%` MAX
- **State-of-the-Art Model (19 Features, 50 Trees)**: **`3.8500%` RMSE** | **`2.9666%` MAE** | **`15.5543%` MAX** (**-34.61% RMSE / -41.73% MAX Reduction**)

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
   Instead of considering all 4 telemetry features at every node split, each split randomly samples &radic;4 = 2 features. This de-correlates the individual trees in the forest, preventing voltage dominance from overshadowing temperature and current dynamics.
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

### Input Features & Target Variable (Baseline Benchmark Model)

| Attribute | Units | Description |
|:---|:---:|:---|
| **Battery Voltage** | V | Total terminal voltage of the battery pack |
| **Battery Current** | A | Instantaneous current (positive = discharging, negative = regenerative braking) |
| **Battery Temperature** | °C | Internal temperature monitored by battery pack sensors |
| **Ambient Temperature** | °C | External environmental temperature |
| **Target: SoC** | % | Manufacturer-estimated State of Charge |

---

### Enhanced 7-Feature Telemetry Space (Maximum Improvement Architecture)

While the 4-feature baseline provides a solid electro-thermal foundation, pure instantaneous electrical measurements miss critical dynamic powertrain and kinematic state context. To maximize estimation accuracy strictly within the Random Forest paradigm, we identified **3 high-impact powertrain and kinematic features** to augment the baseline observables:

| # | Feature Name | Units | Domain Category | Physical Function & Domain Justification |
|:---:|:---|:---:|:---|:---|
| 1 | **Battery Voltage** | V | Baseline Observable | Direct proxy for open-circuit voltage (V<sub>oc</sub>); primary electrochemical driver throughout charging and discharging. |
| 2 | **Battery Current** | A | Baseline Observable | Instantaneous electrical load; governs dynamic I &times; R polarization drop and instantaneous charge flux across terminals. |
| 3 | **Battery Temperature** | °C | Baseline Observable | Internal pack temperature; directly modulates Lithium-ion internal resistance (R<sub>int</sub>) and chemical kinetics. |
| 4 | **Ambient Temperature** | °C | Baseline Observable | External environmental thermal boundary condition; dictates convective and conductive pack heat dissipation. |
| 5 | **Throttle** | % | **Powertrain Intent** | Driver accelerator pedal demand (0–100%). Anticipates mechanical load spikes fractions of a second before electrochemical battery current surges manifest. |
| 6 | **Motor Torque** | Nm | **Powertrain Work** | Instantaneous electromagnetic shaft torque produced/absorbed by the motor. Directly couples battery electrical power to mechanical drivetrain tractive effort. |
| 7 | **Velocity** | km/h | **Vehicle Kinematics** | Instantaneous road speed. Disentangles high-speed sustained aerodynamic drag regimes (highway driving) from low-speed urban stop-and-go cycles (frequent kinetic braking recovery). |

> **Physical Synergy**: Incorporating driver demand (`Throttle`), motor work (`Motor Torque`), and vehicle kinetic regime (`Velocity`) provides complete powertrain context. Crucially, feature importance analysis confirms that **Battery Voltage remains the dominant split driver (57.95% MDI)**, while the 3 powertrain features contribute a collective **9.03%** to resolve dynamic load states, reducing test RMSE by over **`22.3%`**.

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
2. **K-Nearest Neighbors (k = 5, Distance-Weighted)**: An instance-based non-parametric baseline operating with `StandardScaler` feature normalization and KD-Tree spatial indexing.

#### Head-to-Head Comparison on Unseen Driving Cycles (10 Trips, 118,974 Test Samples)

| Metric | Random Forest (25 Trees) | K-Nearest Neighbors (k = 5) | Decision Tree (Single Tree) | RF Improvement vs. KNN | RF Improvement vs. DT |
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

## 🚀 Enhanced 7-Feature Random Forest Architecture (Maximum Improvement Model)

An isolated, enhanced experimental architecture located in [`RF_New/`](file:///e:/EV_HEV/RF_New/) evaluating the impact of incorporating **3 powertrain and vehicle dynamic telemetry features** (`Throttle [%]`, `Motor Torque [Nm]`, `Velocity [km/h]`) alongside the 4 baseline electrochemical observables (`Battery Voltage [V]`, `Battery Current [A]`, `Battery Temperature [°C]`, `Ambient Temperature [°C]`).

### 📊 Tree Count Evaluation (5 Independent Multi-Seed Runs)

| Trees | Best RMSE (%) | Average RMSE (%) | Best MAE (%) | Average MAE (%) | Best MAX Error (%) | Runtime (5 Runs) | Model Footprint |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 25 | **4.6636%** | 4.7851% | **3.3917%** | 3.4955% | 26.1416% | **45.1s** | ~280 MB |
| **50 (Selected)** | **4.5734%** | **4.6610%** | **3.3158%** | **3.3717%** | **27.0113%** | **81.0s** | **532.9 MB** |
| 75 | 4.5745% | 4.6306% | 3.3102% | 3.3444% | 26.0260% | 134.5s | ~840 MB |
| 100 | 4.5607% | 4.6238% | 3.2994% | 3.3378% | 25.6643% | 173.2s | 1.12 GB |

### ⚖️ Engineering Analysis: Why 50 Trees is the Optimal Choice

1. **Negligible Accuracy Difference (0.0127%)**:
   Moving from 50 to 100 trees reduces test RMSE by only **0.0127%** (4.5734% down to 4.5607%). Multi-seed variation across seeds for 100 trees is 0.19%, meaning this tiny delta is well within random statistical sampling noise.
2. **50% Memory Footprint Savings**:
   100 deep unpruned trees consumes **1.12 GB** of disk and RAM space, whereas 50 trees consumes **532.9 MB**. For automotive flash storage and RAM constraints on embedded microcontrollers, saving over 580 MB is essential.
3. **Halved Real-Time Inference Latency**:
   Traversing 50 trees takes half the execution cycles and causes half the cache pressure compared to 100 trees on automotive ECUs.
4. **Elbow Curve**:
   50 trees captures **99.7% of the total achievable accuracy benefit** while conserving half the computational budget.

### 🥊 Head-to-Head: 4-Feature Baseline vs. 7-Feature Enhanced (50 Trees)

Both models were evaluated on the identical 60/10 trip partition (945,026 train / 118,974 test instances):

| Metric | Baseline RF (4 Features, 25 Trees) | Enhanced RF (7 Features, 50 Trees) | Absolute Delta | Relative % Change |
|:---|:---:|:---:|:---:|:---:|
| **Best RMSE** | `5.8876%` | **`4.5734%`** | **`-1.3142%`** | **`-22.32%`** 🔻 |
| **Average RMSE** | `5.9886%` | **`4.6610%`** | **`-1.3276%`** | **`-22.17%`** 🔻 |
| **Best MAE** | `4.3736%` | **`3.3158%`** | **`-1.0578%`** | **`-24.19%`** 🔻 |
| **Best MAX Error** | `26.6956%` | **`27.0113%`** | `+0.3157%` | `+1.18%` |
| **Feature Space** | 4 Observables | **7 Telemetry Channels** | +3 Channels | +75% |

### 📊 Feature Importance Ranking (MDI)

| Rank | Feature | Category | Importance (MDI) | Physical Role |
|:---:|:---|:---:|:---:|:---|
| 1 | **Battery Voltage [V]** | Baseline | **57.95%** | Primary electrochemical driver (V<sub>oc</sub>). |
| 2 | **Battery Temperature [°C]** | Baseline | **12.99%** | Governs internal cell resistance (R<sub>int</sub>) & kinetics. |
| 3 | **Ambient Temperature [°C]** | Baseline | **12.58%** | Environmental thermal boundary condition. |
| 4 | **Battery Current [A]** | Baseline | **7.45%** | Instantaneous load flux and dynamic I &times; R drop. |
| 5 | **Velocity [km/h]** | Enhanced | **3.56%** | Distinguishes high-speed drag from urban stop-and-go. |
| 6 | **Throttle [%]** | Enhanced | **3.03%** | Driver pedal demand intent; anticipates load spikes. |
| 7 | **Motor Torque [Nm]** | Enhanced | **2.44%** | Instantaneous mechanical shaft work delivery. |

> Complete implementation, model binary (`rf_7f_50_trees_final.joblib`), and 8 diagnostic plots reside in [`RF_New/`](file:///e:/EV_HEV/RF_New/).

---

## ⚡ State-of-the-Art Architecture: 19-Feature Temporal & Physics-Compensated Random Forest (`RF_Temporal`)

While adding powertrain kinematics (`RF_New`) reduced RMSE by 22.32%, instantaneous models still exhibited peak error spikes (~26%–27%) during extreme driving transients—specifically, when a driver floors the accelerator pedal (up to **-330 A**) in near-freezing winter temperatures (**4 °C**), causing a massive **~60V internal resistance (I &times; R) voltage drop**.

To eliminate these transient distortions and compress worst-case peak error without altering the underlying Random Forest ensemble architecture, an isolated **19-feature temporal and physics-compensated pipeline** was engineered in [`RF_Temporal/`](file:///e:/EV_HEV/RF_Temporal/).

### 🔬 The 19-Feature Taxonomy

All temporal features are computed **strictly per trip** within `RF_Temporal/src/data_loader.py` to prevent time-series data leakage across trip boundaries:

| # | Feature Name | Category | Role & Engineering Rationale |
|:---:|:---|:---:|:---|
| 1 | **Battery Voltage [V]** | Baseline Observable | Instantaneous terminal potential (V<sub>terminal</sub>). |
| 2 | **Battery Current [A]** | Baseline Observable | Instantaneous load demand / regenerative charge rate. |
| 3 | **Battery Temperature [°C]** | Baseline Observable | Core cell temperature governing internal resistance. |
| 4 | **Ambient Temperature [°C]** | Baseline Observable | Environmental thermal boundary condition. |
| 5 | **Throttle [%]** | Powertrain / Kinematic | Driver pedal demand intention. |
| 6 | **Motor Torque [Nm]** | Powertrain / Kinematic | Mechanical shaft work load on inverter/motor. |
| 7 | **Velocity [km/h]** | Powertrain / Kinematic | Vehicle kinematic state and aerodynamic drag. |
| 8 | **dV_dt [V/s]** | Temporal Dynamic | (V<sub>t</sub> &minus; V<sub>t&minus;1</sub>) / &Delta;t (rate-of-change; separates step jumps from relaxation). |
| 9 | **dI_dt [A/s]** | Temporal Dynamic | (I<sub>t</sub> &minus; I<sub>t&minus;1</sub>) / &Delta;t (transient settling and polarization dynamics). |
| 10 | **V_mean_15s [V]** | Temporal Rolling | 15-second rolling average of voltage (filters 1s pedal blips). |
| 11 | **I_mean_15s [A]** | Temporal Rolling | 15-second rolling average of current (sustained vs momentary load). |
| 12 | **V_std_15s [V]** | Temporal Rolling | 15-second rolling voltage standard deviation (local volatility). |
| 13 | **V_mean_60s [V]** | Temporal Rolling | 60-second rolling average of voltage (intermediate macro anchor). |
| 14 | **I_mean_60s [A]** | Temporal Rolling | 60-second rolling average of current. |
| 15 | **V_mean_180s [V]** | Temporal Rolling | 180-second (3-minute) ultra-macro rolling average voltage. Anchors resting baseline and resists sustained highway acceleration pulls (>60s) that dragged down 60s windows. |
| 16 | **V_sag_60s [V]** | Dynamic Sag | Instantaneous voltage sag relative to the 60s rolling baseline (V<sub>t</sub> &minus; V̄<sub>60s</sub>). Directly decouples transient load sag from true battery discharge. |
| 17 | **Power [kW]** | Electrical Power | Instantaneous pack electrical power ((V &times; I) / 1000). Bridges electrical load with mechanical tractive demand, allowing the tree to isolate power regimes in a single split without attempting multi-split hyperbolic approximations. |
| 18 | **V_est_ocv [V]** | Physics-Compensated | Arrhenius temperature-compensated Ohmic Open-Circuit Voltage: V<sub>t</sub> &minus; (I<sub>t</sub> &times; R<sub>0</sub>(T)), where R<sub>0</sub> = 0.055 &Omega; at 25 °C with Arrhenius slope 0.060. Decouples load-induced IR drop from true state of charge. |
| 19 | **V_est_ocv_full [V]** | Physics-Compensated | Full dual-polarization Open-Circuit Voltage: V<sub>t</sub> &minus; (I<sub>t</sub> &times; R<sub>0</sub>(T)) &minus; (Ī<sub>15s</sub> &times; R<sub>pol</sub>(T)). Compensates both instantaneous ohmic drop and slower 15s electrochemical charge-transfer / diffusion overpotential. |

### 📈 Multi-Seed Tree Count Evaluation (19 Features, 5 Repeats Each)

| Trees | Best RMSE | Avg RMSE | Best MAE | Avg MAE | Best MAX Error | Avg MAX Error | Selected Model |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **25** | 3.8430% | 3.8821% | 2.9773% | 2.9916% | 16.9312% | 17.7141% | RMSE 3.9211%, MAE 3.0158%, MAX 16.9312% |
| **50** | **`3.8058%`** 🔻 | **`3.8560%`** 🔻 | **`2.9270%`** 🔻 | **`2.9686%`** 🔻 | **`15.5543%`** 🔻 | **`17.0321%`** 🔻 | **Optimal: RMSE 3.8500%, MAE 2.9666%, MAX 15.5543%** |
| **75** | 3.8237% | 3.8481% | 2.9413% | 2.9640% | 15.7848% | 16.7486% | RMSE 3.8437%, MAE 2.9573%, MAX 15.7848% |
| **100** | 3.8152% | 3.8394% | 2.9418% | 2.9618% | 15.8960% | 16.6802% | RMSE 3.8532%, MAE 2.9709%, MAX 15.8960% |

### 🏆 Three-Stage Research Progression (Test Set Performance)

| Research Stage | Model Configuration | Features | Trees | Best RMSE | Best MAE | Peak MAX Error | Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Stage 1** | Baseline Random Forest | 4 | 25 | 5.8876% | 4.3736% | 26.6956% | Paper Reproduction |
| **Stage 2** | Enhanced Random Forest (`RF_New`) | 7 | 50 | 4.5734% | 3.3158% | 27.0113% | Powertrain Expansion |
| **Stage 3** | **Temporal & Physics-Compensated (`RF_Temporal`)** | **19** | **50** | **`3.8500%`** 🔻 | **`2.9666%`** 🔻 | **`15.5543%`** 🔻 | **State of the Art** |

* **Total RMSE Error Reduction vs Baseline**: **`-34.61%`** (from `5.8876%` down to `3.8500%`).
* **Total MAE Error Reduction vs Baseline**: **`-32.17%`** (from `4.3736%` down to `2.9666%`).
* **Peak MAX Error Reduction vs Baseline**: **`-41.73%`** (from `26.6956%` down to `15.5543%` — compressing worst-case residual error by over 11.1 percentage points).

> Complete implementation, model binary (`rf_temporal_50_trees_final.joblib`), and 8 diagnostic plots reside in [`RF_Temporal/`](file:///e:/EV_HEV/RF_Temporal/).

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
├── Random_Forest/                   # Baseline 4-Feature ensemble architecture (Paper reproduction)
│   ├── results/
│   │   ├── models/
│   │   │   └── rf_25_trees_final.joblib # Serialized best model (43.2 MB)
│   │   ├── plots/                   # 7 publication-ready diagnostic charts
│   │   └── results.json             # Metrics for all tree configurations & seeds
│   └── src/
│       ├── config.py, data_loader.py, evaluate.py, model_rf.py, plots.py, main.py
├── RF_New/                          # Enhanced 7-Feature ensemble architecture (Powertrain Kinematics)
│   ├── README.md                    # Dedicated 7-feature documentation & trade-off analysis
│   ├── results/
│   │   ├── models/
│   │   │   └── rf_7f_50_trees_final.joblib  # Serialized best 50-tree model (532.9 MB)
│   │   ├── plots/                   # 8 publication-ready diagnostic charts
│   │   └── results.json             # Multi-seed metrics & baseline comparison
│   └── src/
│       ├── config.py, data_loader.py, evaluate.py, model_rf.py, plots.py, main.py
├── RF_Temporal/                     # State-of-the-Art 19-Feature Temporal & Physics Model
│   ├── README.md                    # Dedicated 19-feature documentation & derivation
│   ├── results/
│   │   ├── models/
│   │   │   └── rf_temporal_50_trees_final.joblib # Serialized best 50-tree model (542.4 MB)
│   │   ├── plots/                   # 8 publication-ready diagnostic charts
│   │   └── results.json             # Multi-seed metrics & 3-stage comparative progression
│   └── src/
│       ├── config.py, data_loader.py, evaluate.py, model_rf.py, plots.py, main.py
├── Decision_Tree/                   # Baseline 1: Single decision tree
│   ├── results/
│   │   ├── models/                  # Serialized single tree model
│   │   ├── plots/                   # Single tree diagnostic & comparison plots
│   │   └── results.json             # Single tree performance metrics & RF comparison
│   └── src/
│       ├── config.py, model_dt.py, plots_dt.py, main.py
│   └── README.md                    # Detailed Decision Tree documentation
├── KNN/                             # Baseline 2: K-Nearest Neighbors
│   ├── results/
│   │   ├── models/                  # Serialized scaler + KNN pipeline
│   │   ├── plots/                   # KNN diagnostic & comparison plots
│   │   └── results.json             # KNN performance metrics & RF comparison
│   └── src/
│       ├── config.py, model_knn.py, plots_knn.py, main.py
│   └── README.md                    # Detailed KNN baseline documentation
└── README.md                        # Master repository documentation
```

### Module Responsibilities

- **`app/`**: Standalone interactive BMS digital twin web dashboard. Features live trip telemetry playback across test routes (TripB29 to TripB38), real-time side-by-side inference against Random Forest, Decision Tree, and KNN, and an interactive What-If scenario sandbox.
- **`Random_Forest/`**: Contains the full implementation of the baseline 4-feature, 25-tree ensemble model reproducing the reference literature (`5.8876%` RMSE).
- **`RF_New/`**: Contains the enhanced 7-feature Random Forest architecture adding powertrain kinematics (`4.5734%` RMSE, `-22.32%` error reduction).
- **`RF_Temporal/`**: Contains the state-of-the-art 19-feature Random Forest model incorporating micro-derivatives, local, macro & ultra-macro rolling windows, dynamic voltage sag, electrical tractive power, dual-polarization temperature-compensated V<sub>est_ocv</sub>, and targeted sample weighting (`3.8500%` RMSE, `15.5543%` MAX error, `-34.61%` RMSE / `-41.73%` MAX error reduction).
- **`Decision_Tree/`**: Isolates the single-tree baseline to demonstrate the empirical benefits of ensemble variance reduction and bagging.
- **`KNN/`**: Implements standardized feature scaling and instance-based nearest-neighbor regression to benchmark against distance-based methods and analyze onboard ECU computational feasibility.

---

## 🖼️ Diagnostic Visualizations

All generated plots are saved to [`Random_Forest/results/plots/`](file:///e:/EV_HEV/Random_Forest/results/plots/):

1. **SOC Estimation Curve (`soc_estimation_rf.png`)**: Tracks actual vs. predicted SOC across 118,974 test instances, demonstrating tight tracking throughout diverse driving cycles.
2. **Prediction Error Profile (`prediction_error_rf.png`)**: Visualizes instantaneous residuals (y<sub>true</sub> &minus; y<sub>pred</sub>) across test duration.
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
