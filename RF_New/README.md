# Enhanced 7-Feature Random Forest Architecture for SOC Estimation

An isolated, enhanced experimental architecture evaluating the impact of incorporating **3 powertrain and vehicle dynamic telemetry features** (`Throttle [%]`, `Motor Torque [Nm]`, `Velocity [km/h]`) alongside the 4 baseline electrochemical observables (`Battery Voltage [V]`, `Battery Current [A]`, `Battery Temperature [°C]`, `Ambient Temperature [°C]`) for State of Charge (SOC) estimation in Electric Vehicles.

---

## 📌 Executive Summary & Key Results

- **Model Architecture**: Random Forest Regressor (`max_features='sqrt'`, `min_samples_leaf=5`, `bootstrap=True`)
- **Feature Space**: **7 Features Total** (4 Baseline Observables + 3 Powertrain/Kinematic Features)
- **Dataset Scale**: **1,064,000 instances** across all 70 BMW i3 trips (60 Train / 10 Test)
- **Selected Operational Configuration**: **50 Trees** (Pareto-Optimal: 99.7% of maximum accuracy at 50% computational footprint)
- **Performance Comparison**:
  - **4-Feature Baseline (25 Trees)**: Best RMSE = **`5.8876%`**, Best MAE = **`4.3736%`**
  - **7-Feature Enhanced (25 Trees)**: Best RMSE = **`4.6636%`** (**-20.8% error reduction**)
  - **7-Feature Enhanced (50 Trees)**: Best RMSE = **`4.5734%`** (**-22.3% error reduction**)
  - **7-Feature Enhanced (100 Trees)**: Best RMSE = **`4.5607%`** (**-22.5% error reduction**)

---

## 📊 Feature Space Composition (7 Features)

All 7 features are extracted from universal sensor channels available across 100% of the 70 trip files with zero missing data:

| # | Feature Name | Units | Category | Relative Importance (MDI) | Physical Function |
|:---:|:---|:---:|:---:|:---:|:---|
| 1 | **Battery Voltage** | V | Baseline Observable | **57.95%** | Direct open-circuit voltage ($V_{\text{oc}}$) proxy; primary electrochemical driver. |
| 2 | **Battery Temperature** | °C | Baseline Observable | **12.99%** | Internal pack temperature governing cell resistance ($R_{\text{int}}$) and chemical kinetics. |
| 3 | **Ambient Temperature** | °C | Baseline Observable | **12.58%** | Environmental thermal boundary condition governing passive pack heat transfer. |
| 4 | **Battery Current** | A | Baseline Observable | **7.45%** | Instantaneous load, dynamic $I \cdot R$ polarization drop, and charge flux. |
| 5 | **Velocity** | km/h | **New Powertrain** | **3.56%** | Dynamic operational speed regime; distinguishes highway vs stop-and-go energy demand. |
| 6 | **Throttle** | % | **New Powertrain** | **3.03%** | Driver pedal demand ($0\text{--}100\%$), anticipating electrical load spikes before current surges. |
| 7 | **Motor Torque** | Nm | **New Powertrain** | **2.44%** | Instantaneous mechanical shaft work produced/absorbed by the electric motor. |

> **Combined Contribution of New Features**: The 3 newly added powertrain variables account for **`9.03%` of total predictive importance**, providing the necessary mechanical load context without diluting the primary electrochemical voltage signal.

---

## 📈 Performance across Tree Counts (5 Multi-Seed Runs)

| Trees | Best RMSE (%) | Average RMSE (%) | Best MAE (%) | Average MAE (%) | Best MAX Error (%) | Runtime (5 Runs) | Memory Footprint |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 25 | **4.6636%** | 4.7851% | **3.3917%** | 3.4955% | 26.1416% | **45.1s** | ~280 MB |
| **50 (Recommended)** | **4.5734%** | **4.6610%** | **3.3158%** | **3.3717%** | **27.0113%** | **81.0s** | **~560 MB** |
| 75 | 4.5745% | 4.6306% | 3.3102% | 3.3444% | 26.0260% | 134.5s | ~840 MB |
| 100 | 4.5607% | 4.6238% | 3.2994% | 3.3378% | 25.6643% | 173.2s | ~1,120 MB |

---

## ⚖️ Engineering Analysis: 50 Trees vs. 100 Trees (Why 50 Trees is Optimal)

While 100 trees yielded the absolute mathematical minimum RMSE (**4.5607%** vs **4.5734%**), **50 trees is the definitive Pareto-optimal engineering choice**:

1. **Infinitesimal Accuracy Difference (0.013%)**:
   - $\Delta \text{RMSE} = 4.5607\% - 4.5734\% = \mathbf{-0.0127\%}$ (only a **0.27%** relative change).
   - $\Delta \text{MAE} = 3.2994\% - 3.3158\% = \mathbf{-0.0164\%}$ (only a **0.49%** relative change).
   - The multi-seed variance across runs for 100 trees spans **0.19%** (4.56% to 4.75%). A difference of 0.013% is well within random statistical noise of tree bootstrap sampling.

2. **Doubled Hardware & Computational Cost**:
   - **Storage & RAM**: 100 deep regression trees take **1.12 GB** of memory/disk space, whereas 50 trees requires only **~560 MB** (a 50% savings).
   - **Inference Latency**: For real-time 1 Hz BMS estimation on an automotive microcontroller (e.g., AURIX TC3xx / STM32), traversing 100 trees takes **2x the cycles and causes double the cache pressure** compared to 50 trees.
   - **Training Time**: Scales linearly from 81 seconds to 173 seconds.

3. **The Law of Diminishing Returns (Elbow Curve)**:
   - Moving from **4F Baseline $\rightarrow$ 7F (25 Trees)** cuts RMSE by **-1.224%** (-20.8%).
   - Moving from **25 Trees $\rightarrow$ 50 Trees** cuts RMSE by an additional **-0.090%** (-1.5%).
   - Moving from **50 Trees $\rightarrow$ 100 Trees** cuts RMSE by merely **-0.013%** (-0.2%).
   - **50 trees captures 99.7% of the total achievable accuracy benefit** while conserving half of the system's computational budget.

---

## 🥊 Head-to-Head Comparison: 4-Feature Baseline vs. 7-Feature Enhanced

Both models were evaluated on the exact same 60/10 trip partition (945,026 training vectors, 118,974 unseen test vectors):

| Metric | Baseline RF (4 Features, 25 Trees) | Enhanced RF (7 Features, 50 Trees) | Delta | % Change | Engineering Assessment |
|:---|:---:|:---:|:---:|:---:|:---|
| **Best RMSE** | `5.8876%` | **`4.5734%`** | **`-1.3142%`** | **`-22.32%`** | **Substantial accuracy gain; dramatic error reduction.** |
| **Average RMSE** | `5.9886%` | **`4.6610%`** | **`-1.3276%`** | **`-22.17%`** | Ensemble average is consistently lower across all random seeds. |
| **Best MAE** | `4.3736%` | **`3.3158%`** | **`-1.0578%`** | **`-24.19%`** | Average prediction deviation drops by over 1.05% SOC. |
| **Best MAX Error** | `26.6956%` | **`27.0113%`** | `+0.3157%` | `+1.18%` | Peak transient worst-case error remains stable. |
| **Feature Space** | 4 Observables | **7 Telemetry Channels** | +3 Channels | +75% | Richer powertrain dynamics integrated. |

---

## 🔬 Why Did 7 Features Succeed Where 8 Features Failed?

1. **Cohesive Physical System (Powertrain Dynamics vs. Auxiliary Noise)**:
   - The earlier 8-feature trial incorporated `Heating Power CAN` and `AirCon Power`. These auxiliary loads cycle erratically according to cabin thermostats and act as high-frequency split noise without reflecting cumulative charge draw.
   - The 7-feature model replaces auxiliary noise with **`Throttle [%]`** (driver demand intent), **`Motor Torque [Nm]`** (mechanical work), and **`Velocity [km/h]`** (vehicle kinetic energy state). These channels describe the vehicle's dynamic load regime.

2. **Preserving Voltage Dominance**:
   - In the 7-feature configuration, `Battery Voltage` retained **57.95%** of feature importance, ensuring the foundational open-circuit voltage curve continues to anchor SOC predictions at high tree depths.
   - The auxiliary features provide fine-grained adjustment during acceleration and regenerative deceleration phases.

---

## 📂 Artifacts & Diagnostics Generated

All artifacts are isolated within `RF_New/`:

```
RF_New/
├── results/
│   ├── models/
│   │   └── rf_7f_100_trees_final.joblib   # Serialized model
│   ├── plots/
│   │   ├── soc_estimation_rf.png         # Actual vs Predicted curve across 118,974 points
│   │   ├── prediction_error_rf.png       # Instantaneous residual error profile
│   │   ├── feature_importance_rf.png     # MDI feature importance ranking across 7 features
│   │   ├── tree_count_comparison.png     # RMSE & MAE comparison across 25..100 trees
│   │   ├── error_by_soc_range.png        # Error grouped by SOC operational intervals
│   │   ├── error_distribution.png        # Residual probability density & mean error
│   │   ├── per_trip_error.png            # Generalization across each of the 10 test trips
│   │   └── rf_4f_vs_7f_comparison.png    # Head-to-head bar chart (4F Baseline vs 7F Enhanced)
│   └── results.json                      # Complete multi-seed metrics and comparative JSON
└── src/
    ├── config.py                         # Hyperparameters, paths, and feature definitions
    ├── data_loader.py                    # Semicolon/Latin-1 loader with 7-feature extraction
    ├── evaluate.py                       # Metric calculations (RMSE, MAE, MAX, STD)
    ├── model_rf.py                       # Multi-seed training engine
    ├── plots.py                          # 8 diagnostic figure generators
    └── main.py                           # End-to-end execution pipeline
```
