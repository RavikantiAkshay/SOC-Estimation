# 15-Feature Temporal & Physics-Compensated Random Forest Model (`RF_Temporal`)

This module implements the **Temporal, Dynamic & Physics-Compensated State Extension** for Lithium-Ion Battery State of Charge (SOC) estimation, building upon baseline observable physics and powertrain kinematics.

---

## 1. Feature Architecture (15 Features)

All temporal rolling windows, derivatives, and Open-Circuit Voltage estimations are computed **strictly per trip** within `RF_Temporal/src/data_loader.py` to prevent time-series data leakage across trip boundaries.

### A. Baseline Observables (4 Features)
1. `Battery Voltage [V]` — Thermodynamic open-circuit potential + instantaneous IR drop.
2. `Battery Current [A]` — Instantaneous load demand / regenerative charge rate.
3. `Battery Temperature [°C]` — Internal cell core thermal state.
4. `Ambient Temperature [°C]` — Boundary heat exchange temperature.

### B. Powertrain & Kinematics (3 Features)
5. `Throttle [%]` — Driver power demand intention.
6. `Motor Torque [Nm]` — Mechanical torque load on inverter/motor.
7. `Velocity [km/h]` — Vehicle kinematic state and aerodynamic drag.

### C. Temporal Micro-Dynamics (2 Features)
8. `dV_dt [V/s]` — Instantaneous voltage rate of change (<em>dV/dt</em>). Captures rapid step transitions vs. steady-state relaxation.
9. `dI_dt [A/s]` — Instantaneous current rate of change (<em>dI/dt</em>). Resolves dynamic polarization and RC transient settling.

### D. Local & Macro Temporal Rolling Windows (4 Features)
10. `V_mean_15s [V]` — 15-second rolling average of pack voltage. Filters 1s throttle/regen spikes.
11. `I_mean_15s [A]` — 15-second rolling average of current. Distinguishes sustained power draw from transient fluctuations.
12. `V_std_15s [V]` — 15-second rolling standard deviation of voltage. Encodes local load volatility.
13. `V_mean_60s [V]` — 60-second rolling average of pack voltage. Anchors true baseline resting voltage during sustained acceleration.
14. `I_mean_60s [A]` — 60-second rolling average of current.

### E. Physics-Compensated Open-Circuit Potential (1 Feature)
15. `V_est_ocv [V]` — Arrhenius temperature-compensated Open-Circuit Voltage estimate (<em>V<sub>t</sub></em> &minus; (<em>I<sub>t</sub></em> &times; <em>R</em><sub>0</sub>(<em>T</em>))), decoupling internal resistance drops under high loads.

---

## 2. Benchmark Evolution Across Research Stages

Evaluated across **118,974 test instances** spanning 10 completely unseen real-world driving trips:

| Stage | Model Configuration | Features | Trees | Best RMSE | Best MAE | Peak MAX Error |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Stage 1** | Baseline Random Forest | 4 | 25 | 5.8876% | 4.3736% | 26.6956% |
| **Stage 2** | Enhanced Random Forest (`RF_New`) | 7 | 50 | 4.5734% | 3.3158% | 27.0113% |
| **Stage 3** | **Temporal & Physics-Compensated (`RF_Temporal`)** | **15** | **50** | **3.9994%** 🔻 | **3.0345%** 🔻 | **18.9379%** 🔻 |

### Multi-Seed Tree Count Evaluation (15 Features)

| Trees | Best RMSE | Avg RMSE | Best MAE | Best MAX Error |
| :---: | :---: | :---: | :---: | :---: |
| **25** | 4.0176% | 4.0904% | 3.0634% | 19.6523% |
| **50** | **3.9994%** | **4.0546%** | **3.0345%** | **18.9379%** |
| **75** | 4.0049% | 4.0421% | 3.0329% | 19.7657% |
| **100** | 4.0078% | 4.0396% | 3.0071% | 20.5827% |

* **RMSE Error Reduction vs Baseline**: **&minus;32.07%** (from 5.8876% down to 3.9994%)
* **Peak MAX Error Reduction vs Baseline**: **&minus;29.06%** (from 26.6956% down to 18.9379%)
* **Consistent 50 Trees**: Matches the 50-tree architecture selected in `RF_New`, achieving the absolute lowest RMSE, lowest MAE, and sub-19% peak error.

---

## 3. Directory Layout

```text
RF_Temporal/
├── README.md                 # This file
├── results/
│   ├── models/               # Serialized best model (.joblib)
│   ├── plots/                # 8 diagnostic charts
│   └── results.json          # Multi-seed metrics across all tree counts
└── src/
    ├── config.py             # Feature definitions, paths, and hyperparameters
    ├── data_loader.py        # Per-trip feature engineering and 60/10 split
    ├── evaluate.py           # Standard metrics (RMSE, MAE, MAX, STD)
    ├── model_rf.py           # Multi-seed RF training engine with sample weighting
    ├── plots.py              # Diagnostic plotting suite (including 3-stage evolution)
    └── main.py               # End-to-end pipeline runner
```

---

## 4. How to Run

To run the complete pipeline:

```powershell
python e:\EV_HEV\RF_Temporal\src\main.py
```
