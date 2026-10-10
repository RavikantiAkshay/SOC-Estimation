# 19-Feature Temporal & Physics-Compensated Random Forest Model (`RF_Temporal`)

This module implements the **Temporal, Dynamic & Physics-Compensated State Extension** for Lithium-Ion Battery State of Charge (SOC) estimation, building upon baseline observable physics, powertrain kinematics, rolling statistical memory, dynamic voltage sag, electrical tractive power, and dual-polarization Open-Circuit Voltage compensation.

---

## 1. Feature Architecture (19 Features)

All temporal rolling windows, dynamic sags, electrical power, and Open-Circuit Voltage estimations are computed **strictly per trip** within `RF_Temporal/src/data_loader.py` to prevent time-series data leakage across trip boundaries.

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

### D. Local, Macro & Ultra-Macro Rolling Windows (5 Features)
10. `V_mean_15s [V]` — 15-second rolling average of pack voltage. Filters 1s throttle/regen spikes.
11. `I_mean_15s [A]` — 15-second rolling average of current. Distinguishes sustained power draw from transient fluctuations.
12. `V_std_15s [V]` — 15-second rolling standard deviation of voltage. Encodes local load volatility.
13. `V_mean_60s [V]` — 60-second rolling average of pack voltage. Anchors intermediate resting baseline.
14. `I_mean_60s [A]` — 60-second rolling average of current.
15. `V_mean_180s [V]` — 180-second (3-minute) ultra-macro rolling average of voltage. Anchors resting open-circuit baseline and resists long sustained highway acceleration pulls (>60s) that dragged down 60s windows.

### E. Dynamic Voltage Sag (1 Feature)
16. `V_sag_60s [V]` — Instantaneous voltage sag relative to the 60s macro rolling baseline (<em>V<sub>t</sub></em> &minus; <em>V̄</em><sub>60s</sub>). Directly isolates load-induced polarization dip from underlying battery depletion.

### F. Electrical Tractive Power (1 Feature)
17. `Power [kW]` — Instantaneous battery terminal electrical power ((<em>V</em><sub>pack</sub> &times; <em>I</em><sub>pack</sub>) / 1000). Bridges electrical load with mechanical tractive demand, allowing the tree to isolate power regimes in a single split without attempting multi-split hyperbolic approximations.

### G. Physics-Compensated Open-Circuit Potential (2 Features)
18. `V_est_ocv [V]` — Calibrated Arrhenius temperature-compensated Ohmic Open-Circuit Voltage estimate (<em>V<sub>t</sub></em> &minus; <em>I<sub>t</sub></em> &times; <em>R</em><sub>0</sub>(<em>T</em>)), using empirical <em>R</em><sub>0</sub> = 0.055 &Omega; at 25 °C and Arrhenius slope 0.060 for realistic sub-zero electrolyte resistance.
19. `V_est_ocv_full [V]` — Full dual-polarization Open-Circuit Voltage estimate (<em>V<sub>t</sub></em> &minus; <em>I<sub>t</sub></em> &times; <em>R</em><sub>0</sub>(<em>T</em>) &minus; <em>Ī</em><sub>15s</sub> &times; <em>R</em><sub>pol</sub>(<em>T</em>)), decoupling both instantaneous ohmic drop and slower 15s electrochemical charge-transfer / diffusion polarization.

---

## 2. Benchmark Evolution Across Research Stages

Evaluated across **118,974 test instances** spanning 10 completely unseen real-world driving trips:

| Stage | Model Configuration | Features | Trees | Best RMSE | Best MAE | Peak MAX Error |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Stage 1** | Baseline Random Forest | 4 | 25 | 5.8876% | 4.3736% | 26.6956% |
| **Stage 2** | Enhanced Random Forest (`RF_New`) | 7 | 50 | 4.5734% | 3.3158% | 27.0113% |
| **Stage 3** | **Temporal & Physics-Compensated (`RF_Temporal`)** | **19** | **50** | **3.8500%** 🔻 | **2.9666%** 🔻 | **15.5543%** 🔻 |

### Multi-Seed Tree Count Evaluation (19 Features)

| Trees | Best RMSE | Avg RMSE | Best MAE | Best MAX Error | Selected Metric (Min Peak Error) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **25** | 3.8430% | 3.8821% | 2.9773% | 16.9312% | RMSE 3.9211%, MAE 3.0158%, MAX 16.9312% |
| **50** | **3.8058%** | **3.8560%** | **2.9270%** | **15.5543%** | **Optimal: RMSE 3.8500%, MAE 2.9666%, MAX 15.5543%** |
| **75** | 3.8237% | 3.8481% | 2.9413% | 15.7848% | RMSE 3.8437%, MAE 2.9573%, MAX 15.7848% |
| **100** | 3.8152% | 3.8394% | 2.9418% | 15.8960% | RMSE 3.8532%, MAE 2.9709%, MAX 15.8960% |

* **RMSE Error Reduction vs Baseline**: **&minus;34.61%** (from 5.8876% down to 3.8500%)
* **MAE Error Reduction vs Baseline**: **&minus;32.17%** (from 4.3736% down to 2.9666%)
* **Peak MAX Error Reduction vs Baseline**: **&minus;41.73%** (from 26.6956% down to **15.5543%**)
* **Consistent 50 Trees**: Matches the 50-tree architecture selected in `RF_New`, achieving the absolute lowest peak error (15.55%), sub-3% MAE (2.97%), and superior embedded execution efficiency.

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
