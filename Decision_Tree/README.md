# Decision Tree Baseline for SOC Estimation

This directory contains the independent, modular implementation of the **Decision Tree** baseline for State of Charge (SOC) estimation in Electric Vehicles.

---

## 🎯 Baseline Objectives

1. **Single-Tree Benchmark**: Quantify the performance of a single decision tree ($N_{\text{estimators}} = 1$) to prove the statistical necessity of Random Forest's bootstrap aggregating (bagging) and feature sub-sampling.
2. **Variance & Overfitting Analysis**: Evaluate how a single high-variance tree behaves compared to an ensemble when exposed to noisy, unseen dynamic driving cycles.
3. **Worst-Case Error Bound**: Measure peak estimation error to evaluate BMS safety limits.

---

## ⚙️ Configuration & Hyperparameters

- **Estimators**: `1` (Single decision tree)
- **Max Features**: `'sqrt'` (Matches the individual tree constraint used in the paper's Random Forest)
- **Min Samples Leaf**: `5` (Regularization parameter matching the base learner)
- **Random Seeds Tested**: `[42, 101, 7, 999, 1234]` across 5 independent runs to account for feature split stochasticity

---

## 📊 Benchmark Results (Head-to-Head Comparison)

Tested on **118,974 unseen test instances** across **10 real-world EV trips**:

| Metric | Random Forest (25 Trees) | Decision Tree (Best Single Tree) | RF Improvement |
| :--- | :---: | :---: | :---: |
| **RMSE** | **`5.8876%`** | `7.0223%` | **RF is 16.16% better** |
| **MAE** | **`4.3736%`** | `5.2075%` | **RF is 16.01% better** |
| **Max Error** | **`26.6956%`** | **`41.7567%`** | **RF is 36.07% better** |
| **Std Dev** | **`5.8876%`** | `7.0162%` | **RF is 16.09% better** |

---

## 🔬 BMS Engineering Perspective: Why Random Forest Wins

1. **36% Reduction in Worst-Case Error**:
   - A single Decision Tree suffered a catastrophic maximum error of **`41.76%`** on transient spikes. In a physical BMS, an error this large would trigger false under-voltage protection, shut down the contactors, or strand the driver.
   - Random Forest caps the peak error at **`26.70%`** by averaging out uncorrelated tree deviations.
2. **Variance Dampening**:
   - Ensemble bagging dampens the sensitivity to measurement noise in current and voltage sensors during regenerative braking.

---

## 🚀 How to Run

To train, evaluate across 5 seeds, and generate diagnostic plots:

```bash
python Decision_Tree/src/main.py
```

Generated artifacts:
- Serialized Model: `Decision_Tree/results/models/dt_model.joblib`
- Numerical metrics: `Decision_Tree/results/results.json`
- Visualization figures: `Decision_Tree/results/plots/`
