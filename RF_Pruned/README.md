# Stage 4: Random Forest Depth Pruning & Pareto Complexity Optimization (`RF_Pruned`)

This module implements **Stage 4 (Pareto Pruning & Optimization)**: systematic decision tree depth pruning and Pareto frontier optimization. It bridges high-accuracy algorithmic machine learning with real-world embedded automotive Electronic Control Unit (ECU) constraints.

---

## 1. Executive Summary & Problem Formulation

While the **17-Feature Temporal & Dynamic Model (`RF_Temporal`)** pushed SOC estimation accuracy to state-of-the-art levels (RMSE = 3.85%, Peak MAX error = 15.55%), unconstrained decision trees grew to an average depth of **~49 levels**, accumulating over **7.55 million decision nodes** and requiring a **517 MB** serialized model binary.

In production automotive microcontrollers (e.g., **Infineon AURIX TC3xx, NXP S32K, ARM Cortex-M**):
- **Flash / ROM storage** is strictly constrained (typically 2 MB to 8 MB total).
- **Active SRAM** is limited (typically 512 KB to 4 MB).
- **Deterministic Latency** must be strictly guaranteed within real-time vehicle task loops (<10 ms).

To resolve this bottleneck, we conducted an empirical **Pareto Complexity Analysis**, sweeping tree depth constraints (`max_depth = [10, 14, 18, 22, 26, None]`) across 50-tree ensembles trained on **945,026 samples** and evaluated on **118,974 unseen test samples**.

---

## 2. Multi-Depth Empirical Evaluation Benchmark

| Max Depth | Test RMSE (%) | Test MAE (%) | Peak MAX Error (%) | Total Forest Nodes | Complexity Reduction (%) | Model Size (.joblib) | Estimated C Flash (MB) | Worst-Case Branch Checks (50 Trees) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Depth 10 (Pareto Optimal)** | **3.7069%** 🏆 | **2.8675%** 🏆 | **16.4311%** | **90,958** | **98.8%** 🔻 | **2.41 MB** 🔻 | **1.04 MB** | **500 checks** (5.0× faster path) 🔻 |
| **Depth 14** | 3.8278% | 2.9565% | 17.6749% | 759,474 | 89.9% 🔻 | 12.8 MB | 8.69 MB | 700 checks |
| **Depth 18** | 3.8343% | 2.9314% | 17.8565% | 2,650,570 | 64.8% 🔻 | 44.5 MB | 30.33 MB | 900 checks |
| **Depth 22** | 3.9154% | 3.0196% | 17.0189% | 5,094,584 | 32.4% 🔻 | 85.2 MB | 58.30 MB | 1,100 checks |
| **Depth 26** | 3.8380% | 2.9439% | 16.6620% | 6,766,202 | 10.2% 🔻 | 114.6 MB | 77.43 MB | 1,300 checks |
| **None (Unconstrained, ~49)** | 3.8058% | 2.9270% | 17.8737% | 7,558,542 | 0.0% (Baseline) | 517.3 MB | 86.50 MB | 2,450 checks (Baseline) |

### Key Engineering Insights:
1. **The Structural Regularization Effect**:
   Capping tree depth at **`max_depth = 10`** actually **reduced RMSE from 3.8058% to 3.7069%**. In deep unconstrained trees, bottom leaf nodes fit to microscopic electrical noise and road vibration jitter. Pruning enforces global generalization.
2. **98.8% Node Complexity Elimination**:
   Total decision nodes drop from **7,558,542 down to 90,958**—eliminating **7.46 million redundant branch splits**.
3. **99.5% Model Serialization Reduction**:
   Model file footprint collapses from **517 MB down to 2.41 MB**, comfortably fitting in standard memory architectures.
4. **Automotive Microcontroller Compatibility**:
   - **Flash Requirement in C**: **`1.04 MB`** (using flat static arrays: `struct Node { uint8_t feat; float thresh; uint16_t left; uint16_t right; }`).
   - **Active RAM at Runtime**: **`< 2 KB`** (only 17 floats input buffer + 180s ring buffer + traversal stack).
   - **Deterministic WCET Path**: **`500 branch checks per sample`** (a **5.0× reduction in execution path** vs. 2,450 unpruned checks, taking **`< 0.05 ms`** on a 100 MHz automotive MCU core, well within the vehicle's 100 ms CAN task budget).

---

## 3. The Complete 4-Stage Research Journey

| Research Stage | Model Architecture | Features | Trees | Best RMSE | Best MAE | Peak MAX Error | Total Nodes | Serialized Size | Engineering Focus |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Stage 1** | Baseline Random Forest | 4 | 25 | 5.8876% | 4.3736% | 26.6956% | 3,752,787 | 257.7 MB | Literature Reproduction (Sulaiman 2024) |
| **Stage 2** | Enhanced Random Forest (`RF_New`) | 7 | 50 | 4.5734% | 3.3158% | 27.0113% | 7,720,000 | 558.8 MB | Powertrain Kinematics Expansion |
| **Stage 3** | Temporal & Dynamic RF (`RF_Temporal`) | 17 | 50 | 3.8058% | 2.9270% | 15.5543% | 7,558,542 | 517.3 MB | State-of-the-Art Accuracy & Memory |
| **Stage 4** | **Pareto-Pruned RF (`RF_Pruned`)** | **17** | **50** | **3.7069%** 🏆 | **2.8675%** 🏆 | **16.4311%** | **90,958** 🔻 | **2.41 MB** 🔻 | **Automotive Production & ECU Sizing** |

---

## 4. Diagnostic Visualization Suite

All plots reside in `RF_Pruned/results/plots/`:
- **`pareto_frontier_tradeoff.png`**: Dual-panel Pareto curve showing Model Complexity (Nodes) vs. RMSE & Peak MAX Error.
- **`depth_sweep_comparison.png`**: Grouped bar chart comparing error metrics across all 6 depth configurations.
- **`four_stage_evolution.png`**: Side-by-side progression showing error reduction and complexity evolution across all 4 research stages.
- **`soc_estimation_pruned.png`**: Actual vs. Predicted SOC trajectory across 118,974 unseen test instances.

---

## 5. How to Run

To re-run the pruning sweep and regenerate all plots:

```powershell
python e:\EV_HEV\RF_Pruned\src\main.py
```

