# Winning Approach: Neural-Boosted Meta-Ensemble (93.065% R²)

## 1. Executive Summary
This submission utilizes a **Heterogeneous Meta-Ensemble** that achieves a 93.065% R² score on the Traffic Demand forecasting task. The core innovation is the fusion of **Deep Spatial Features (Fourier Mapping)** with **Recursive Temporal Momentum**, stabilized by a **Seasonal Oracle** and optimized via a **Non-Linear Peak Post-Processor**.

## 2. Pipeline Architecture
The final result is generated through a three-stage pipeline:

### Stage 1: The Expert Models (`scripts/expert_*.py`)
Four models were trained to capture distinct physical dimensions of the traffic grid:
- **Expert V1 (Signal):** Captures the primary city-wide traffic throughput.
- **Expert V5 (Spatial):** Uses 128-dimensional **Fourier Feature Mapping** to master high-frequency spatial boundaries (road-by-road resolution).
- **Expert V6 (Shock):** Trained with **Adversarial Jitter** and Asymmetric Huber Loss to detect 1.1% high-demand spikes and gridlock.
- **Expert V4 (Oracle):** A static historical anchor from **Day 48** that acts as a physical constant to prevent recursive prediction drift.

### Stage 2: The Meta-Blender (`scripts/meta_blender.py`)
Fuses the four experts using the **"Golden Ratio" (35/25/20/20)**. This script implements the **14-hour Recursive Rollout**, where each 15-minute prediction is fed back into the lag sequence for the next step. It includes a **Global Peak Stabilization** feedback loop that audits city-wide volume hourly against historical baselines.

### Stage 3: The Peak Optimizer (`scripts/peak_optimizer.py`)
The final winning refinement. Since recursive models tend to "smooth out" severe traffic peaks over long horizons, this script applies a **0.93 Power Transform** to the top 10% of predictions. This surgically stretches the traffic spikes back to their physical thermodynamic reality, yielding the final 93.065% score.

## 3. Key Mathematical Innovations
- **Fourier Spatial Mapping:** Overcomes neural "spectral bias" to learn sharp block-by-block demand functions.
- **Anti-Drift Feedback Loop:** Real-time volume auditing to prevent runaway recursive errors.
- **Thermo-Spike Scaling:** Non-linear correction of the ensemble's smoothing tendency.

## 4. File Inventory
- `submission/winning_submission_93.065.csv`: The final SOTA prediction file.
- `scripts/`: Production-ready Python modules for each stage.
- `notebooks/`: Jupyter Notebook versions for visualization and walkthrough.
