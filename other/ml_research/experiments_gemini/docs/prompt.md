# Project: Gridlock - Traffic Demand Prediction
## Phase 4: Zero-Inflated Modeling & Multi-Horizon Forecasting

### 1. Project Context & Current State
- **Goal:** Predict continuous traffic `demand` [0.0, 1.0] for city blocks (`geohash`) in 15-minute intervals.
- **SOTA Score Achieved:** **~88% (R²)** (Honest, drift-free score on strict TimeSeriesSplit via `traffic_demand_pipeline_v9_hybrid.py`).
- **The "91.7% Leak" Resolved:** Earlier models (v5) hit 91.7% by using global target encoding (peeking into future test sets) and random splits. Fixing this leakage and moving to a strict chronological split recalibrated the baseline to an honest ~80% CV and ~88% test score.
- **The "Recursive Drift" Resolved:** Autoregressive models predicting lag `T-1` drifted heavily over the 14-hour test set. `v9_hybrid` fixed this by using a PyTorch GCN to extract static spatial embeddings and feeding them into a direct-prediction Gradient Boosting Meta-Ensemble (LightGBM + CatBoost).
- **Validation:** Strict `TimeSeriesSplit` (3-fold to 5-fold) is mandatory.

---

### 2. The "Road to 95+" Prompt for Codex/Next Assistant

**Copy and paste the following into the next session:**

---

**Context & Current State:**
We are predicting traffic `demand` (0.0 to 1.0) using 15-minute spatio-temporal data. We have stabilized the architecture. We currently use a Hybrid model (GCN for spatial embeddings + LightGBM/CatBoost for direct prediction) to avoid recursive prediction drift, achieving an honest, leakage-free **~88% R²**.

**The Mission:**
Bridge the gap from 88% to **95%+ R²**. Because we cannot use standard recursive lags without drifting, we must extract more signal from the structural and domain-specific nature of the data.

Act as a Principal Machine Learning Architect. I want you to design and implement three distinct pipeline strategies to break the 90% barrier.

**Strategy 1: Zero-Inflated (Two-Stage) Hybrid Modeling**
Traffic demand is heavily skewed towards 0.0 (empty roads). Tree-based regressors struggle with this zero-inflation.
*   **Stage 1 (Classifier):** Train an XGBoost or LightGBM classifier to predict the probability that a geohash has *any* traffic at a given time ($P(Demand > 0.01)$).
*   **Stage 2 (Regressor):** Train a CatBoost/LightGBM regressor *strictly* on rows where $Demand > 0.01$.
*   **Fusion:** Integrate the GNN spatial embeddings from `v9` into both stages. Output $P(Active) \times Predicted\_Volume$.

**Strategy 2: Time-of-Day Expert Routing (MoE)**
Traffic physics fundamentally change based on the hour. A single model interpolating between 3 AM and 8 AM struggles.
*   **Architecture:** Create an explicit Mixture of Experts (MoE).
*   **Experts:** Train three separate meta-ensembles: `Night_Expert` (22:00-06:00), `Rush_Expert` (07:00-09:00, 17:00-19:00), and `Normal_Expert` (everything else).
*   **Routing:** During inference, route the test row to the corresponding expert based on the `hour` feature.

**Strategy 3: Multi-Horizon Sequence-to-Sequence (Seq2Seq)**
Instead of predicting one slot at a time and feeding it back into the model (causing drift), predict the entire future window simultaneously.
*   **Architecture:** Use a PyTorch Transformer Encoder-Decoder or a 1D-CNN architecture.
*   **Input:** The last 8 known slots (from the end of the train set) for a geohash.
*   **Output:** Directly output the next $N$ slots (the entire 14-hour horizon) in a single pass.
*   **Loss:** Use a sequence-level loss (e.g., Dynamic Time Warping (DTW) loss or a heavily weighted Huber loss on peak traffic).

**Constraints:**
*   **Strict Chronology:** Maintain `TimeSeriesSplit`. Never use future target means as features.
*   Keep the architecture modular. Start by drafting Strategy 1 in a new script.

---
