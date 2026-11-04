# Design Document: Spatio-Temporal Fusion Model (v8)

## 1. Objective
Achieve 95%+ R² by moving beyond tabular GBDTs and basic Deep Learning. We will integrate learned spatial signatures, neighborhood-aware attention, and stability-focused residual training.

## 2. Model Architecture: `FusionSTA_Model`

### A. Inputs
- **Categorical:** `geohash_id` (long), `Weather` (long), `RoadType` (long).
- **Numeric:** `Temperature`, `Lanes`, `hour_sin/cos`, `day`.
- **History:** Sequence of past `demand` snapshots for the target geohash and its neighbors.

### B. Core Modules
1. **Differentiable Geohash Embeddings (Task 1):**
   - Each geohash maps to a $D$-dimensional vector.
   - Captures latent site characteristics (e.g., "Main artery" vs. "Dead-end").
   - Interacts with `hour_sin/cos` via a small MLP to generate a time-dependent "spatial signature."

2. **Spatial-Temporal Attention (STA) (Task 2):**
   - **Spatial Branch:** Graph Attention (GAT) layer over the adjacency matrix defined by `geohash.neighbors()`.
   - **Temporal Branch:** Multi-Head Attention (Transformer) across the history of snapshots.
   - **Fusion:** Weighted sum or concatenation of spatial context and temporal momentum.

3. **Residual Head & Stability (Task 3):**
   - **The Baseline:** The model takes as input a "Stable Mean" (e.g., `geo5_hour_te`).
   - **The Task:** Predict the *deviation* from this mean.
   - **Scheduled Sampling:** During training, randomly replace historical demand with the model's own previous prediction to improve robustness in multi-step inference.

## 3. Training & Optimization
- **Loss Function:** Huber Loss (less sensitive to outliers than MSE, but smoother than MAE).
- **Validation:** Strict `TimeSeriesSplit`.
- **Normalization:** Standard scaling for numeric features; learned embeddings for categorical.

## 4. Implementation Plan
1. **Data Pipeline:** Build a generator that produces batches of `(History_Windows, Spatial_Adjacency, Target_Demand)`.
2. **Model Definition:** Implement `nn.Embedding`, `nn.MultiheadAttention`, and the `ResidualHead`.
3. **Training Loop:** Integrated with Scheduled Sampling logic.
4. **Inference Script:** Handles the recursive rollout for the 14-hour test set.
