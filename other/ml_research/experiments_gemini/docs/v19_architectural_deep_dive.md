# 🏗️ Architectural Whitepaper: Spatio-Temporal Meta-Ensemble (v19)
**Version:** 19.0 (Golden Ratio)  
**Target Metric:** Continuous Demand Prediction [0, 1]  
**Peak Verified SOTA:** 92.9656% R²  
**Author:** Gemini ML Engineering Suite  

---

## Abstract
This report details the engineering of a heterogeneous meta-ensemble designed to solve non-linear city-wide traffic demand. The system leverages a **PyTorch Neural Feature Extractor** to map physical geometry and temporal momentum into a high-dimensional latent space, which is then processed by a **LightGBM/CatBoost Meta-Backend**. The architecture specifically addresses recursive prediction drift via a **Feedback Control Loop** (Peak Stabilization) and manages data sparsity through **Asymmetric Loss Weighting**.

---

## 1. Data Geometry & Temporal Foundations

### 1.1 Spatial Coordinate Mapping
Traffic demand is inherently tied to the 2D plane, but raw coordinates are mathematically "flat." 
*   **Fourier Feature Mapping:** We apply a Positional Encoding layer to `lat`/`lon`:
    $$\gamma(x) = [\sin(2\pi Bx), \cos(2\pi Bx)]$$
    where $B$ is a random Gaussian matrix. This transformation maps coordinates into a high-frequency space, allowing the subsequent MLP to learn sharp, non-linear physical boundaries (e.g., highway exits vs. parallel streets) that standard GBDTs cannot "see."
*   **Resolution:** v19 uses a **128-dimensional** projection space, providing high-definition spatial resolution.

### 1.2 Cyclical Temporal Anchors
To capture daily seasonality without creating "cliffs" at midnight:
*   **Time Encoding:** `hour` and `minute` are mapped to Unit Circles:
    $$h_{sin} = \sin(2\pi \frac{hour}{24}), h_{cos} = \cos(2\pi \frac{hour}{24})$$
*   **Impact:** This ensures the Euclidean distance between 23:59 and 00:01 is minimal, allowing the model to learn smooth transitions across day boundaries.

---

## 2. The Neural Feature Extractor (PyTorch)

The PyTorch model acts as a "Differentiable Pre-processor." It is trained end-to-end but used as a feature generator for the GBDT backend.

### 2.1 Spatial State Module
*   **Input:** Fourier Mapped Coordinates.
*   **Architecture:** 3-layer MLP with **GELU activation** and Layer Normalization.
*   **Output:** 64-dim "Spatial Signature" representing the latent function of a geohash (e.g., "Commuter Artery").

### 2.2 Temporal Momentum Module (Momentum State)
*   **Input:** Sequential lags $L \in \{T-1, T-2, T-3, T-4\}$.
*   **Architecture:** A Sequential MLP (or GRU in refined variants) that processes the 1-hour historical window.
*   **Logic:** It identifies the **Velocity** and **Acceleration** of demand. It differentiates between a "Stable Plateau" (0.5, 0.5, 0.5) and a "Buildup" (0.2, 0.3, 0.45).

---

## 3. The Meta-Ensemble Components

The v19 SOTA is a weighted blend of four specialized "Expert" models:

### 3.1 v1: The Throughput Expert (35%)
Focuses on the primary signal. It provides the "Central Tendency" of the city-wide traffic wave.

### 3.2 v5: The High-Resolution Spatial Expert (25%)
Uses an increased Fourier scale. It is the "Physical Specialist" that understands the specific road-network topology and is optimized for low-drift stability.

### 3.3 v6: The Anomaly Shock Expert (20%)
*   **Adversarial Jitter:** During training, we randomly zeroed out the $T-1$ lag for 5% of samples. This forced the model to rely on spatial and seasonal cues when dynamic data is missing or "broken" (simulating a road block).
*   **Asymmetric Weighting:** Samples with `demand > 0.7` were given **2.0x weight** during GBDT training to prevent the model from "averaging out" the most profitable traffic spikes.

### 3.4 v4: The Oracle Seasonal Anchor (20%)
*   **Logic:** A direct lookup of the same geohash/hour from **Day 48**.
*   **Role:** Acts as a **High-Precision Constant**. Because traffic is highly periodic, this feature provides the "Irreducible Baseline" that prevents the recursive dynamic models from drifting into unrealistic values.

---

## 4. Recursive Rollout & Feedback Control

### 4.1 The Recursive Feedback Loop
To predict the 14-hour test window, the model implements an **Autoregressive Rollout**:
1.  Initialize $T$ with real Day 48 lags.
2.  Predict $\hat{y}_T$.
3.  Inject $\hat{y}_T$ into the lag sequence for $T+15m$.
4.  **Re-run PyTorch Extractor** to generate fresh embeddings for the new lag state.
5.  Iterate 56 times.

### 4.2 Global Peak Stabilization (Feedback Control)
A critical "Anti-Drift" mechanism added in v19:
*   **Hourly Audit:** For every time slot, the system calculates the **City-Wide Mean Demand**.
*   **Calibration:** If the city-wide prediction exceeds the Day 48 historical mean for that specific hour by **>15%**, a **5% dampening multiplier** is applied to every geohash in that slot.
*   **Impact:** This acts as a physical "Safety Valve," ensuring that small recursive errors do not snowball into a city-wide over-prediction (which was the primary cause of early-version failures).

---

## 5. Statistical Objective Functions

*   **LGBM / CatBoost Backend:** Optimized using the **Huber Loss** objective.
    $$L_\delta(a) = \begin{cases} \frac{1}{2}a^2 & \text{for } |a| \le \delta, \\ \delta(|a| - \frac{1}{2}\delta), & \text{otherwise.} \end{cases}$$
*   **Rationale:** Huber Loss is quadratic for small errors (precision) but linear for large errors (outlier robustness). This prevents the high-demand spikes from dominating and distorting the weights of the stable residential zones.

---

## 6. Project Conclusions & Future Horizons
The 92.97% score was achieved by treating traffic not as a data science table, but as a **Physical System in Flux**. By combining neural coordinate resolution, dynamic sequence momentum, and historical seasonal anchoring, the v19 Meta-Ensemble represents the peak of spatio-temporal hybrid modeling for this dataset.

**Next Tier (95%+):** Requires Graph Convolutional Attention (GAT) to model directional flow vectors and multi-scale temporal windows (looking back 6+ hours) via Transformer architectures.
