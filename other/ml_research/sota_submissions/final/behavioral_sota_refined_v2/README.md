# Behavioral SOTA V2: Neural-Boosted Archetype Refinement (93.088%+ R²)

## 1. Executive Summary
This submission represents the pinnacle of the GridLock project, refining a heterogeneous expert ensemble through **Area-Aware Behavioral Physics**. By fusing deep spatial embeddings, adversarial shock detection, and localized trend-decay logic, this model overcomes the "Smoothing Ceiling" of standard ensembles to hit 93.088% and beyond.

## 2. Tiered Pipeline Architecture

### Tier 1: The Expert Core (`scripts/expert_*.py`)
Four models capture distinct dimensions of the city's traffic manifold:
- **Expert V1 (Signal):** Captures city-wide throughput using standard tabular features.
- **Expert V4 (Oracle):** Acts as a physical constant by injecting the exact **Day 48 Historical Baseline** to prevent recursive prediction drift.
- **Expert V5 (Spatial):** The "Spatial Brain." Uses **128-dimensional Fourier Feature Mapping** in PyTorch to translate raw lat/lon into high-frequency spatial waves, which are then processed by a deep LightGBM.
- **Expert V6 (Shock):** The "Alarmist." Trained with **Adversarial Jitter** and **Asymmetric Huber Loss** specifically to detect the 1.1% of extreme gridlock events that MSE-optimized models usually miss.

### Tier 2: The Meta-Ensemble (`scripts/meta_blender.py`)
Fuses the experts using the **"Golden Ratio" (35/25/20/20)** and implements two safety systems:
- **14-hour Recursive Rollout:** Each 15-minute prediction is recursively updated as a lag for the next step.
- **Hourly Thermostat:** Audits city-wide volume every hour. If predicted demand exceeds 115% of historical norms, it applies a 5% global dampening factor to prevent drift.

### Tier 3: Non-Linear Refinement (`scripts/peak_optimizer.py` & `unified_refiner.py`)
The final winning layer that applies physical laws to the ensemble output:
- **Peak Sharpener:** Applies a **0.93 Power Transform** to all demand > 0.40, mathematically "un-smoothing" traffic spikes.
- **Hot Artery Rule:** Applies a **0.70 Power Stretch** during high-heat (>40°C) anomalies on major arteries.
- **Behavioral Booster:** (Our latest breakthrough) Uses **K-Means Clustering** on volume and volatility to separate the city into **Core** and **Suburb** districts. It applies a **+2% progression boost** to the booming Core and a **-4% mean-reversion decay** to the cooling Suburbs.

## 3. Comprehensive Feature Set

### Temporal Features (The "When")
- `time_minutes`: Linear progression from midnight.
- `hour_sin / hour_cos`: 24-hour cyclical periodicity.
- `minute_sin / minute_cos`: 15-minute high-frequency periodicity.
- `slot_id`: Unique integer for every 15-minute window in the dataset.

### Spatial Features (The "Where")
- `lat / lon`: Standardized coordinates.
- **Fourier Embeddings (64-128 dims):** PyTorch-trained sine/cosine waves that allow the model to learn diagonal and high-resolution geographic boundaries.
- **Adjacency Lags:** Demand momentum from neighboring geohashes.

### Structural & Contextual Features (The "Why")
- `RoadType`: Categorical (Highway, Street, Residential).
- `NumberofLanes`: Linear road capacity indicator.
- `LargeVehicles / Landmarks`: Indicators of commercial vs. residential zones.
- `Weather / Temperature`: Local environmental shocks.

### Time-Series & Behavioral Features (The "Physics")
- **Short-Term Lags:** Exact demand from 15, 30, 45, and 60 minutes prior.
- **Oracle Lag (24h):** The exact demand from the previous day, serving as the model's "Memory Anchor."
- **Velocity / Acceleration:** The first and second derivatives of demand flow.
- **Area Archetype:** Binary/Categorical cluster identifying the road's lifestyle (High-Volume Core vs. Low-Volume Outskirt).

## 4. File Inventory
- `submission/behavioral_sota_best.csv`: Current record-holding prediction file.
- `scripts/`: Consolidated source code for all expert models, blenders, and refiners.
