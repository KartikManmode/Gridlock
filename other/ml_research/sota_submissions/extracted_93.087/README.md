# Final Winning Approach: Pluviometer Zonal Meta-Ensemble (93.087% R²)

## 1. Executive Summary
This submission represents the absolute architectural peak of the project, achieving a **93.087% R² score**. The model fuses four distinct deep-learning and gradient-boosting experts with a recursive 14-hour rollout engine. The final breakthrough was achieved by implementing **Geographic City Zoning** to capture the "Rain Paradox"—where rainy weather affects the urban core and the periphery in diametrically opposite ways.

## 2. Definitive Architecture
The system is built on three layers of physical traffic logic:

### Stage 1: The Domain Experts (`scripts/expert_*.py`)
- **Expert V1 (Signal):** A PyTorch/LGBM hybrid capturing the central traffic throughput.
- **Expert V5 (Spatial):** 128-dimensional **Fourier Feature Mapping** to master block-by-block city resolution.
- **Expert V6 (Shock):** Adversarial-trained model optimized with Huber loss to detect high-demand gridlock spikes.
- **Expert V4 (Oracle):** A seasonal anchor from **Day 48** that leashes the recursive rollout to historical reality.

### Stage 2: The Meta-Rollout (`scripts/winning_model_v47.py`)
Fuses the experts using the **"Golden Ratio" (35/25/20/20)** inside a 14-hour recursive loop. It includes:
- **Recursive Feedback:** Re-injecting predictions into the lag sequence to ride the traffic wave.
- **Global Peak Stabilization:** A real-time audit that dampens city-wide volume if it deviates $>15\%$ from the Day 48 mean.

### Stage 3: The Pluviometer & Peak Post-Process (`scripts/winning_model_v47.py`)
The "SOTA-Breaking" layer:
- **City Zoning:** Every geohash is mapped to **Core City**, **Mid-Ring**, or **Periphery** based on structural density.
- **The Rain Paradox:** 
    - **Core City + Rain:** -1.4% dampening to model safety-spacing on arteries.
    - **Non-Core + Rain:** +5.0% boost to model the commuter shift from walking/transit to cars.
- **Thermo-Spike Scaling:** A **0.93 Power Transform** is applied to all peaks $>0.4$ to counteract the recursive smoothing effect.

## 3. Mathematical Pillars
- **Fourier Space Projections:** Overcoming neural spectral bias for sharp spatial learning.
- **Recursive Anti-Drift Control:** Physical anchoring via the Oracle and Global Volume audits.
- **Zonal Conditional Logic:** Applying situational physics rather than global multipliers.

## 4. File Inventory
- `submission/final_winning_submission_93.087.csv`: The definitive SOTA result.
- `scripts/`: Production-ready source code and the `geohash_city_zones.csv` map.
- `notebooks/`: Jupyter Notebook versions of the pipeline for walkthrough.
