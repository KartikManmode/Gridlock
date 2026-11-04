# Project Gridlock: Architecture & Repository Map

## 1. Project Overview & Understanding
**Project Gridlock** is an intelligent urban traffic monitoring, congestion forecasting, and incident enforcement platform. It bridges data-driven machine learning models (achieving **93.087% R²** in competitive forecasting) with a production-ready smart city backend architecture.

### Key Milestones:
- **Phase 1: Exploration & Baselines**: Baseline Linear Regression and LightGBM models over Day 48/49 geohash-level data.
- **Phase 2: The Oracle & Feature Engineering**: Discovery of yesterday's demand (Day 48 same-hour lag) as an anti-drift anchor, combined with Bangalore Police real-world violation records.
- **Phase 3: Specialized Expert Development**: Creation of 4 domain experts:
  1. *Signal Expert (V1)*: PyTorch/LGBM central throughput & high-frequency cycles.
  2. *Spatial Expert (V5)*: 128-dimensional Fourier feature mapping and zone topology.
  3. *Shock Expert (V6)*: Adversarial Huber-loss model for rapid traffic spikes.
  4. *Oracle Expert (V4)*: Day 48 seasonal anchor.
- **Phase 4: Meta-Fusion & Neural Boosting**: Golden Ratio blending (35/25/20/20) and global hourly anchoring.
- **Phase 5: SOTA Refinement (Winning Model v47 - 93.087%)**:
  - *Zonal Conditional Logic (The Rain Paradox)*: Core City rain dampening (-1.4%) vs Periphery rain surge (+5.0%).
  - *Thermo-Spike Scaling*: Power transform ($y^{0.93}$) for high-demand peaks ($>0.40$).
  - *Extreme Heat Arteries*: High-temperature expansion ($>40^\circ\text{C}$).

---

## 2. System Architecture & Data Flow Alignment

The backend directory (`backend/`) is structured directly according to the Data Flow Diagram:
[`backend/Traffic Congestion Forecasting Platform-2026-11-03-212236.mmd`](backend/Traffic%20Congestion%20Forecasting%20Platform-2026-11-03-212236.mmd)
and the Database Schema:
[`database/Traffic Congestion Monitoring Architecture-2026-11-03-212351.mmd`](database/Traffic%20Congestion%20Monitoring%20Architecture-2026-11-03-212351.mmd).

```
                      +---------------------------------------+
                      |       UserEntity (Admin/Op/Citizen)   |
                      +---------------------------------------+
                        | (Credentials)     ^ (JWT/Roles) | (Forecast Req)  ^ (Prediction)
                        v                   |             v                 |
                 +--------------+    +------------------------------------------+
                 | P1: 1.0 Auth |    |    P3: 3.0 Predictive Forecasting        |
                 +--------------+    +------------------------------------------+
                   ^          |           | (Lags)   ^ (Preds)     | (Log)
                   |          v           v          |             v
                 +--------------+    +------------------+    +------------------+
                 | D1: USERS    |    | MLEngine: Python |    | D3: PREDICTION_  |
                 +--------------+    | ML Engine (SOTA) |    | LOGS             |
                                     +------------------+    +------------------+
                                              ^
  +-----------------------+                   | (Lags)
  | SensorEntity (Feeds)  |                   |
  +-----------------------+                   |
     | (Readings)                             |
     v                                        |
  +---------------------------+               |
  | P2: 2.0 Zone & Sensors    |---------------+
  +---------------------------+
     | (Insert & Update)
     v
  +--------------------------------+
  | D2: ZONES & SENSOR_RECORDS     |------+
  +--------------------------------+      |
                                          |
  +------------------------------------+  | (Aggregations)
  | UserEntity (Report Violation)      |  |
  +------------------------------------+  |
     |                                    v
     v                         +-------------------------------+
  +-------------------------+  | P5: 5.0 Analytics & Dashboard |
  | P4: 4.0 Incident/Disp.  |  +-------------------------------+
  +-------------------------+                 | (Heatmap & KPIs)
     | (Insert/Dispatch)                      v
     v                         +-------------------------------+
  +-------------------------+  | UserEntity (Dynamic Dashboard)|
  | D4: INCIDENTS/DISPATCHES|  +-------------------------------+
  +-------------------------+
```

---

## 3. Directory Layout & File Organization

| Directory / Location | Purpose | Relevant DFD / ERD Store |
| :--- | :--- | :--- |
| **`backend/`** | Production backend platform with FastAPI REST APIs. | All 5 Processes (P1-P5) |
| ├── `backend/p1_auth_rbac/` | Authentication, password hashing, JWT tokens, RBAC roles. | P1: Process 1.0 / `D1: users` |
| ├── `backend/p2_zone_sensors/` | Zone management, geohash spatial resolution, sensor stream ingestion. | P2: Process 2.0 / `D2: zones, sensor_records` |
| ├── `backend/p3_forecasting/` | Lag feature extraction, forecast request handling, forecast logging. | P3: Process 3.0 / `D3: prediction_logs` |
| ├── `backend/ml_engine/` | SOTA 93.087% ML Engine: Signal, Spatial, Shock, Oracle, Pluviometer, Thermo-Spike. | MLEngine |
| ├── `backend/p4_incidents_dispatch/` | Traffic violation reports, officer dispatching, hotspot analysis. | P4: Process 4.0 / `D4: incidents, dispatches` |
| ├── `backend/p5_analytics_dashboard/`| Summary KPIs, dynamic geospatial heatmaps, hotspot aggregation. | P5: Process 5.0 |
| ├── `backend/database/` | SQLite (`gridlock.db`) & PostgreSQL DDL schemas, connection pool, seed scripts. | D1, D2, D3, D4 |
| ├── `backend/api/` | REST route controllers for all 5 processes. | API layer |
| ├── `backend/main.py` | Unified FastAPI application entrypoint. | API Server |
| └── `backend/test_backend.py` | Automated end-to-end verification test suite. | Testing |
| **`database/`** | Database architecture specifications. | ERD specification |
| ├── `Traffic Congestion Monitoring Architecture-2026-11-03-212351.mmd` | Mermaid Entity-Relationship Diagram. | ERD |
| └── `schema.sql` | Production PostgreSQL schema script. | SQL DDL |
| **`dataset/`** | Raw competition traffic sensor dataset. | D2: Sensor Records |
| ├── `train.csv` | Day 48 sensor logs across Bangalore geohashes (15-min intervals). | Historical Data |
| ├── `test.csv` | Day 49 prediction targets. | Evaluation Targets |
| └── `sample_submission.csv`| Benchmark submission template. | Template |
| **`Phase2/`** | Incident violations data & analysis. | D4: Incidents & Hotspots |
| ├── `jan to may police violation_anonymized791b166.csv` | Bangalore Traffic Police violation records. | Real-world incidents |
| └── `analysis.py` | Hotspot clustering, high-impact parking analysis. | Analytics |
| **`ml_research/`** | ML experimentation, legacy baselines, and model iterations. | Research |
| ├── `baselines/` | Phase 1 baseline scripts (`linear_reg.py`, `advanced_traffic_model.py`, `eda.py`, etc.). | Baselines |
| ├── `experiments_gemini/` | Gemini model series (`pipeline_v5` to `v47_pluviometer_engine.py`). | Model Iterations |
| ├── `experiments_codex/` | Codex deep learning models (`dl_graph_model.py`, `dl_sequence_model.py`). | GNN & Sequence |
| ├── `experiments_claude/` | Claude pipeline v6. | Feature Pipelines |
| ├── `sota_submissions/` | Extracted 93.087 SOTA and 93.065 SOTA packages. | Winning Packages |
| └── `training_logs/` | CatBoost training artifacts (`catboost_info/`). | Logs |
| **`artifacts/`** | Model prediction CSVs and evaluation visualization plots. | Outputs |
| └── `plots/` | Feature importance, residuals, SHAP values, demand distributions. | Plots |
| **`archives/`** | Compressed bundles (`definitive_sota_submission_93.087.zip`, etc.). | Archives |
| **`practice_dsa/`** | Competitive programming binary search practice files. | Practice |
| **`teachme/`** | Socratic ML tutor skill (`SKILL.md`). | Documentation & Tutoring |
| **`progress.md`** | Hackathon progress summary across Phases 1 through 5. | History |

---

## 4. Running the Platform

1. **Verify Backend Tests**:
   ```bash
   python3 -m backend.test_backend
   ```
2. **Start Interactive API Server**:
   ```bash
   python3 -m uvicorn backend.main:app --port 8000 --reload
   ```
3. **Open Documentation**:
   Navigate to `http://localhost:8000/docs` in your browser.
