# Project Gridlock Backend Architecture

This backend implements the complete **Traffic Congestion Forecasting Platform** specified in [`Traffic Congestion Forecasting Platform-2026-11-03-212236.mmd`](./Traffic%20Congestion%20Forecasting%20Platform-2026-11-03-212236.mmd) and the Database ERD in [`database/Traffic Congestion Monitoring Architecture-2026-11-03-212351.mmd`](../database/Traffic%20Congestion%20Monitoring%20Architecture-2026-11-03-212351.mmd).

---

## 1. Data Flow Mapping

| DFD Component | Directory / Module | Role & Implementation | Data Store |
| :--- | :--- | :--- | :--- |
| **P1: 1.0 Authentication & RBAC** | [`p1_auth_rbac/`](./p1_auth_rbac/) | PBKDF2 SHA-256 password hashing, signed JWT tokens, role-based access control (Admin, Operator, Citizen). | `D1: users` |
| **P2: 2.0 Zone & Sensor Management** | [`p2_zone_sensors/`](./p2_zone_sensors/) | Streaming sensor feed ingestion (`SensorEntity`), geohash decoding, city tier classification (Core vs Non-Core), live congestion tracking. | `D2: zones` & `traffic_sensor_records` |
| **P3: 3.0 Predictive Forecasting** | [`p3_forecasting/`](./p3_forecasting/) | Handles user forecast requests (Zone ID, Time Window), selects historical lag features from D2, queries ML Engine, logs forecasts. | `D3: prediction_logs` |
| **MLEngine: Python ML Engine** | [`ml_engine/`](./ml_engine/) | 93.087% SOTA ensemble combining 4 Experts (Signal 35%, Spatial 25%, Shock 20%, Oracle 20%) + Pluviometer Rain Paradox + Thermo-Spike 0.93 power. | In-memory / Models |
| **P4: 4.0 Incident & Dispatch** | [`p4_incidents_dispatch/`](./p4_incidents_dispatch/) | Violation reporting (Speeding, Parking, Signal), severity evaluation, enforcement unit dispatch, Phase 2 hotspot analysis. | `D4: incident_violations` & `enforcement_dispatches` |
| **P5: 5.0 Analytics & Dashboard** | [`p5_analytics_dashboard/`](./p5_analytics_dashboard/) | Joins D2 + D3 + D4 to generate city-wide KPIs (average speed, congestion index) and dynamic geospatial heatmaps for UserEntity. | Aggregated D2, D3, D4 |
| **Database Layer** | [`database/`](./database/) | SQLite and PostgreSQL schemas, connection pool, and automated seed script loading real project data. | All Data Stores |
| **API Entrypoint** | [`api/`](./api/) & [`main.py`](./main.py) | Unified FastAPI application with interactive documentation (`/docs`). | REST Controllers |

---

## 2. Quickstart & Verification

### Running the End-to-End Test Suite:
```bash
python3 -m backend.test_backend
```

### Running the API Server:
```bash
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation will be available at: `http://localhost:8000/docs`.

### Seeding Initial Data:
```bash
python3 -m backend.database.seed_data
```
Populates the database with:
- 3 Default Users (`admin@gridlock.city`, `operator@gridlock.city`, `citizen@gridlock.city`)
- 50 Bangalore urban zones mapped from `geohash_city_zones.csv`
- Sensor readings from `dataset/train.csv`
- Incident violations and officer dispatches from Phase 2
- Predictive forecast logs
