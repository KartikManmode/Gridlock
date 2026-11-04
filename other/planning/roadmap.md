# Project Gridlock: Comprehensive Roadmap & Execution Plan

This document synthesizes Claude's Product Blueprint with our SOTA 93.087% ML Engine into a unified, actionable roadmap from current state to production.

---

## 1. Where We Are Currently

1. **Machine Learning Foundation:**
   - SOTA Winning Model v47 achieving **93.087% R²**.
   - 4 Domain Experts: Signal (V1), Spatial Fourier (V5), Shock (V6), and Day 48 Oracle (V4).
   - Zonal Pluviometer Engine (Rain Paradox) + Thermo-Spike power transform ($y^{0.93}$).
   - Ensemble variance model confidence calculation ($0.65$ – $0.98$).

2. **Empirical Domain Data:**
   - Real Bangalore Traffic Police violation records (Jan–May) parsed for chokepoints and parking clusters.

3. **Backend Implementation (`backend/`):**
   - Directly implements the Mermaid Data Flow Diagram:
     - `p1_auth_rbac/`: Authentication, PBKDF2 hashing, JWT tokens, role checks.
     - `p2_zone_sensors/`: Ingestion pipeline, geohash spatial decoding, live congestion levels.
     - `p3_forecasting/`: Lag extraction, ML Engine queries, prediction logging.
     - `ml_engine/`: In-memory 4-expert ensemble + Pluviometer post-processing.
     - `p4_incidents_dispatch/`: Violation reports, officer dispatches, hotspot ranking.
     - `p5_analytics_dashboard/`: Dynamic geospatial heatmap points, executive city KPIs.
   - Database schemas (SQLite & PostgreSQL) matching the 6 ERD tables.
   - End-to-end test suite (`python3 -m backend.test_backend`) verified.

---

## 2. Milestones M1 – M5 (Execution Plan)

```
[M1: Data & Ingest] ──> [M2: API & Auth] ──> [M3: Map & Playback] ──> [M4: Operations] ──> [M5: Polish]
```

### Milestone M1 · Data & Polyglot Persistence
- **Objective:** Establish the persistent database infrastructure (PostgreSQL + MongoDB) and load baseline datasets.
- **Tasks:**
  1. Set up Docker Compose configuration for **PostgreSQL 16** (general data) and **MongoDB 7.0** (ML prediction logs).
  2. Implement database seeding and ingestion scripts:
     - Load 50 Bangalore zones into PostgreSQL `zones` table.
     - Load sensor readings into PostgreSQL `traffic_sensor_records` table.
     - Seed real Bangalore police violation records into PostgreSQL `incident_violations` table.
     - Batch-ingest SOTA 93.087% prediction documents into MongoDB `prediction_logs` collection (including multi-horizon rollout arrays, expert breakdowns, and confidence scores).
- **Deliverable:** Working Docker Compose environment with both PostgreSQL and MongoDB populated.

### Milestone M2 · Node.js API & Role-Based Access Control
- **Objective:** Build the Node.js API Gateway with TypeScript, typed contracts, and authentication.
- **Tasks:**
  1. Connect Node.js backend to PostgreSQL (via Prisma / pg pool) and MongoDB (via Mongoose / Mongo Native driver).
  2. Implement JWT authentication and role checks (`admin`, `operator`, `citizen`).
  3. Expose operational REST endpoints:
     - `/api/auth`: Login, register, profile.
     - `/api/zones`: Junctions and capacity limits.
     - `/api/heatmap`: Aggregate zones from Postgres with latest prediction documents from MongoDB.
     - `/api/incidents` & `/api/dispatch`: Violation reporting and officer deployment.
- **Deliverable:** Functional Node.js backend with full Swagger / OpenAPI documentation.

### Milestone M3 · React Interactive Map & Time Playback
- **Objective:** Build the interactive, time-scrubbable map frontend.
- **Tasks:**
  1. Initialize React + Vite application with Tailwind CSS and TypeScript.
  2. Integrate MapLibre GL for keyless vector map tiles.
  3. Implement deck.gl `PolygonLayer` to render geohash bounding boxes with GPU acceleration.
  4. Build the **15-Minute Playback Slider** with Play/Pause controls across Day 49 (00:00 to 23:45) querying MongoDB pre-computed forecasts.
- **Deliverable:** Smooth, 60 FPS congestion playback in the browser.

### Milestone M4 · Incident Operations & Dispatch
- **Objective:** Connect citizen reporting with operator enforcement dispatch.
- **Tasks:**
  1. Build Citizen reporting drawer (photo upload, coordinates, violation category).
  2. Build Operator incident triage board with active chokepoints and risk scores.
  3. Implement One-Click Officer Dispatch dialog (officer badge number, dispatch notes).
  4. Incident resolution workflow with resolution notes and timestamp.
- **Deliverable:** Functional end-to-end incident reporting and dispatch cycle.

### Milestone M5 · Polish, Testing & Demonstration
- **Objective:** Production readiness and walkthrough.
- **Tasks:**
  1. Root `Makefile` (`make dev`, `make test`, `make seed`, `make build`).
  2. Vitest unit tests for frontend components; Pytest suite for backend.
  3. Record video demonstration showing the full forecast-to-dispatch workflow.
- **Deliverable:** Fully documented, production-ready repository.
