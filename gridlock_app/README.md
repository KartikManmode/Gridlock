# Project Gridlock: Traffic Congestion Monitoring & Predictive Forecasting Platform

A smart-city urban traffic forecasting and enforcement platform uniting SOTA machine learning (93.087% R²) with a modern polyglot stack: **React** (Frontend), **Node.js** (Backend API Gateway), **PostgreSQL** (General Relational Data), and **MongoDB** (ML Predictions & Multi-Horizon Rollouts).

---

## 🛠️ Technology Stack Summary

- **Frontend:** React 18, Vite, TypeScript, Tailwind CSS, MapLibre GL, deck.gl (GPU polygon rendering), TanStack Query.
- **Backend:** Node.js (LTS 20+), TypeScript, Express.js / Fastify, JWT Authentication, bcrypt, Zod, Socket.io.
- **Database (Relational):** PostgreSQL 16 for core business entities (`users`, `zones`, `traffic_sensor_records`, `incident_violations`, `enforcement_dispatches`).
- **Database (Document / NoSQL):** MongoDB 7.0+ for high-volume ML prediction logs (`prediction_logs`, multi-horizon rollout arrays, expert breakdowns, confidence vectors, TTL expiration).
- **Machine Learning Core:** Python SOTA Engine (93.087% R²) with 4 Domain Experts, Pluviometer Rain Paradox, and Thermo-Spike power scaling ($y^{0.93}$).
- **Detailed Specification:** See [`planning/techstack.md`](planning/techstack.md).

---

## 📁 Repository Structure

```
gridlock/
├── backend/          # Complete production backend platform (FastAPI, P1–P5, ML Engine, DB)
│   ├── p1_auth_rbac/              # Process 1.0: Authentication & RBAC (JWT, roles)
│   ├── p2_zone_sensors/           # Process 2.0: Zone & Sensor Ingestion Pipeline
│   ├── p3_forecasting/            # Process 3.0: Predictive Congestion Forecasting Service
│   ├── ml_engine/                 # Python ML Engine (SOTA 93.087% Pluviometer Ensemble)
│   ├── p4_incidents_dispatch/     # Process 4.0: Incident Violations & Police Dispatches
│   ├── p5_analytics_dashboard/    # Process 5.0: Dynamic Heatmap & Summary KPIs
│   ├── database/                  # SQLite & PostgreSQL DDL, connection pool, seed scripts
│   ├── api/                       # REST route controllers
│   ├── main.py                    # Server application entrypoint
│   └── test_backend.py            # Automated end-to-end verification suite
├── data/             # All raw and structured project datasets
│   ├── dataset/                   # Sensor records (train.csv, test.csv, sample_submission.csv)
│   ├── Phase2/                    # Bangalore Traffic Police violation records & analysis
│   └── artifacts/                 # Prediction CSVs & visualization plots (SHAP, residuals)
├── database/         # Architecture models and DDL specifications
│   ├── Traffic Congestion Monitoring Architecture-2026-11-03-212351.mmd  # ERD diagram
│   └── schema.sql                 # Production PostgreSQL schema
├── frontend/         # React 18 + Vite + TypeScript web application
│   ├── src/components/            # MapLibreHeatmap, TimeSlider, ZoneDrawer, IncidentModal
│   ├── src/services/api.ts        # REST API client with local fallback fixtures
│   └── src/utils/geohash.ts       # Mathematical Geohash-6 bounding polygon engine
├── ml_research/      # ML exploration, expert models, and training logs
│   ├── baselines/                 # Phase 1: linear_reg, advanced_traffic_model, pipelines
│   ├── experiments_gemini/        # Gemini models (v3 to v47 Pluviometer)
│   ├── experiments_codex/         # Deep learning sequence and GNN models
│   ├── experiments_claude/        # Claude pipeline v6
│   ├── sota_submissions/         # SOTA 93.087% and 93.065% winning model packages
│   └── training_logs/             # Training logs and CatBoost artifacts
├── practice_dsa/     # Competitive programming practice files (binary search)
├── archives/         # Competition zip packages and legacy archives
├── docs/             # Documentation and teachme Socratic skill
└── planning/         # Product blueprints, tech stack, roadmap, and design docs
    ├── product_blueprint.md       # Definitive Product Blueprint (Claude v0.2 Refactored)
    ├── techstack.md               # Polyglot Tech Stack & Architecture Decision Records
    ├── roadmap.md                 # Execution Roadmap across Milestones M1–M5
    ├── Planning product blueprint and project design - Claude.pdf # Claude Blueprint
    ├── PROJECT_MAP.md             # Repository Data Flow & File Inventory
    └── progress.md                # Hackathon ML evolution history
```

---

## 🚀 Quickstart

### 1. Run Automated Backend Tests
Verifies all 5 Data Flow processes end-to-end:
```bash
python3 -m backend.test_backend
```

### 2. Start the Backend API Server
```bash
python3 -m uvicorn backend.main:app --port 8000 --reload
```
Interactive Swagger documentation: `http://localhost:8000/docs`.

### 3. Start the React Frontend Application
```bash
cd frontend
npm run dev
```
Open your browser at: `http://localhost:5173`.

### 4. Architecture Diagrams
- **Data Flow Diagram (DFD):** [`backend/Traffic Congestion Forecasting Platform-2026-11-03-212236.mmd`](backend/Traffic%20Congestion%20Forecasting%20Platform-2026-11-03-212236.mmd)
- **Database ERD:** [`database/Traffic Congestion Monitoring Architecture-2026-11-03-212351.mmd`](database/Traffic%20Congestion%20Monitoring%20Architecture-2026-11-03-212351.mmd)
- **Complete Architecture Map:** [`PROJECT_MAP.md`](PROJECT_MAP.md)
