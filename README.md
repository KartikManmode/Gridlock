# Gridlock

> Intelligent Urban Traffic Congestion Forecasting & Rapid Enforcement Dispatch Platform.

Project Gridlock integrates:
- **FastAPI 2.0 Backend**: 5 core DFD processes (Auth/RBAC, Zone & Sensor Telemetry, ML Forecasting, Incident Enforcement, Analytics Dashboard).
- **SOTA 93.087% ML Ensemble**: 4 Domain Experts (Signal, Spatial, Shock, Oracle) with Pluviometer (Rain Paradox) and Thermo-Spike scaling.
- **Polyglot Cloud Storage**: Neon Serverless PostgreSQL (relational operational data) & MongoDB Atlas Cloud (horizon rollout documents).
- **Real-Time Web Frontend**: React 18, Vite, MapLibre GL, and SVG 2D Tactical fallback for live heatmaps and incident triage.

## Architecture

- [`gridlock_app/`](gridlock_app/): Production-ready web application containing `backend/`, `frontend/`, `database/`, and deployment manifests.
- [`other/`](other/): Research notebooks, model competition experiments, training logs, planning blueprints, and DSA practice modules.
