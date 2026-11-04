# Project Gridlock: Polyglot Database Architecture

This directory contains the production-grade database architecture for **Project Gridlock**, combining **PostgreSQL 16** (for relational entities) and **MongoDB 7.0** (for high-volume ML prediction telemetry).

---

## 1. Quick Start

### Start Databases (Docker Compose)
```bash
docker compose up -d
```

### Seed Both Databases (PostgreSQL + MongoDB)
```bash
./.venv/bin/python3 database/seed_database.py
```

### Run Verification & Health Checks
```bash
./.venv/bin/python3 database/verify_database.py
```

---

## 2. Database Partitioning & Schema Details

```
┌────────────────────────────────────────┐       ┌────────────────────────────────────────┐
│        POSTGRESQL 16 (Port 5433)       │       │         MONGODB 7.0 (Port 27017)       │
├────────────────────────────────────────┤       ├────────────────────────────────────────┤
│ • users (RBAC: Admin, Operator, User)  │       │ • prediction_logs (4,800 Day 49 Docs)  │
│ • zones (50 Geohash-6 cells & bounds)  │       │ • model_registry (SOTA v47 93.087%)    │
│ • traffic_sensor_records (Telemetry)   │       │ • feature_snapshots (Weather, Paradox) │
│ • incident_violations (Chokepoints)    │       │                                        │
│ • enforcement_dispatches (1:1 Officer) │       │ Indexes: Compound (geohash + time),    │
│                                        │       │          TTL 30-Day auto-purge         │
└────────────────────────────────────────┘       └────────────────────────────────────────┘
```

---

## 3. Connection Credentials & Environment

Defined in [`.env`](../.env) and [`.env.example`](../.env.example):

| Database | Host | Port | Database Name | User | Password |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PostgreSQL** | `localhost` | `5433` | `gridlock` | `gridlock_admin` | `gridlock_secret_2026` |
| **MongoDB** | `localhost` | `27017` | `gridlock_predictions` | `mongo_admin` | `mongo_secret_2026` |

*Note: Port 5433 is used for PostgreSQL to prevent port collisions with any pre-existing local PostgreSQL installations on port 5432.*

---

## 4. Seeded Data Summary

- **Users:** 3 accounts (`admin@gridlock.city`, `operator@gridlock.city`, `citizen@gridlock.city`) with PBKDF2 SHA-256 hashed passwords.
- **Zones:** 50 real geohash cells matching `geohash_city_zones.csv`.
- **Incidents & Dispatches:** Pre-populated chokepoints and officer badge dispatches (`KA-TP-1042`).
- **ML Predictions:** 4,800 multi-horizon prediction rollout documents (50 zones $\times$ 96 15-minute time slots across Day 49) with 4-expert contributions and Pluviometer context flags.
