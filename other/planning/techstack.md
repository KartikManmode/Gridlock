# Project Gridlock: Polyglot Technology Stack & Architecture

This document defines the complete production technology stack for **Project Gridlock**, featuring a **React** frontend, a **Node.js** application backend, and a **Polyglot Database Architecture** combining **PostgreSQL** (for relational business entities) and **MongoDB** (for ML predictions and multi-horizon time-series logs).

---

## 1. High-Level Architecture Overview

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                                 PRESENTATION TIER                                 │
│        React 18  │  Vite  │  TypeScript  │  Tailwind CSS                          │
│        MapLibre GL (Basemap)  │  deck.gl (GPU Polygons)                           │
│        TanStack Query v5  │  Recharts  │  Lucide Icons                            │
└────────────────────────────────────────┬──────────────────────────────────────────┘
                                         │ HTTPS / REST & WebSockets
┌────────────────────────────────────────▼──────────────────────────────────────────┐
│                                APPLICATION TIER                                   │
│        Node.js (LTS 20+)  │  TypeScript  │  Express.js / Fastify                  │
│        JWT Auth + bcrypt  │  Zod Validation  │  Socket.io (Live Telemetry)        │
└──────────────────┬─────────────────────────────────────────────┬──────────────────┘
                   │                                             │
      ┌────────────▼──────────────┐                ┌─────────────▼──────────────┐
      │  RELATIONAL DATABASE      │                │  DOCUMENT DATABASE (NOSQL) │
      │  PostgreSQL 16            │                │  MongoDB 7.0+              │
      │  (General Business Data)  │                │  (ML Predictions & Lags)   │
      ├───────────────────────────┤                ├────────────────────────────┤
      │ • USERS                   │                │ • PREDICTION_LOGS          │
      │ • ZONES                   │                │ • MULTI-HORIZON ROLLOUTS   │
      │ • TRAFFIC_SENSOR_RECORDS  │                │ • EXPERT CONTRIBUTIONS     │
      │ • INCIDENT_VIOLATIONS     │                │ • FEATURE VECTOR SNAPSHOTS │
      │ • ENFORCEMENT_DISPATCHES  │                │ • TTL HISTORICAL ARCHIVES  │
      └───────────────────────────┘                └─────────────▲──────────────┘
                                                                 │ Bulk Ingest / RPC
                                                   ┌─────────────┴──────────────┐
                                                   │    PYTHON ML ENGINE        │
                                                   │    SOTA Model v47 (93.087%)│
                                                   │    PyTorch, LightGBM,      │
                                                   │    XGBoost, Pluviometer    │
                                                   └────────────────────────────┘
```

---

## 2. Core Technology Matrix

| Layer | Component | Chosen Technology | Version / Spec | Justification & Role |
| :--- | :--- | :--- | :--- | :--- |
| **Frontend** | Framework | **React** | 18.x / 19.x | Declarative component UI, concurrent rendering, rich ecosystem. |
| **Frontend** | Tooling & Build | **Vite** | 5.x | Instant Hot Module Replacement (HMR), optimized Rollup bundle. |
| **Frontend** | Language | **TypeScript** | 5.x | Strict end-to-end type safety shared across UI and backend contracts. |
| **Frontend** | Styling | **Tailwind CSS** | 3.4+ | Utility-first CSS, responsive dark-mode layout for control-room UI. |
| **Frontend** | Map Basemap | **MapLibre GL** | 4.x | Open-source, keyless vector map tiles (no API token / billing limits). |
| **Frontend** | Geospatial Rendering| **deck.gl** | 9.x | WebGL GPU-accelerated polygon rendering for smooth 60 FPS time scrubbing. |
| **Frontend** | Server State | **TanStack Query** | 5.x | Stale-while-revalidate caching, automatic background refetching. |
| **Frontend** | Charts | **Recharts** | 2.x | Predicted vs. observed volume trends and zone demand curves. |
| **Backend** | Runtime | **Node.js** | LTS 20.x | High-throughput non-blocking I/O event loop for asynchronous traffic. |
| **Backend** | Web Framework | **Express.js / Fastify** | Latest | Lightweight, robust HTTP route handling and middleware chaining. |
| **Backend** | Language | **TypeScript** | 5.x | Type safety, clean interfaces, ES modules. |
| **Backend** | Validation | **Zod** | 3.x | Schema validation and runtime type inference for client inputs. |
| **Backend** | Security & Auth | **JWT + bcrypt** | Latest | Stateless bearer token authentication, salted password hashing. |
| **Backend** | Real-Time Push | **Socket.io / ws** | Latest | Bi-directional WebSocket stream for live chokepoint alerts and sensor deltas. |
| **Database (SQL)** | Relational Store | **PostgreSQL** | 16.x | ACID transactions, strict relational integrity (Users, Zones, Incidents). |
| **Database (SQL)** | ORM / Driver | **Prisma / pg-pool** | Latest | Type-safe query building, connection pooling, automated migrations. |
| **Database (NoSQL)**| Document Store | **MongoDB** | 7.0+ | High-throughput document storage for multi-slot ML forecasts & vectors. |
| **Database (NoSQL)**| ODM / Driver | **Mongoose / Mongo Native**| Latest | Flexible document schemas, compound indexes, TTL expiration. |
| **ML Engine** | SOTA Model | **Winning Model v47** | 93.087% R² | Pluviometer Zonal Ensemble (Signal, Spatial, Shock, Oracle). |
| **ML Stack** | Scientific Python | **Python 3.9+** | Latest | PyTorch, LightGBM, XGBoost, NumPy, Pandas, Scikit-Learn. |
| **DevOps** | Containerization | **Docker & Compose** | Compose v2 | Multi-container setup (React, Node, Postgres, MongoDB). |
| **DevOps** | Automation | **Makefile / npm scripts** | Standard | `npm run dev`, `npm test`, `npm run seed`. |

---

## 3. Database Partitioning: PostgreSQL vs. MongoDB

Project Gridlock uses a **Polyglot Persistence Architecture**, routing data to the storage engine matching its access pattern and schema dynamics.

### 3.1. PostgreSQL: General Relational Business Data
PostgreSQL handles entities that require strict referential integrity, foreign key cascading, and relational consistency:

```
[USERS] 1 ────< [INCIDENT_VIOLATIONS] 1 ──── 0..1 [ENFORCEMENT_DISPATCHES]
   │                      │
   │                      │
   v                      v
[PREDICTION_LOGS]*     [ZONES] 1 ────< [TRAFFIC_SENSOR_RECORDS]
(*Logical foreign key reference to MongoDB _id)
```

1. **`users` (D1):**
   - User identity, credentials, roles (`admin`, `operator`, `citizen`).
   - Relational rule: Deleting a user sets `reported_by = NULL` (preserves historical accountability).
2. **`zones` (D2):**
   - Geographic junctions, Geohash-6 codes, central `(latitude, longitude)`, capacity, and current congestion status (`LOW`, `MEDIUM`, `HIGH`, `SEVERE`).
3. **`traffic_sensor_records` (D2):**
   - Time-series sensor inputs: `zone_id`, `vehicle_count`, `average_speed`, `weather_condition`, `recorded_at`.
4. **`incident_violations` (D4):**
   - Reported citizen and sensor infractions: `violation_type` (`SPEEDING`, `PARKING`, `SIGNAL`), `severity` (`LOW`, `MEDIUM`, `HIGH`), `status`.
5. **`enforcement_dispatches` (D4):**
   - 1:1 dispatched officer assignments: `incident_id` (UNIQUE), `officer_badge_number`, `dispatch_time`, `resolution_notes`, `resolved_at`.

---

### 3.2. MongoDB: High-Volume ML Predictions & Telemetry
ML predictions generate large volumes of hierarchical, semi-structured documents (e.g. 50+ zones $\times$ 96 slots/day $\times$ 14-hour rollouts $\times$ 4 expert breakdowns). Storing these in PostgreSQL causes table bloat and rigid schema migrations. MongoDB stores them as native documents:

#### `prediction_logs` Document Schema:
```json
{
  "_id": "ObjectId('67285a1f2e10a8b94101e402')",
  "zone_id": 14,
  "geohash": "qp02zt",
  "junction_name": "Silk Board Chokepoint",
  "requested_by": 2,
  "target_timestamp": "2026-11-04T09:15:00.000Z",
  "predicted_volume": 0.7420,
  "congestion_index": 74.20,
  "congestion_level": "SEVERE",
  "model_confidence": 0.9308,
  "model_version": "v47_pluviometer",
  
  "expert_breakdown": {
    "signal_expert": 0.7100,
    "spatial_expert": 0.7850,
    "shock_expert": 0.8200,
    "oracle_expert": 0.6900
  },
  
  "multi_horizon_rollout": [
    { "slot": "+15m", "volume": 0.7420, "index": 74.2 },
    { "slot": "+30m", "volume": 0.7810, "index": 78.1 },
    { "slot": "+45m", "volume": 0.8150, "index": 81.5 },
    { "slot": "+60m", "volume": 0.7600, "index": 76.0 }
  ],
  
  "context_snapshot": {
    "weather": "Rainy",
    "temperature": 27.5,
    "rain_paradox_applied": "CORE_DAMPENING",
    "thermo_spike_applied": true,
    "active_incident_count": 2
  },
  
  "created_at": "2026-11-04T09:00:00.000Z"
}
```

#### MongoDB Optimization Strategies:
- **Compound Spatial-Temporal Index:**
  `db.prediction_logs.createIndex({ geohash: 1, target_timestamp: 1 })`
  Enables sub-5ms lookups when the frontend time-slider queries forecasts for a specific 15-minute slot.
- **TTL Index (Time-To-Live):**
  `db.prediction_logs.createIndex({ created_at: 1 }, { expireAfterSeconds: 2592000 })`
  Automatically purges or archives raw 15-minute prediction payloads after 30 days, keeping the database fast and lean.

---

## 4. Architectural Decision Records (ADRs)

### ADR-01: Offline Batch Forecast Ingestion (Claude Blueprint Principle)
- **Status:** Accepted
- **Context:** Claude's planning blueprint emphasizes: *"Model runs offline in batches; the API never trains or predicts. Flat latency, stateless API, no ML dependencies in web tier. Batch rows are written with `requested_by` empty."*
- **Decision:**
  - **Batch Prediction Pipeline:** The Python ML pipeline runs offline, processes Day 49 test keys, and outputs prediction documents directly into **MongoDB**.
  - **Node.js Web Tier:** Completely stateless. Never executes heavy matrix operations or model retraining during user requests. Queries pre-computed prediction documents from MongoDB using compound indexes (`geohash + target_timestamp`) for sub-10ms time-slider playback.
  - **Optional Ad-Hoc Worker:** Retained only as an asynchronous background worker for simulated "What-If" scenarios.

### ADR-02: Geohash-Native Geometry (No PostGIS Dependency)
- **Status:** Accepted
- **Context:** Claude's blueprint establishes: *"PostgreSQL, no PostGIS. Cell edges come from the geohash string, so no spatial queries are needed. Zones store center point only; the frontend and API compute cell polygons from the geohash."*
- **Decision:** Store only central `latitude` and `longitude` in PostgreSQL `zones`. Let the client (`deck.gl`) and Node.js backend calculate the cell bounding box vertices on the fly using standard geohash formulas:
  $$\text{geohash-6} \to (\text{lat}_{\min}, \text{lat}_{\max}, \text{lon}_{\min}, \text{lon}_{\max})$$
- **Consequences:** Eliminates PostGIS dependencies, drastically simplifies Docker builds, and speeds up query throughput.

### ADR-03: Polyglot Persistence: PostgreSQL (General Relational) + MongoDB (ML Predictions)
- **Status:** Accepted
- **Context:** Claude's blueprint specified 6 relational tables with strict database-level integrity (CHECK constraints, foreign keys, unique dispatch 1:1 linkage). However, ML predictions generate multi-slot time-series rollouts, expert contribution arrays, and confidence logs that cause relational table bloat.
- **Decision:**
  - **PostgreSQL 16:** Houses `users`, `zones`, `traffic_sensor_records`, `incident_violations`, and `enforcement_dispatches`. Enforces relational integrity rules:
    - Deleting a user sets `reported_by = NULL` (preserves historical accountability).
    - Deleting a zone removes its sensor logs (`ON DELETE CASCADE`).
    - One dispatch per incident enforced by `UNIQUE (incident_id)`.
  - **MongoDB 7.0+:** Houses `prediction_logs` with multi-horizon rollout arrays, expert breakdowns, and TTL automatic 30-day expiration.

### ADR-04: React + MapLibre GL + deck.gl
- **Status:** Accepted
- **Context:** As specified in Claude's blueprint: *"React + Vite, MapLibre, deck.gl. Keyless basemap and GPU-drawn cell polygons that stay smooth during playback."*
- **Decision:** Use **MapLibre GL** for keyless vector map tiles, coupled with **deck.gl `PolygonLayer`** for GPU-accelerated WebGL polygon rendering.
- **Consequences:** Renders silky-smooth 60 FPS transitions during playback while scrubbing the 15-minute time slider.

---

## 5. Development & Deployment Stack

```
                                  [ Client Browser ]
                                          │
                         ┌────────────────┴────────────────┐
                         │                                 │
                   (Static Bundle)                   (API Requests)
                         │                                 │
                         v                                 v
                 [ React + Vite UI ]             [ Node.js API Gateway ]
               (Port 3000 / 5173)                  (Port 4000 / 8000)
                                                           │
                                        ┌──────────────────┴──────────────────┐
                                        │                                     │
                                        v                                     v
                             [ PostgreSQL 16 ]                         [ MongoDB 7.0 ]
                             (Port 5432: Core)                        (Port 27017: ML)
```

| Environment | Frontend | Backend | Databases | ML Ingestion |
| :--- | :--- | :--- | :--- | :--- |
| **Local Dev** | `npm run dev` (Vite, port 5173) | `npm run dev` (tsx/ts-node, port 4000) | Docker Compose (Postgres + Mongo) | Python batch script (`python -m ml_engine.ingest`) |
| **CI / Testing**| Vitest / React Testing Library | Jest / Supertest | Ephemeral Docker containers | Pre-computed fixtures |
| **Production** | Static CDN (Vercel / Cloudflare) | Container (Node.js Alpine) | Managed RDS Postgres + Atlas Mongo | Scheduled AWS Lambda / Cloud Run Worker |
