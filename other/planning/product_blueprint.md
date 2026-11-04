# Gridlock Heatmap: Product Blueprint & System Design

> **Document Version:** v0.2 · Refactored  
> **Source Blueprint:** Adapted directly from Claude's Master Planning Blueprint ([`Planning product blueprint and project design - Claude.pdf`](./Planning%20product%20blueprint%20and%20project%20design%20-%20Claude.pdf))  
> **Core Tagline:** *From a leaderboard CSV to a traffic-operations platform: forecast, see, report, dispatch.*

---

## 1. Project Idea Statement

**Gridlock Heatmap** turns raw machine learning demand forecasts into an operational, role-based traffic-operations platform. Traffic operators see where congestion is coming on a **time-scrubbable map**, log incidents and violations in those zones, and dispatch enforcement officers. Citizens can view the live heatmap and report road problems.

- **The Gap Today:** The forecasting pipeline achieves high accuracy (historically 91.5%–91.8%, now elevated to **93.087% SOTA**), but its only output is a static `Index,demand` CSV. Nobody can act on it.
- **The Bet:** Forecasts become genuinely useful when they sit next to observed traffic and an operational workflow for responding to choke points. The machine learning model runs offline and remains untouched by web traffic.

---

## 2. Product Brief & Role-Based Workflows

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER ROLES (RBAC)                              │
├──────────────────────┬──────────────────────────────┬───────────────────────┤
│        ADMIN         │           OPERATOR           │        CITIZEN        │
│                      │                              │                       │
│ • Manages users &    │ • Reads forecasts on map     │ • Views live heatmap  │
│   role assignments   │ • Triages incident reports   │ • Reports violations  │
│ • System-wide stats  │ • Dispatches officers        │ • Tracks status of    │
│ • Zone configuration │ • Resolves & closes cases    │   own reports         │
└──────────────────────┴──────────────────────────────┴───────────────────────┘
```

### Scope Boundaries

| In (MVP Scope) | Out (Deferred for Future) |
| :--- | :--- |
| ✅ User login with roles (`admin`, `operator`, `citizen`) | ❌ Live or streaming ML inference on web requests |
| ✅ Zone heatmap with **15-minute time slider & playback** | ❌ Model training or retraining from the UI |
| ✅ Hover tooltips and click panels for zone metadata | ❌ Direct physical IoT/sensor hardware integration |
| ✅ **Predicted vs. Observed** time-series chart per zone | ❌ Multi-cloud deployment, SSO, SMS alerting |
| ✅ Incident reporting and lifecycle tracking (`REPORTED` $\to$ `DISPATCHED` $\to$ `RESOLVED`) | |
| ✅ Enforcement officer dispatching and resolution logging | |
| ✅ Automated batch load of predictions and sensor logs | |

---

## 3. Key Architecture Decisions (ADRs)

| Decision | Why (Rationale) |
| :--- | :--- |
| **Model runs offline in batches; the API never trains or predicts** | Ensures flat, sub-10ms latency, stateless API instances, and zero heavy ML dependencies (PyTorch/LightGBM) in the web tier. Batch forecast rows are written with `requested_by` set to `NULL`. |
| **Pure Geohash math (No PostGIS overhead)** | Cell boundary coordinates are derived mathematically from the Geohash-6 string ($1.22\text{ km} \times 0.61\text{ km}$). No spatial queries or PostGIS extensions are required. |
| **Zones store center point only** | The frontend (`deck.gl`) and backend dynamically calculate cell polygons from the geohash. This eliminates polygon data duplication and sync errors. |
| **Polyglot Database Split** | **PostgreSQL** guarantees relational constraints for Users, Zones, and Dispatches. **MongoDB** provides high-throughput document storage for multi-horizon forecast vectors and time-series telemetry. |
| **React + Vite + MapLibre GL + deck.gl** | Open-source, keyless vector basemap tiles with WebGL GPU-drawn cell polygons that maintain smooth 60 FPS transitions during time slider playback. |
| **Node.js (TypeScript) Backend Gateway** | Event-driven, non-blocking asynchronous I/O optimized for concurrent client requests and bi-directional WebSockets for live triage updates. |

---

## 4. Proxy-Data Assumptions & Derivation Logic

Because the competition dataset provides normalized, raw columns without human-readable junction labels, the platform derives realistic operational columns using a single documented ingest pipeline:

1. **Temporal Anchoring:** Day 48 and Day 49 map to a configurable real-world anchor date (e.g. October 19–20, 2026).
2. **`junction_name`:** Derived deterministically from the `geohash` code and city zone tier (e.g. `Silk Board Chokepoint - qp02zt`, `MG Road Junction - qp02z1`).
3. **`capacity`:** Derived from the `NumberofLanes` column ($\text{lanes} \times 500$ vehicles/hour).
4. **`vehicle_count`:** Calculated from observed `demand` multiplied by junction capacity.
5. **`predicted_volume`:** Direct reflection of predicted demand ($[0.0, 1.0]$).
6. **`average_speed`:** Inversely proportional to demand:
   $$\text{speed} = \max(5.0, (1.0 - \text{demand}) \times 60.0)\text{ km/h}$$
7. **`congestion_index`:** Scaled demand: $\text{demand} \times 100$ ($[0.0, 100.0]$).
8. **`model_confidence`:** Derived from inter-expert model variance ($[0.65, 0.98]$) or nullable for legacy rows.
9. **`weather_condition`:** Extracted directly from the real `Weather` column (e.g. `CLEAR`, `RAINY`, `SNOWY`).
10. **Incidents & Dispatches:** Seeded from real Bangalore Traffic Police violation data (`jan to may police violation_anonymized791b166.csv`) and enriched with officer badge numbers.

---

## 5. Endpoints Specification (Claude Blueprint Alignment)

The backend provides typed REST contracts supporting the operational UI:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            API ENDPOINT CONTRACTS                           │
├────────────────────┬──────────┬─────────────────────────────────────────────┤
│ ENDPOINT           │ METHOD   │ PURPOSE                                     │
├────────────────────┼──────────┼─────────────────────────────────────────────┤
│ /api/auth/login    │ POST     │ Authenticate with email/password, return JWT│
│ /api/auth/me       │ GET      │ Current user profile and role permissions   │
│ /api/zones         │ GET      │ List all zones with geohash center points   │
│ /api/heatmap       │ GET      │ Query zone congestion for selected time slot│
│ /api/series        │ GET      │ Predicted vs. Observed time-series per zone │
│ /api/stats         │ GET      │ City-wide KPIs (average speed, chokepoints) │
│ /api/incidents     │ GET/POST │ List active violations / citizen reporting  │
│ /api/dispatch      │ POST     │ Assign officer badge number to an incident  │
│ /api/dispatch/close│ POST     │ Record officer resolution and close case    │
└────────────────────┴──────────┴─────────────────────────────────────────────┘
```

---

## 6. Execution Milestones (M1 – M5)

```
[M1: Data Layer] ──> [M2: API + Auth] ──> [M3: Map UI] ──> [M4: Operations] ──> [M5: Polish]
```

### M1 · Data Layer & Ingest
- Docker Compose setup for **PostgreSQL 16** (relational core) and **MongoDB 7.0** (predictions).
- Database migrations and schemas with strict CHECK constraints.
- Ingestion script (`python -m backend.database.ingest` or Node.js equivalent):
  - Ingest zones from `test.csv`.
  - Ingest sensor readings from `train.csv`.
  - Ingest SOTA 93.087% prediction documents into MongoDB.
  - Ingest real Bangalore police violation clusters into `incident_violations`.

### M2 · API & Role-Based Access Control
- Node.js API Gateway with TypeScript.
- JWT login and role verification (`admin`, `operator`, `citizen`).
- Core endpoints: `/api/zones`, `/api/heatmap`, `/api/series`, `/api/stats`, `/api/incidents`, `/api/dispatch`.
- Sub-10ms response times on `/api/heatmap` using compound database indexing.

### M3 · Interactive Map & Time Playback
- React + Vite application with Tailwind CSS.
- MapLibre GL keyless vector map canvas.
- deck.gl `PolygonLayer` GPU-accelerated geohash cell tiles.
- **15-Minute Playback Slider:** Scrubbing across Day 49 with Play, Pause, and 15-minute step controls.
- Hover tooltips (Junction Name, Geohash, Predicted Volume, Congestion Level).

### M4 · Incident Operations & Dispatch
- **Citizen Portal:** Clean modal to report traffic violations with photo attachment and geolocation.
- **Operator Triage Board:** Real-time list of active chokepoints ranked by congestion risk.
- **Officer Dispatch Dialog:** Dispatch traffic warden by badge number (`KA-TP-XXXX`).
- **Resolution Flow:** Record officer notes, mark case as `RESOLVED`, update dispatch timestamp.

### M5 · Polish & Demonstration
- End-to-end integration tests (Supertest / Vitest / Pytest).
- Single-command developer workflow via `Makefile` (`make dev`, `make test`, `make seed`).
- Comprehensive `README.md` and live demonstration walkthrough video.
