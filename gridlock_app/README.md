# Gridlock Application (Production Tier)

Unified Intelligent Urban Traffic Congestion Forecasting & Rapid Enforcement Dispatch Platform.

## Stack Overview
- **Backend:** Node.js (v20+ / v26) + Express.js 2.0 (ES Modules)
- **Frontend:** React 18 + Vite + Tailwind CSS + MapLibre GL
- **ML Engine:** SOTA 93.087% Ensemble (*Signal, Spatial, Shock, Oracle*) + Pluviometer + Thermo-Spike power transform
- **Relational Store:** Neon Serverless PostgreSQL (Users, Zones, Sensors, Incidents, Dispatches)
- **Document Store:** MongoDB Atlas Cloud (Prediction rollouts & feature snapshots)

## Quick Start

### 1. Install Dependencies
```bash
# Backend
cd backend && npm install

# Frontend
cd ../frontend && npm install
```

### 2. Verify Database Connectivity
```bash
npm run db:verify
```

### 3. Run Test Suite
```bash
npm run test:backend
```

### 4. Start Local Development
```bash
# Terminal 1 - Express API (Port 8000)
npm run dev:backend

# Terminal 2 - Vite UI (Port 5173)
npm run dev:frontend
```
