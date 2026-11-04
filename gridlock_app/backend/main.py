"""Traffic Congestion Forecasting & Incident Enforcement Platform.

Main entry point uniting all 5 processes from the Data Flow Diagram:
- P1: 1.0 Authentication & RBAC
- P2: 2.0 Zone & Sensor Management
- P3: 3.0 Predictive Congestion Forecasting (coupled with Python ML Engine)
- P4: 4.0 Incident & Enforcement Dispatch
- P5: 5.0 Analytics & Dashboard Aggregation
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.database.connection import db
from backend.api.routes_auth import router as auth_router
from backend.api.routes_zones_sensors import router as zones_sensors_router
from backend.api.routes_forecasting import router as forecasting_router
from backend.api.routes_incidents import router as incidents_router
from backend.api.routes_analytics import router as analytics_router

app = FastAPI(
    title="Traffic Congestion Forecasting Platform API",
    description="High-performance backend for smart city traffic monitoring, SOTA ML congestion forecasting, and incident enforcement.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all 5 processes
app.include_router(auth_router, prefix="/api")
app.include_router(zones_sensors_router, prefix="/api")
app.include_router(forecasting_router, prefix="/api")
app.include_router(incidents_router, prefix="/api")
app.include_router(analytics_router, prefix="/api")


@app.on_event("startup")
def on_startup():
    db.init_db()


@app.get("/")
def root():
    return {
        "platform": "Project Gridlock Traffic Forecasting System",
        "status": "OPERATIONAL",
        "architecture": {
            "P1": "1.0 Authentication & RBAC",
            "P2": "2.0 Zone & Sensor Management",
            "P3": "3.0 Predictive Congestion Forecasting",
            "P4": "4.0 Incident & Enforcement Dispatch",
            "P5": "5.0 Analytics & Dashboard Aggregation",
            "ML_Engine": "SOTA 93.087% Ensemble (Signal, Spatial, Shock, Oracle + Pluviometer)",
            "Data_Stores": ["D1: users", "D2: zones/sensor records", "D3: prediction_logs", "D4: incidents/dispatches"]
        },
        "docs_url": "/docs"
    }


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)
