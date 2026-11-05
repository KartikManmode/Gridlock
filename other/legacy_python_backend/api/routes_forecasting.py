"""API Routes for Process 3.0: Predictive Congestion Forecasting."""

from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Header, status
from backend.p3_forecasting.models import ForecastRequest, ForecastResponse
from backend.p3_forecasting.forecasting_service import ForecastingService
from backend.p1_auth_rbac.auth_service import AuthService

router = APIRouter(prefix="/forecast", tags=["3.0 Predictive Congestion Forecasting"])


@router.post("", response_model=ForecastResponse)
def request_traffic_forecast(req: ForecastRequest, authorization: Optional[str] = Header(None)):
    """UserEntity entrypoint: Requests congestion forecast for Zone ID and Time Window."""
    user_id = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        payload = AuthService.verify_token(token)
        if payload:
            user_id = payload.get("sub")
    try:
        return ForecastingService.generate_forecast(req, requested_by_user_id=user_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/logs")
def get_prediction_logs(limit: int = 50):
    """Retrieves recent logged predictions from D3."""
    return ForecastingService.get_recent_predictions(limit=limit)


@router.get("/atlas-rollouts")
def get_atlas_rollouts(target_timestamp: Optional[str] = None):
    """Retrieves precomputed Day 49 multi-horizon rollouts from MongoDB Atlas."""
    from database.mongo_connection import mongo_repo
    if not mongo_repo.is_connected():
        return {"connected": False, "count": 0, "rollouts": []}
    if target_timestamp:
        data = mongo_repo.get_predictions_for_slot(target_timestamp)
    else:
        data = list(mongo_repo.prediction_logs.find({}, {"_id": 0}).limit(50))
    return {"connected": True, "count": len(data), "rollouts": data}


@router.get("/atlas-series/{zone_id}")
def get_atlas_zone_series(zone_id: int, limit: int = 96):
    """Retrieves chronological forecast time-series for a zone from MongoDB Atlas."""
    from database.mongo_connection import mongo_repo
    if not mongo_repo.is_connected():
        return {"connected": False, "series": []}
    series = mongo_repo.get_zone_forecast_series(zone_id=zone_id, limit=limit)
    return {"connected": True, "zone_id": zone_id, "count": len(series), "series": series}
