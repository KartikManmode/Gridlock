"""API Routes for Process 2.0: Zone & Sensor Management."""

from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Header, status
from backend.p2_zone_sensors.models import (
    ZoneCreateRequest, ZoneResponse, SensorStreamRecord, SensorRecordResponse
)
from backend.p2_zone_sensors.zone_service import ZoneService
from backend.p2_zone_sensors.sensor_service import SensorService
from backend.p1_auth_rbac.auth_service import AuthService

router = APIRouter(tags=["2.0 Zone & Sensor Management"])


@router.get("/zones", response_model=List[ZoneResponse])
def list_zones():
    return ZoneService.get_all_zones()


@router.get("/zones/{zone_id}", response_model=ZoneResponse)
def get_zone(zone_id: int):
    zone = ZoneService.get_zone_by_id(zone_id)
    if not zone:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Zone not found.")
    return zone


@router.post("/zones", response_model=ZoneResponse, status_code=status.HTTP_201_CREATED)
def create_zone(req: ZoneCreateRequest, authorization: Optional[str] = Header(None)):
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        AuthService.check_permission(token, "zones:manage")
    try:
        return ZoneService.create_zone(req)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/sensors/stream", status_code=status.HTTP_201_CREATED)
def stream_sensor_reading(record: SensorStreamRecord):
    """SensorEntity entrypoint: Ingests real-time traffic sensor feeds into D2."""
    try:
        inserted = SensorService.ingest_reading(record)
        return {"status": "SUCCESS", "message": "Sensor reading recorded", "record": inserted}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
