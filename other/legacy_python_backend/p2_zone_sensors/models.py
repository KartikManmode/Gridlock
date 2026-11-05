"""Process 2.0: Zone & Sensor Management Models.

Data Transfer Objects for Zones and Traffic Sensors.
Corresponds to D2: zones & traffic_sensor_records Tables.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class CongestionLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    SEVERE = "SEVERE"


class ZoneCreateRequest(BaseModel):
    geohash: str = Field(..., min_length=4, max_length=12)
    junction_name: str
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    capacity: int = Field(..., gt=0)


class ZoneResponse(BaseModel):
    id: int
    geohash: str
    junction_name: str
    latitude: float
    longitude: float
    capacity: int
    current_congestion_level: CongestionLevel


class SensorStreamRecord(BaseModel):
    zone_id: Optional[int] = None
    geohash: Optional[str] = None
    vehicle_count: int = Field(..., ge=0)
    average_speed: float = Field(..., ge=0.0)
    weather_condition: str = "CLEAR"


class SensorRecordResponse(BaseModel):
    id: int
    zone_id: int
    recorded_at: str
    vehicle_count: int
    average_speed: float
    weather_condition: str
