"""Process 3.0: Predictive Congestion Forecasting Models.

Data Transfer Objects for Forecast Requests and Prediction Logs.
Corresponds to D3: prediction_logs Table.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    zone_id: int
    time_window_minutes: int = Field(default=15, ge=5, le=1440)
    weather_condition: Optional[str] = "CLEAR"
    temperature: Optional[float] = 28.0


class ForecastResponse(BaseModel):
    id: int
    zone_id: int
    junction_name: str
    target_timestamp: str
    predicted_volume: float
    congestion_index: float
    model_confidence: float
    congestion_level: str
    recommendation: str


class PredictionLogRecord(BaseModel):
    id: int
    zone_id: int
    requested_by: Optional[int]
    target_timestamp: str
    predicted_volume: float
    congestion_index: float
    model_confidence: float
    created_at: str
