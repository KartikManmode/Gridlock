"""Process 5.0: Analytics & Dashboard Models.

Data Transfer Objects for Executive KPIs, Heatmap Nodes, and System Aggregations.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel


class CityKPIs(BaseModel):
    total_active_zones: int
    average_city_speed_kmh: float
    city_wide_congestion_index: float
    active_incidents_count: int
    dispatched_units_count: int
    recent_forecasts_count: int
    highest_congestion_zone: Optional[str]


class HeatmapPoint(BaseModel):
    zone_id: int
    junction_name: str
    geohash: str
    latitude: float
    longitude: float
    weight: float
    congestion_level: str
    active_violations: int


class DashboardAggregationResponse(BaseModel):
    kpis: CityKPIs
    heatmap: List[HeatmapPoint]
    active_hotspots: List[dict]
