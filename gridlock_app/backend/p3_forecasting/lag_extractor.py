"""Lag Feature Extractor for Predictive Congestion Forecasting.

Queries historical sensor records from D2 and builds the lag feature vector
forwarded to the Python ML Engine.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict
from backend.database.connection import db
from backend.p2_zone_sensors.zone_service import ZoneService


class LagExtractor:
    """Extracts temporal lag features from D2 sensor records."""

    @classmethod
    def build_feature_vector(
        cls,
        zone_id: int,
        target_dt: datetime.datetime,
        weather: str = "CLEAR",
        temperature: float = 28.0
    ) -> Dict[str, Any]:
        zone = ZoneService.get_zone_metadata(zone_id)
        capacity = max(1, zone["capacity"])

        # Query recent sensor records (D2)
        records = db.fetch_all(
            """SELECT vehicle_count, average_speed, weather_condition, recorded_at
               FROM traffic_sensor_records
               WHERE zone_id = ?
               ORDER BY recorded_at DESC LIMIT 5""",
            (zone_id,)
        )

        lag_15m = 0.35
        lag_30m = 0.35
        rolling_mean = 0.35

        if records:
            demands = [r["vehicle_count"] / float(capacity) for r in records]
            lag_15m = demands[0]
            lag_30m = demands[1] if len(demands) > 1 else demands[0]
            rolling_mean = sum(demands) / len(demands)

        # Query active incidents (D4) for shock impact
        active_incidents = db.fetch_all(
            """SELECT id FROM incident_violations
               WHERE zone_id = ? AND status IN ('REPORTED', 'DISPATCHED')""",
            (zone_id,)
        )

        return {
            "zone_id": zone_id,
            "geohash": zone["geohash"],
            "latitude": zone["latitude"],
            "longitude": zone["longitude"],
            "capacity": capacity,
            "zone_tier": zone.get("zone_tier", "Core City"),
            "hour": target_dt.hour,
            "minute": target_dt.minute,
            "lag_15m_demand": lag_15m,
            "lag_30m_demand": lag_30m,
            "lag_24h_demand": rolling_mean,  # Proxy for 24h seasonal anchor
            "rolling_mean_demand": rolling_mean,
            "weather_condition": weather,
            "temperature": temperature,
            "active_incidents": len(active_incidents),
            "is_artery": zone["capacity"] >= 2000
        }
