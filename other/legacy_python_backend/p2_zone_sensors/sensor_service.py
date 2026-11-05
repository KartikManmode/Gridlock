"""Process 2.0: Sensor Stream Ingestion Service.

Ingests streaming sensor logs, writes to D2 (traffic_sensor_records),
and dynamically updates the zone's current_congestion_level.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from backend.database.connection import db
from backend.p2_zone_sensors.models import SensorStreamRecord, CongestionLevel
from backend.p2_zone_sensors.zone_service import ZoneService


class SensorService:
    """Handles sensor ingestion streams and recalculates congestion."""

    @classmethod
    def ingest_reading(cls, record: SensorStreamRecord) -> Dict[str, Any]:
        """Ingests a single reading from SensorEntity into D2."""
        zone_id = record.zone_id
        if zone_id is None:
            if not record.geohash:
                raise ValueError("Must provide either zone_id or geohash.")
            zone = ZoneService.get_zone_by_geohash(record.geohash)
            if not zone:
                raise ValueError(f"No registered zone found for geohash '{record.geohash}'.")
            zone_id = zone["id"]
        else:
            zone = ZoneService.get_zone_by_id(zone_id)
            if not zone:
                raise ValueError(f"No registered zone found for ID {zone_id}.")

        # Insert sensor record into D2
        record_id = db.execute(
            """INSERT INTO traffic_sensor_records (zone_id, vehicle_count, average_speed, weather_condition)
               VALUES (?, ?, ?, ?)""",
            (zone_id, record.vehicle_count, record.average_speed, record.weather_condition.upper())
        )

        # Dynamic congestion level calculation
        capacity = max(1, zone["capacity"])
        occupancy_ratio = record.vehicle_count / float(capacity)
        
        if occupancy_ratio >= 0.90:
            level = CongestionLevel.SEVERE
        elif occupancy_ratio >= 0.65:
            level = CongestionLevel.HIGH
        elif occupancy_ratio >= 0.35:
            level = CongestionLevel.MEDIUM
        else:
            level = CongestionLevel.LOW

        ZoneService.update_congestion_level(zone_id, level)

        return db.fetch_one("SELECT * FROM traffic_sensor_records WHERE id = ?", (record_id,))

    @classmethod
    def get_recent_readings(cls, zone_id: int, limit: int = 20) -> List[Dict[str, Any]]:
        """Queries recent historical sensor logs for lag calculations."""
        return db.fetch_all(
            """SELECT * FROM traffic_sensor_records 
               WHERE zone_id = ? 
               ORDER BY recorded_at DESC LIMIT ?""",
            (zone_id, limit)
        )
