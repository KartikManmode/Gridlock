"""Process 2.0: Zone Management Service.

CRUD and query operations for urban traffic monitoring zones.
Corresponds to D2: zones Table.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from backend.database.connection import db
from backend.p2_zone_sensors.models import ZoneCreateRequest, CongestionLevel
from backend.p2_zone_sensors.geospatial import decode_geohash, classify_zone_tier


class ZoneService:
    """Manages urban zones, capacities, and live congestion states."""

    @classmethod
    def get_all_zones(cls) -> List[Dict[str, Any]]:
        return db.fetch_all("SELECT * FROM zones ORDER BY junction_name ASC")

    @classmethod
    def get_zone_by_id(cls, zone_id: int) -> Optional[Dict[str, Any]]:
        return db.fetch_one("SELECT * FROM zones WHERE id = ?", (zone_id,))

    @classmethod
    def get_zone_by_geohash(cls, geohash: str) -> Optional[Dict[str, Any]]:
        return db.fetch_one("SELECT * FROM zones WHERE geohash = ?", (geohash,))

    @classmethod
    def create_zone(cls, req: ZoneCreateRequest) -> Dict[str, Any]:
        existing = cls.get_zone_by_geohash(req.geohash)
        if existing:
            raise ValueError(f"Zone with geohash '{req.geohash}' already exists (ID: {existing['id']}).")
        
        zone_id = db.execute(
            """INSERT INTO zones (geohash, junction_name, latitude, longitude, capacity, current_congestion_level)
               VALUES (?, ?, ?, ?, ?, 'LOW')""",
            (req.geohash, req.junction_name, req.latitude, req.longitude, req.capacity)
        )
        return cls.get_zone_by_id(zone_id)

    @classmethod
    def update_congestion_level(cls, zone_id: int, level: CongestionLevel) -> None:
        db.execute(
            "UPDATE zones SET current_congestion_level = ? WHERE id = ?",
            (level.value, zone_id)
        )

    @classmethod
    def get_zone_metadata(cls, zone_id: int) -> Dict[str, Any]:
        zone = cls.get_zone_by_id(zone_id)
        if not zone:
            raise ValueError(f"Zone with ID {zone_id} not found.")
        tier = classify_zone_tier(zone["latitude"], zone["longitude"])
        return {**zone, "zone_tier": tier}
