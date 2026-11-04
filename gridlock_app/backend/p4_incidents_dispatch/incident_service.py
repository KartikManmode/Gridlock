"""Process 4.0: Incident & Violation Service.

Handles citizen and operator violation reporting into D4 (incident_violations).
Returns alerts and confirmations to UserEntity.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from backend.database.connection import db
from backend.p4_incidents_dispatch.models import IncidentReportRequest, IncidentResponse, IncidentStatus, SeverityLevel
from backend.p2_zone_sensors.zone_service import ZoneService


class IncidentService:
    """Manages violation reporting and tracking."""

    @classmethod
    def report_violation(
        cls,
        req: IncidentReportRequest,
        reported_by_user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        zone = ZoneService.get_zone_by_id(req.zone_id)
        if not zone:
            raise ValueError(f"Zone ID {req.zone_id} not found.")

        severity = req.severity or SeverityLevel.MEDIUM

        incident_id = db.execute(
            """INSERT INTO incident_violations (zone_id, reported_by, violation_type, severity, status)
               VALUES (?, ?, ?, ?, 'REPORTED')""",
            (req.zone_id, reported_by_user_id, req.violation_type.value, severity.value)
        )

        record = db.fetch_one(
            """SELECT i.*, z.junction_name 
               FROM incident_violations i
               JOIN zones z ON i.zone_id = z.id
               WHERE i.id = ?""",
            (incident_id,)
        )
        return record

    @classmethod
    def get_incidents(cls, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        if status:
            return db.fetch_all(
                """SELECT i.*, z.junction_name, z.geohash, d.officer_badge_number, d.dispatch_time
                   FROM incident_violations i
                   JOIN zones z ON i.zone_id = z.id
                   LEFT JOIN enforcement_dispatches d ON i.id = d.incident_id
                   WHERE i.status = ?
                   ORDER BY i.occurred_at DESC LIMIT ?""",
                (status, limit)
            )
        return db.fetch_all(
            """SELECT i.*, z.junction_name, z.geohash, d.officer_badge_number, d.dispatch_time
               FROM incident_violations i
               JOIN zones z ON i.zone_id = z.id
               LEFT JOIN enforcement_dispatches d ON i.id = d.incident_id
               ORDER BY i.occurred_at DESC LIMIT ?""",
            (limit,)
        )
