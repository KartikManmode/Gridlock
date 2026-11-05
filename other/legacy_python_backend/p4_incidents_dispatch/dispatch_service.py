"""Process 4.0: Enforcement Dispatch Service.

Handles dispatching traffic police / enforcement units to violation chokepoints
and logging resolution notes into D4 (enforcement_dispatches).
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from backend.database.connection import db
from backend.p4_incidents_dispatch.models import DispatchUnitRequest, ResolveIncidentRequest


class DispatchService:
    """Manages police unit dispatch and resolution tracking."""

    @classmethod
    def dispatch_unit(cls, req: DispatchUnitRequest) -> Dict[str, Any]:
        incident = db.fetch_one("SELECT * FROM incident_violations WHERE id = ?", (req.incident_id,))
        if not incident:
            raise ValueError(f"Incident ID {req.incident_id} does not exist.")

        # Check if already dispatched
        existing = db.fetch_one("SELECT * FROM enforcement_dispatches WHERE incident_id = ?", (req.incident_id,))
        if existing:
            raise ValueError(f"Enforcement unit already dispatched to Incident ID {req.incident_id}.")

        dispatch_id = db.execute(
            """INSERT INTO enforcement_dispatches (incident_id, officer_badge_number, resolution_notes)
               VALUES (?, ?, ?)""",
            (req.incident_id, req.officer_badge_number, req.notes)
        )

        # Update incident status to DISPATCHED
        db.execute(
            "UPDATE incident_violations SET status = 'DISPATCHED' WHERE id = ?",
            (req.incident_id,)
        )

        return db.fetch_one("SELECT * FROM enforcement_dispatches WHERE id = ?", (dispatch_id,))

    @classmethod
    def resolve_incident(cls, req: ResolveIncidentRequest) -> Dict[str, Any]:
        incident = db.fetch_one("SELECT * FROM incident_violations WHERE id = ?", (req.incident_id,))
        if not incident:
            raise ValueError(f"Incident ID {req.incident_id} does not exist.")

        # Update dispatch resolution
        db.execute(
            """UPDATE enforcement_dispatches 
               SET resolution_notes = ?, resolved_at = CURRENT_TIMESTAMP
               WHERE incident_id = ?""",
            (req.resolution_notes, req.incident_id)
        )

        # Update incident status to RESOLVED
        db.execute(
            "UPDATE incident_violations SET status = 'RESOLVED' WHERE id = ?",
            (req.incident_id,)
        )

        return db.fetch_one("SELECT * FROM incident_violations WHERE id = ?", (req.incident_id,))
