"""Process 4.0: Incident & Enforcement Dispatch Models.

Data Transfer Objects for Violations and Enforcement Dispatches.
Corresponds to D4: incident_violations & enforcement_dispatches Tables.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ViolationType(str, Enum):
    SPEEDING = "SPEEDING"
    PARKING = "PARKING"
    SIGNAL = "SIGNAL"


class SeverityLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class IncidentStatus(str, Enum):
    REPORTED = "REPORTED"
    DISPATCHED = "DISPATCHED"
    RESOLVED = "RESOLVED"
    DISMISSED = "DISMISSED"


class IncidentReportRequest(BaseModel):
    zone_id: int
    violation_type: ViolationType
    severity: Optional[SeverityLevel] = SeverityLevel.MEDIUM
    description: Optional[str] = None


class DispatchUnitRequest(BaseModel):
    incident_id: int
    officer_badge_number: str = Field(..., min_length=3)
    notes: Optional[str] = None


class ResolveIncidentRequest(BaseModel):
    incident_id: int
    resolution_notes: str


class IncidentResponse(BaseModel):
    id: int
    zone_id: int
    junction_name: str
    violation_type: ViolationType
    severity: SeverityLevel
    status: IncidentStatus
    occurred_at: str


class DispatchResponse(BaseModel):
    id: int
    incident_id: int
    officer_badge_number: str
    dispatch_time: str
    resolution_notes: Optional[str]
    resolved_at: Optional[str]
