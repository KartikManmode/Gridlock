"""API Routes for Process 4.0: Incident & Enforcement Dispatch."""

from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Header, status
from backend.p4_incidents_dispatch.models import (
    IncidentReportRequest, DispatchUnitRequest, ResolveIncidentRequest
)
from backend.p4_incidents_dispatch.incident_service import IncidentService
from backend.p4_incidents_dispatch.dispatch_service import DispatchService
from backend.p1_auth_rbac.auth_service import AuthService

router = APIRouter(tags=["4.0 Incident & Enforcement Dispatch"])


@router.post("/incidents", status_code=status.HTTP_201_CREATED)
def report_incident(req: IncidentReportRequest, authorization: Optional[str] = Header(None)):
    """UserEntity (Citizen/Operator) reports a traffic violation."""
    user_id = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        payload = AuthService.verify_token(token)
        if payload:
            user_id = payload.get("sub")
    try:
        incident = IncidentService.report_violation(req, reported_by_user_id=user_id)
        return {
            "status": "REPORTED",
            "message": "Incident violation registered successfully",
            "incident": incident
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/incidents")
def list_incidents(status_filter: Optional[str] = None, limit: int = 50):
    return IncidentService.get_incidents(status=status_filter, limit=limit)


@router.post("/dispatch", status_code=status.HTTP_201_CREATED)
def dispatch_unit(req: DispatchUnitRequest, authorization: Optional[str] = Header(None)):
    """Operator dispatches an enforcement officer to a violation hotspot."""
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        AuthService.check_permission(token, "dispatch:manage")
    try:
        dispatch = DispatchService.dispatch_unit(req)
        return {
            "status": "DISPATCHED",
            "message": f"Officer {req.officer_badge_number} dispatched to incident {req.incident_id}",
            "dispatch": dispatch
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/dispatch/resolve")
def resolve_incident(req: ResolveIncidentRequest, authorization: Optional[str] = Header(None)):
    """Marks incident resolved and records resolution notes."""
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        AuthService.check_permission(token, "dispatch:manage")
    try:
        updated = DispatchService.resolve_incident(req)
        return {"status": "RESOLVED", "incident": updated}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
