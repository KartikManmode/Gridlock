"""API Routes for Process 5.0: Analytics & Dashboard Aggregation."""

from __future__ import annotations

from fastapi import APIRouter
from backend.p5_analytics_dashboard.models import DashboardAggregationResponse
from backend.p5_analytics_dashboard.analytics_service import AnalyticsService

router = APIRouter(prefix="/analytics", tags=["5.0 Analytics & Dashboard Aggregation"])


@router.get("/dashboard", response_model=DashboardAggregationResponse)
def get_dashboard_data():
    """Renders dynamic heatmap, summary KPIs, and active hotspots for UserEntity."""
    return AnalyticsService.get_dashboard_summary()
