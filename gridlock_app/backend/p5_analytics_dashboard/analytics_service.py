"""Process 5.0: Analytics & Dashboard Service.

Joins D2, D3, and D4 data stores to render dynamic heatmaps,
executive KPIs, and active hotspot reports to UserEntity.
"""

from __future__ import annotations

from typing import Any, Dict
from backend.p5_analytics_dashboard.models import DashboardAggregationResponse
from backend.p5_analytics_dashboard.kpi_calculator import KPICalculator
from backend.p5_analytics_dashboard.heatmap_generator import HeatmapGenerator
from backend.p4_incidents_dispatch.hotspot_analyzer import HotspotAnalyzer


class AnalyticsService:
    """Dashboard aggregation coordinator."""

    @classmethod
    def get_dashboard_summary(cls) -> DashboardAggregationResponse:
        kpis = KPICalculator.calculate_summary_kpis()
        heatmap = HeatmapGenerator.generate_heatmap_points()
        hotspots = HotspotAnalyzer.get_active_hotspots()

        return DashboardAggregationResponse(
            kpis=kpis,
            heatmap=heatmap,
            active_hotspots=hotspots
        )
