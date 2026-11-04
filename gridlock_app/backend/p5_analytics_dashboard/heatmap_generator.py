"""Dynamic Geospatial Heatmap Generator.

Fuses:
- D2: Zone coordinates and sensor traffic volumes
- D3: Predictive congestion indices
- D4: Active violation counts
Produces normalized geospatial points for interactive map overlays.
"""

from __future__ import annotations

from typing import List
from backend.database.connection import db
from backend.p5_analytics_dashboard.models import HeatmapPoint


class HeatmapGenerator:
    """Generates dynamic heatmap payload."""

    @classmethod
    def generate_heatmap_points(cls) -> List[HeatmapPoint]:
        query = """
            SELECT z.id as zone_id, z.junction_name, z.geohash, z.latitude, z.longitude,
                   z.current_congestion_level,
                   COALESCE((
                       SELECT predicted_volume FROM prediction_logs 
                       WHERE zone_id = z.id ORDER BY created_at DESC LIMIT 1
                   ), 0.35) as latest_forecast_volume,
                   COALESCE((
                       SELECT COUNT(id) FROM incident_violations 
                       WHERE zone_id = z.id AND status IN ('REPORTED', 'DISPATCHED')
                   ), 0) as active_violations
            FROM zones z
        """
        rows = db.fetch_all(query)
        points = []

        level_weights = {
            "LOW": 0.25,
            "MEDIUM": 0.50,
            "HIGH": 0.75,
            "SEVERE": 1.00
        }

        for r in rows:
            base_w = level_weights.get(r["current_congestion_level"], 0.3)
            pred_w = float(r["latest_forecast_volume"])
            viol_bump = min(0.2, int(r["active_violations"]) * 0.05)
            
            # Blended heatmap intensity
            final_weight = min(1.0, max(0.05, base_w * 0.5 + pred_w * 0.4 + viol_bump))

            points.append(HeatmapPoint(
                zone_id=r["zone_id"],
                junction_name=r["junction_name"],
                geohash=r["geohash"],
                latitude=r["latitude"],
                longitude=r["longitude"],
                weight=round(final_weight, 3),
                congestion_level=r["current_congestion_level"],
                active_violations=int(r["active_violations"])
            ))

        return points
