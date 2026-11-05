"""Hotspot and High-Impact Violation Analyzer.

Integrates the empirical analysis from Phase 2 (Bangalore Police Violation dataset):
- High-impact parking identification (main roads, footpaths, double parking)
- Spatial violation density clustering
- Chokepoint hotspot ranking for proactive enforcement
"""

from __future__ import annotations

from typing import Any, Dict, List
from backend.database.connection import db


class HotspotAnalyzer:
    """Calculates active chokepoints and ranks critical enforcement zones."""

    HIGH_IMPACT_KEYWORDS = [
        "MAIN ROAD", "FOOTPATH", "DOUBLE PARKING", "ROAD CROSSING", "SIGNAL JUMP"
    ]

    @classmethod
    def get_active_hotspots(cls) -> List[Dict[str, Any]]:
        """Identifies top zones with active unresolved violations (D4)."""
        query = """
            SELECT z.id as zone_id, z.junction_name, z.geohash, z.latitude, z.longitude,
                   COUNT(i.id) as active_violation_count,
                   SUM(CASE WHEN i.severity = 'HIGH' THEN 3 WHEN i.severity = 'MEDIUM' THEN 2 ELSE 1 END) as risk_score
            FROM zones z
            JOIN incident_violations i ON z.id = i.zone_id
            WHERE i.status IN ('REPORTED', 'DISPATCHED')
            GROUP BY z.id
            ORDER BY risk_score DESC, active_violation_count DESC
            LIMIT 10
        """
        return db.fetch_all(query)

    @classmethod
    def classify_violation_impact(cls, raw_violation_text: str) -> str:
        """Determines if a violation creates high traffic congestion impact."""
        text = raw_violation_text.upper()
        if any(kw in text for kw in cls.HIGH_IMPACT_KEYWORDS):
            return "HIGH"
        return "MEDIUM"
