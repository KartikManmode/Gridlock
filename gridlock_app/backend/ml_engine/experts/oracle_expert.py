"""Expert V4: Oracle Expert.

The seasonal historical memory anchor from Day 48 baseline.
Provides anti-drift leash preventing recursive runaway predictions.
"""

from __future__ import annotations

from typing import Dict, Any


class OracleExpert:
    """Historical seasonal anchor."""

    @staticmethod
    def predict(features: Dict[str, Any]) -> float:
        # Yesterday's demand at the exact same hour/slot
        yesterday_same_time = features.get("lag_24h_demand")
        if yesterday_same_time is not None and yesterday_same_time >= 0:
            return float(yesterday_same_time)
        
        # Fallback to recent rolling average
        return float(features.get("rolling_mean_demand", 0.35))
