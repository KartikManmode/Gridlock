"""Expert V1: Signal Expert.

Captures central traffic throughput and high-frequency cyclical patterns
using sinusoidal time projections and recent lag momentum.
"""

from __future__ import annotations

import math
from typing import Dict, Any


class SignalExpert:
    """Predicts base periodic throughput."""

    @staticmethod
    def predict(features: Dict[str, Any]) -> float:
        hour = features.get("hour", 12)
        minute = features.get("minute", 0)
        recent_lag = features.get("lag_15m_demand", 0.3)

        # Diurnal peak cycles: Morning rush (8-10 AM) and Evening rush (5-8 PM)
        t_hours = hour + minute / 60.0
        morning_peak = math.exp(-((t_hours - 9.0) ** 2) / (2 * (1.8 ** 2)))
        evening_peak = math.exp(-((t_hours - 18.0) ** 2) / (2 * (2.2 ** 2)))
        base_cycle = 0.05 + 0.55 * max(morning_peak, evening_peak * 1.1)

        # Momentum blending
        return 0.4 * base_cycle + 0.6 * recent_lag
