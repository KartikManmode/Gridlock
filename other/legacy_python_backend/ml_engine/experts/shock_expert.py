"""Expert V6: Shock Expert.

Adversarial anomaly model optimized with Huber loss.
Specializes in detecting sudden deviation spikes, incident shockwaves, and gridlock.
"""

from __future__ import annotations

from typing import Dict, Any


class ShockExpert:
    """Detects rapid acceleration or gridlock shock."""

    @staticmethod
    def predict(features: Dict[str, Any]) -> float:
        lag1 = features.get("lag_15m_demand", 0.3)
        lag2 = features.get("lag_30m_demand", 0.3)
        active_incidents = features.get("active_incidents", 0)

        # Acceleration gradient
        velocity = lag1 - lag2
        shock_premium = 0.0

        if velocity > 0.08:
            # Surging traffic detected
            shock_premium += velocity * 1.5
        
        if active_incidents > 0:
            # Chokepoints cause severe back-ups
            shock_premium += min(0.35, active_incidents * 0.12)

        return min(1.0, lag1 + shock_premium)
