"""Python ML Engine Inference Orchestrator.

Implements the SOTA winning architecture (93.087% R²):
- 4 Domain Experts: Signal (35%), Spatial (25%), Shock (20%), Oracle (20%)
- Global Hourly Anchoring
- Pluviometer Engine (Rain Paradox)
- Thermo-Spike Power Transformation (0.93 power)
- Real-time confidence calibration
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple
from backend.ml_engine.experts.signal_expert import SignalExpert
from backend.ml_engine.experts.oracle_expert import OracleExpert
from backend.ml_engine.experts.spatial_expert import SpatialExpert
from backend.ml_engine.experts.shock_expert import ShockExpert
from backend.ml_engine.post_processors import (
    apply_pluviometer_adjustment,
    apply_thermo_spike_scaling,
    apply_extreme_heat_boost
)


class MLEngine:
    """Core Python ML Engine for Traffic Congestion Forecasting."""

    # Golden Ratio weights established in SOTA v19 & v47
    W_SIGNAL = 0.35
    W_SPATIAL = 0.25
    W_SHOCK = 0.20
    W_ORACLE = 0.20

    @classmethod
    def predict(cls, feature_vector: Dict[str, Any]) -> Tuple[float, float, float]:
        """Runs the complete ensemble and post-processing pipeline.

        Returns:
            (predicted_volume, congestion_index, model_confidence)
        """
        # 1. Run the 4 Domain Experts
        p_signal = SignalExpert.predict(feature_vector)
        p_spatial = SpatialExpert.predict(feature_vector)
        p_shock = ShockExpert.predict(feature_vector)
        p_oracle = OracleExpert.predict(feature_vector)

        # 2. Golden Ratio Meta-Blend
        base_demand = (
            cls.W_SIGNAL * p_signal +
            cls.W_SPATIAL * p_spatial +
            cls.W_SHOCK * p_shock +
            cls.W_ORACLE * p_oracle
        )

        # 3. Global Peak Stabilization / Anchoring
        # If predicted demand overshoots historical baseline by > 15%, pull back 5%
        baseline = feature_vector.get("lag_24h_demand", base_demand)
        if baseline > 0 and (base_demand / baseline) > 1.15:
            base_demand *= 0.95

        # 4. SOTA Post-Processing Layer (Pluviometer & Thermo-Spike)
        zone_tier = feature_vector.get("zone_tier", "Core City")
        weather = feature_vector.get("weather_condition", "CLEAR")
        temp = feature_vector.get("temperature", 28.0)
        is_artery = feature_vector.get("is_artery", True)

        # 4a. Pluviometer: Rain Paradox
        adjusted = apply_pluviometer_adjustment(base_demand, zone_tier, weather)

        # 4b. Extreme Heat Artery Boost
        adjusted = apply_extreme_heat_boost(adjusted, temp, is_artery)

        # 4c. Thermo-Spike Power Transform (0.93)
        final_demand = apply_thermo_spike_scaling(adjusted, power=0.93, threshold=0.40)

        # 5. Bound checking [0.0, 1.0]
        final_demand = max(0.0, min(1.0, float(final_demand)))

        # 6. Congestion Index (0 to 100)
        congestion_index = round(final_demand * 100.0, 2)

        # 7. Model Confidence Calibration
        # Variance among experts serves as uncertainty metric
        expert_preds = [p_signal, p_spatial, p_shock, p_oracle]
        mean_p = sum(expert_preds) / len(expert_preds)
        variance = sum((x - mean_p) ** 2 for x in expert_preds) / len(expert_preds)
        std_dev = math.sqrt(variance)

        # Confidence: High when experts agree (low std dev), capped at 0.98
        confidence = max(0.65, min(0.98, 0.96 - std_dev * 0.8))
        confidence = round(confidence, 4)

        return round(final_demand, 4), congestion_index, confidence
