"""Post-Processing Transformations for SOTA ML Engine.

Implements the proven SOTA winning innovations from Project Gridlock:
1. Pluviometer Engine (Rain Paradox):
   - Core City + Rain: -1.4% dampening (Safety distance thinning on arteries)
   - Non-Core + Rain: +5.0% boost (Commuter mode shift from transit/walking to cabs)
2. Thermo-Spike Power Transformation (y^0.93):
   - Counteracts regression smoothing on peak congestion spikes (> 0.40)
3. Extreme Heat Artery Boost:
   - Boosts high-artery demand when temperatures exceed 40.0°C
"""

from __future__ import annotations


def apply_pluviometer_adjustment(
    predicted_volume: float,
    zone_tier: str,
    weather_condition: str
) -> float:
    """Applies the Rain Paradox conditional adjustment."""
    weather = weather_condition.upper()
    if "RAIN" in weather:
        if zone_tier == "Core City":
            # Rule 1: Core City Safety Thinning (-1.4%)
            predicted_volume *= 0.986
        else:
            # Rule 2: Non-Core Mode-Shift Surge (+5.0%)
            predicted_volume *= 1.05
    return predicted_volume


def apply_thermo_spike_scaling(
    predicted_volume: float,
    power: float = 0.93,
    threshold: float = 0.40
) -> float:
    """Stretches high-demand peaks using non-linear power transform."""
    if predicted_volume > threshold:
        predicted_volume = predicted_volume ** power
    return predicted_volume


def apply_extreme_heat_boost(
    predicted_volume: float,
    temperature_c: float,
    is_artery: bool
) -> float:
    """Applies heat-artery expansion when temperature exceeds 40°C."""
    if temperature_c > 40.0 and is_artery:
        predicted_volume = predicted_volume ** 0.70
    return predicted_volume
