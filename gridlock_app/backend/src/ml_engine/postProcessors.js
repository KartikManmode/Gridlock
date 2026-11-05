/**
 * Post-Processing Transformations for SOTA ML Engine.
 * 1. Pluviometer Engine (Rain Paradox):
 *    - Core City + Rain: -1.4% dampening (Safety distance thinning on arteries)
 *    - Non-Core + Rain: +5.0% boost (Commuter mode shift from transit/walking to cabs)
 * 2. Thermo-Spike Power Transformation (y^0.93):
 *    - Counteracts regression smoothing on peak congestion spikes (> 0.40)
 * 3. Extreme Heat Artery Boost:
 *    - Boosts high-artery demand when temperatures exceed 40.0°C
 */

export function applyPluviometerAdjustment(predictedVolume, zoneTier, weatherCondition) {
  const weather = (weatherCondition || '').toUpperCase();
  let volume = predictedVolume;
  if (weather.includes('RAIN')) {
    if (zoneTier === 'Core City') {
      volume *= 0.986; // -1.4%
    } else {
      volume *= 1.05;  // +5.0%
    }
  }
  return volume;
}

export function applyThermoSpikeScaling(predictedVolume, power = 0.93, threshold = 0.40) {
  if (predictedVolume > threshold) {
    return Math.pow(predictedVolume, power);
  }
  return predictedVolume;
}

export function applyExtremeHeatBoost(predictedVolume, temperatureC, isArtery) {
  if (temperatureC > 40.0 && isArtery) {
    return Math.pow(predictedVolume, 0.70);
  }
  return predictedVolume;
}
