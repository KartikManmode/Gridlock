import { SignalExpert } from './experts/signalExpert.js';
import { SpatialExpert } from './experts/spatialExpert.js';
import { ShockExpert } from './experts/shockExpert.js';
import { OracleExpert } from './experts/oracleExpert.js';
import {
  applyPluviometerAdjustment,
  applyThermoSpikeScaling,
  applyExtremeHeatBoost,
} from './postProcessors.js';

export class MLEngine {
  // Golden Ratio weights established in SOTA v19 & v47
  static W_SIGNAL = 0.35;
  static W_SPATIAL = 0.25;
  static W_SHOCK = 0.20;
  static W_ORACLE = 0.20;

  static predict(featureVector) {
    // 1. Run the 4 Domain Experts
    const pSignal = SignalExpert.predict(featureVector);
    const pSpatial = SpatialExpert.predict(featureVector);
    const pShock = ShockExpert.predict(featureVector);
    const pOracle = OracleExpert.predict(featureVector);

    // 2. Golden Ratio Meta-Blend
    let baseDemand =
      this.W_SIGNAL * pSignal +
      this.W_SPATIAL * pSpatial +
      this.W_SHOCK * pShock +
      this.W_ORACLE * pOracle;

    // 3. Global Peak Stabilization / Anchoring
    const baseline = featureVector.lag_24h_demand !== undefined ? featureVector.lag_24h_demand : baseDemand;
    if (baseline > 0 && baseDemand / baseline > 1.15) {
      baseDemand *= 0.95;
    }

    // 4. Pluviometer Post-Processing
    const zoneTier = featureVector.zone_tier || 'Core City';
    const weather = featureVector.weather_condition || 'CLEAR';
    let adjustedDemand = applyPluviometerAdjustment(baseDemand, zoneTier, weather);

    // 5. Thermo-Spike Transformation
    adjustedDemand = applyThermoSpikeScaling(adjustedDemand, 0.93, 0.40);

    // 6. Extreme Heat Expansion
    const temp = featureVector.temperature !== undefined ? featureVector.temperature : 28.0;
    const isArtery = Boolean(featureVector.is_artery);
    adjustedDemand = applyExtremeHeatBoost(adjustedDemand, temp, isArtery);

    // 7. Clamp within valid bounds [0.05, 1.0]
    const finalVolume = Math.max(0.05, Math.min(1.0, adjustedDemand));

    // Congestion Index (0-100 scale)
    const congestionIndex = parseFloat((finalVolume * 100.0).toFixed(2));

    // Real-time confidence calibration
    const rollingMean = featureVector.rolling_mean_demand !== undefined ? featureVector.rolling_mean_demand : 0.35;
    const varianceDelta = Math.abs(finalVolume - rollingMean);
    const confidence = parseFloat(Math.max(0.50, Math.min(0.95, 0.70 + 0.25 * (1.0 - varianceDelta))).toFixed(4));

    return {
      predictedVolume: parseFloat(finalVolume.toFixed(4)),
      congestionIndex,
      confidence,
      expertBreakdown: {
        signal_expert: parseFloat(pSignal.toFixed(4)),
        spatial_expert: parseFloat(pSpatial.toFixed(4)),
        shock_expert: parseFloat(pShock.toFixed(4)),
        oracle_expert: parseFloat(pOracle.toFixed(4)),
      },
    };
  }
}
