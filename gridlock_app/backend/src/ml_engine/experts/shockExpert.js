/**
 * Expert V6: Shock Expert.
 * Adversarial anomaly model.
 * Specializes in detecting sudden deviation spikes, incident shockwaves, and gridlock.
 */
export class ShockExpert {
  static predict(features) {
    const lag1 = features.lag_15m_demand !== undefined ? features.lag_15m_demand : 0.3;
    const lag2 = features.lag_30m_demand !== undefined ? features.lag_30m_demand : 0.3;
    const activeIncidents = features.active_incidents || 0;

    // Acceleration gradient
    const velocity = lag1 - lag2;
    let shockPremium = 0.0;

    if (velocity > 0.08) {
      // Surging traffic detected
      shockPremium += velocity * 1.5;
    }

    if (activeIncidents > 0) {
      // Chokepoints cause severe back-ups
      shockPremium += Math.min(0.35, activeIncidents * 0.12);
    }

    return Math.min(1.0, lag1 + shockPremium);
  }
}
