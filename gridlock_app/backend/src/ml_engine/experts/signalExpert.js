/**
 * Expert V1: Signal Expert.
 * Captures central traffic throughput and high-frequency cyclical patterns
 * using sinusoidal time projections and recent lag momentum.
 */
export class SignalExpert {
  static predict(features) {
    const hour = features.hour !== undefined ? features.hour : 12;
    const minute = features.minute !== undefined ? features.minute : 0;
    const recentLag = features.lag_15m_demand !== undefined ? features.lag_15m_demand : 0.3;

    // Diurnal peak cycles: Morning rush (8-10 AM) and Evening rush (5-8 PM)
    const tHours = hour + minute / 60.0;
    const morningPeak = Math.exp(-Math.pow(tHours - 9.0, 2) / (2 * Math.pow(1.8, 2)));
    const eveningPeak = Math.exp(-Math.pow(tHours - 18.0, 2) / (2 * Math.pow(2.2, 2)));
    const baseCycle = 0.05 + 0.55 * Math.max(morningPeak, eveningPeak * 1.1);

    // Momentum blending
    return 0.4 * baseCycle + 0.6 * recentLag;
  }
}
