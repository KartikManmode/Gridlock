/**
 * Expert V4: Oracle Expert.
 * The seasonal historical memory anchor.
 * Provides anti-drift leash preventing recursive runaway predictions.
 */
export class OracleExpert {
  static predict(features) {
    const yesterdaySameTime = features.lag_24h_demand;
    if (yesterdaySameTime !== undefined && yesterdaySameTime !== null && yesterdaySameTime >= 0) {
      return Number(yesterdaySameTime);
    }
    return Number(features.rolling_mean_demand !== undefined ? features.rolling_mean_demand : 0.35);
  }
}
