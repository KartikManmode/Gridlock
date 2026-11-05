import { db } from '../config/postgres.js';

export class HeatmapGenerator {
  static async generateHeatmapPoints() {
    const query = `
      SELECT z.id as zone_id, z.junction_name, z.geohash, z.latitude, z.longitude,
             z.current_congestion_level,
             COALESCE((
                 SELECT predicted_volume FROM prediction_logs 
                 WHERE zone_id = z.id ORDER BY created_at DESC LIMIT 1
             ), 0.35) as latest_forecast_volume,
             COALESCE((
                 SELECT COUNT(id)::int FROM incident_violations 
                 WHERE zone_id = z.id AND status IN ('REPORTED', 'DISPATCHED')
             ), 0) as active_violations
      FROM zones z
    `;
    const rows = await db.fetchAll(query);

    const levelWeights = {
      LOW: 0.25,
      MEDIUM: 0.50,
      HIGH: 0.75,
      SEVERE: 1.00,
    };

    return rows.map((r) => {
      const baseW = levelWeights[r.current_congestion_level] || 0.30;
      const predW = parseFloat(r.latest_forecast_volume) || 0.35;
      const violBump = Math.min(0.2, (parseInt(r.active_violations, 10) || 0) * 0.05);

      const finalWeight = Math.min(1.0, Math.max(0.05, baseW * 0.5 + predW * 0.4 + violBump));

      return {
        zone_id: r.zone_id,
        junction_name: r.junction_name,
        geohash: r.geohash,
        latitude: parseFloat(r.latitude),
        longitude: parseFloat(r.longitude),
        weight: parseFloat(finalWeight.toFixed(3)),
        congestion_level: r.current_congestion_level,
        latest_forecast_volume: parseFloat(predW.toFixed(3)),
        active_violations: parseInt(r.active_violations, 10) || 0,
      };
    });
  }
}
