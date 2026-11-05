import { db } from '../config/postgres.js';

export class HotspotAnalyzer {
  static async getActiveHotspots() {
    const query = `
      SELECT z.id as zone_id, z.junction_name, z.geohash, z.latitude, z.longitude,
             COUNT(i.id)::int as active_violation_count,
             SUM(CASE WHEN i.severity = 'HIGH' THEN 3 WHEN i.severity = 'MEDIUM' THEN 2 ELSE 1 END)::int as risk_score
      FROM zones z
      JOIN incident_violations i ON z.id = i.zone_id
      WHERE i.status IN ('REPORTED', 'DISPATCHED')
      GROUP BY z.id
      ORDER BY risk_score DESC, active_violation_count DESC
      LIMIT 10
    `;
    return await db.fetchAll(query);
  }
}
