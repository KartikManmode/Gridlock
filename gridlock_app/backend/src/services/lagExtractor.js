import { db } from '../config/postgres.js';
import { ZoneService } from './zoneService.js';

export class LagExtractor {
  static async buildFeatureVector(zoneId, targetDt, weather = 'CLEAR', temperature = 28.0) {
    const zone = await ZoneService.getZoneMetadata(zoneId);
    const capacity = Math.max(1, zone.capacity || 2000);

    // Query recent sensor records (D2)
    const records = await db.fetchAll(
      `SELECT vehicle_count, average_speed, weather_condition, recorded_at
       FROM traffic_sensor_records
       WHERE zone_id = $1
       ORDER BY recorded_at DESC LIMIT 5`,
      [zoneId]
    );

    let lag15m = 0.35;
    let lag30m = 0.35;
    let rollingMean = 0.35;

    if (records.length > 0) {
      const demands = records.map((r) => r.vehicle_count / capacity);
      lag15m = demands[0];
      lag30m = demands.length > 1 ? demands[1] : demands[0];
      rollingMean = demands.reduce((acc, v) => acc + v, 0) / demands.length;
    }

    // Query active incidents (D4) for shock impact
    const activeIncidents = await db.fetchAll(
      `SELECT id FROM incident_violations
       WHERE zone_id = $1 AND status IN ('REPORTED', 'DISPATCHED')`,
      [zoneId]
    );

    const dt = targetDt instanceof Date ? targetDt : new Date(targetDt);

    return {
      zone_id: zoneId,
      geohash: zone.geohash,
      latitude: zone.latitude,
      longitude: zone.longitude,
      capacity,
      zone_tier: zone.zone_tier || 'Core City',
      hour: dt.getUTCHours(),
      minute: dt.getUTCMinutes(),
      lag_15m_demand: lag15m,
      lag_30m_demand: lag30m,
      lag_24h_demand: rollingMean,
      rolling_mean_demand: rollingMean,
      weather_condition: weather,
      temperature,
      active_incidents: activeIncidents.length,
      is_artery: capacity >= 2000,
    };
  }
}
