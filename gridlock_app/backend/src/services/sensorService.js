import { db } from '../config/postgres.js';
import { ZoneService, CongestionLevel } from './zoneService.js';

export class SensorService {
  static async ingestReading(record) {
    let zoneId = record.zone_id;
    let zone = null;

    if (!zoneId) {
      if (!record.geohash) {
        throw new Error('Must provide either zone_id or geohash.');
      }
      zone = await ZoneService.getZoneByGeohash(record.geohash);
      if (!zone) {
        throw new Error(`No registered zone found for geohash '${record.geohash}'.`);
      }
      zoneId = zone.id;
    } else {
      zone = await ZoneService.getZoneById(zoneId);
      if (!zone) {
        throw new Error(`No registered zone found for ID ${zoneId}.`);
      }
    }

    const weather = (record.weather_condition || 'CLEAR').toUpperCase();

    // Insert sensor record into D2
    const inserted = await db.fetchOne(
      `INSERT INTO traffic_sensor_records (zone_id, vehicle_count, average_speed, weather_condition)
       VALUES ($1, $2, $3, $4)
       RETURNING *`,
      [zoneId, record.vehicle_count, record.average_speed, weather]
    );

    // Dynamic congestion level calculation
    const capacity = Math.max(1, zone.capacity || 2000);
    const occupancyRatio = record.vehicle_count / capacity;

    let level = CongestionLevel.LOW;
    if (occupancyRatio >= 0.90) {
      level = CongestionLevel.SEVERE;
    } else if (occupancyRatio >= 0.65) {
      level = CongestionLevel.HIGH;
    } else if (occupancyRatio >= 0.35) {
      level = CongestionLevel.MEDIUM;
    }

    await ZoneService.updateCongestionLevel(zoneId, level);

    return inserted;
  }

  static async getRecentReadings(zoneId, limit = 20) {
    return await db.fetchAll(
      `SELECT * FROM traffic_sensor_records 
       WHERE zone_id = $1 
       ORDER BY recorded_at DESC LIMIT $2`,
      [zoneId, limit]
    );
  }
}
