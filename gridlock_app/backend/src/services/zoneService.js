import { db } from '../config/postgres.js';
import { decodeGeohash, classifyZoneTier } from '../utils/geospatial.js';

export const CongestionLevel = {
  LOW: 'LOW',
  MEDIUM: 'MEDIUM',
  HIGH: 'HIGH',
  SEVERE: 'SEVERE',
};

export class ZoneService {
  static async getAllZones() {
    return await db.fetchAll('SELECT * FROM zones ORDER BY junction_name ASC');
  }

  static async getZoneById(zoneId) {
    return await db.fetchOne('SELECT * FROM zones WHERE id = $1', [zoneId]);
  }

  static async getZoneByGeohash(geohash) {
    return await db.fetchOne('SELECT * FROM zones WHERE geohash = $1', [geohash]);
  }

  static async createZone({ geohash, junction_name, latitude, longitude, capacity }) {
    const existing = await this.getZoneByGeohash(geohash);
    if (existing) {
      throw new Error(`Zone with geohash '${geohash}' already exists (ID: ${existing.id}).`);
    }

    let lat = latitude;
    let lon = longitude;
    if (lat === undefined || lon === undefined) {
      const decoded = decodeGeohash(geohash);
      lat = decoded.lat;
      lon = decoded.lon;
    }

    const tier = classifyZoneTier(lat, lon);

    const inserted = await db.fetchOne(
      `INSERT INTO zones (geohash, junction_name, latitude, longitude, capacity, current_congestion_level, zone_tier)
       VALUES ($1, $2, $3, $4, $5, 'LOW', $6)
       RETURNING *`,
      [geohash, junction_name, lat, lon, capacity || 2000, tier]
    );
    return inserted;
  }

  static async updateCongestionLevel(zoneId, level) {
    await db.query(
      'UPDATE zones SET current_congestion_level = $1 WHERE id = $2',
      [level, zoneId]
    );
  }

  static async getZoneMetadata(zoneId) {
    const zone = await this.getZoneById(zoneId);
    if (!zone) {
      throw new Error(`Zone with ID ${zoneId} not found.`);
    }
    const tier = zone.zone_tier || classifyZoneTier(zone.latitude, zone.longitude);
    return { ...zone, zone_tier: tier };
  }
}
