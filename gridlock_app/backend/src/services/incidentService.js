import { db } from '../config/postgres.js';
import { ZoneService } from './zoneService.js';

export const SeverityLevel = {
  LOW: 'LOW',
  MEDIUM: 'MEDIUM',
  HIGH: 'HIGH',
};

export const IncidentStatus = {
  REPORTED: 'REPORTED',
  DISPATCHED: 'DISPATCHED',
  RESOLVED: 'RESOLVED',
  DISMISSED: 'DISMISSED',
};

export class IncidentService {
  static async reportViolation(req, reportedByUserId = null) {
    const zone = await ZoneService.getZoneById(req.zone_id);
    if (!zone) {
      throw new Error(`Zone ID ${req.zone_id} not found.`);
    }

    const severity = req.severity || SeverityLevel.MEDIUM;
    const desc = req.description || '';

    const inserted = await db.fetchOne(
      `INSERT INTO incident_violations (zone_id, reported_by, violation_type, severity, status, description)
       VALUES ($1, $2, $3, $4, 'REPORTED', $5)
       RETURNING id`,
      [req.zone_id, reportedByUserId, req.violation_type, severity, desc]
    );

    return await db.fetchOne(
      `SELECT i.*, z.junction_name 
       FROM incident_violations i
       JOIN zones z ON i.zone_id = z.id
       WHERE i.id = $1`,
      [inserted.id]
    );
  }

  static async getIncidents(status = null, limit = 50) {
    if (status) {
      return await db.fetchAll(
        `SELECT i.*, z.junction_name, z.geohash, d.officer_badge_number, d.dispatch_time
         FROM incident_violations i
         JOIN zones z ON i.zone_id = z.id
         LEFT JOIN enforcement_dispatches d ON i.id = d.incident_id
         WHERE i.status = $1
         ORDER BY i.occurred_at DESC LIMIT $2`,
        [status, limit]
      );
    }

    return await db.fetchAll(
      `SELECT i.*, z.junction_name, z.geohash, d.officer_badge_number, d.dispatch_time
       FROM incident_violations i
       JOIN zones z ON i.zone_id = z.id
       LEFT JOIN enforcement_dispatches d ON i.id = d.incident_id
       ORDER BY i.occurred_at DESC LIMIT $1`,
      [limit]
    );
  }
}
