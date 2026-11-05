import { db } from '../config/postgres.js';

export class DispatchService {
  static async dispatchUnit(req) {
    const incident = await db.fetchOne(
      'SELECT * FROM incident_violations WHERE id = $1',
      [req.incident_id]
    );
    if (!incident) {
      throw new Error(`Incident ID ${req.incident_id} does not exist.`);
    }

    const existing = await db.fetchOne(
      'SELECT * FROM enforcement_dispatches WHERE incident_id = $1',
      [req.incident_id]
    );
    if (existing) {
      throw new Error(`Enforcement unit already dispatched to Incident ID ${req.incident_id}.`);
    }

    const inserted = await db.fetchOne(
      `INSERT INTO enforcement_dispatches (incident_id, officer_badge_number, resolution_notes)
       VALUES ($1, $2, $3)
       RETURNING *`,
      [req.incident_id, req.officer_badge_number, req.notes || '']
    );

    await db.query(
      "UPDATE incident_violations SET status = 'DISPATCHED' WHERE id = $1",
      [req.incident_id]
    );

    return inserted;
  }

  static async resolveIncident(req) {
    const incident = await db.fetchOne(
      'SELECT * FROM incident_violations WHERE id = $1',
      [req.incident_id]
    );
    if (!incident) {
      throw new Error(`Incident ID ${req.incident_id} does not exist.`);
    }

    await db.query(
      `UPDATE enforcement_dispatches 
       SET resolution_notes = $1, resolved_at = NOW()
       WHERE incident_id = $2`,
      [req.resolution_notes || 'Incident cleared and verified.', req.incident_id]
    );

    await db.query(
      "UPDATE incident_violations SET status = 'RESOLVED' WHERE id = $1",
      [req.incident_id]
    );

    return {
      status: 'RESOLVED',
      incident_id: req.incident_id,
      notes: req.resolution_notes,
    };
  }
}
