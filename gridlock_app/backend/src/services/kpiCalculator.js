import { db } from '../config/postgres.js';

export class KPICalculator {
  static async calculateSummaryKPIs() {
    // Total Zones (D2)
    const zones = await db.fetchAll('SELECT id, junction_name, current_congestion_level FROM zones');
    const totalZones = zones.length;

    // Average speed from latest sensor logs (D2)
    const speedRow = await db.fetchOne(`
      SELECT AVG(average_speed) as avg_spd, AVG(vehicle_count) as avg_cnt 
      FROM (
        SELECT average_speed, vehicle_count 
        FROM traffic_sensor_records 
        ORDER BY recorded_at DESC LIMIT 100
      ) sub
    `);
    const avgSpeed = speedRow && speedRow.avg_spd !== null ? parseFloat(Number(speedRow.avg_spd).toFixed(2)) : 38.5;

    // City-wide Congestion Index from latest prediction logs (D3)
    const predRow = await db.fetchOne(`
      SELECT AVG(congestion_index) as avg_cong, COUNT(id)::int as pred_cnt 
      FROM (
        SELECT congestion_index, id 
        FROM prediction_logs 
        ORDER BY created_at DESC LIMIT 50
      ) sub
    `);
    const avgCong = predRow && predRow.avg_cong !== null ? parseFloat(Number(predRow.avg_cong).toFixed(2)) : 42.0;
    const predCnt = predRow && predRow.pred_cnt ? parseInt(predRow.pred_cnt, 10) : 0;

    // Incidents & Dispatches (D4)
    const incRow = await db.fetchOne(`
      SELECT 
        SUM(CASE WHEN status IN ('REPORTED', 'DISPATCHED') THEN 1 ELSE 0 END)::int as active_inc,
        SUM(CASE WHEN status = 'DISPATCHED' THEN 1 ELSE 0 END)::int as dispatched_inc
      FROM incident_violations
    `);
    const activeInc = incRow && incRow.active_inc ? parseInt(incRow.active_inc, 10) : 0;
    const dispatchedInc = incRow && incRow.dispatched_inc ? parseInt(incRow.dispatched_inc, 10) : 0;

    // Highest Congestion Zone
    const highestZone = await db.fetchOne(`
      SELECT junction_name FROM zones 
      WHERE current_congestion_level IN ('SEVERE', 'HIGH')
      LIMIT 1
    `);
    const highestName = highestZone
      ? highestZone.junction_name
      : zones.length > 0
      ? zones[0].junction_name
      : 'None';

    return {
      total_active_zones: totalZones,
      average_city_speed_kmh: avgSpeed,
      city_wide_congestion_index: avgCong,
      active_incidents_count: activeInc,
      dispatched_units_count: dispatchedInc,
      recent_forecasts_count: predCnt,
      highest_congestion_zone: highestName,
    };
  }
}
