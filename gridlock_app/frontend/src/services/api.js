const API_BASE = import.meta.env.VITE_API_BASE_URL 
  ? `${import.meta.env.VITE_API_BASE_URL.replace(/\/$/, '')}/api` 
  : '/api';

// Realistic fallback zones matching dataset geohash coordinates
export const FALLBACK_ZONES = [
  { id: 1, geohash: 'qp02zt', junction_name: 'Silk Board Chokepoint', latitude: -5.46295, longitude: 90.68665, capacity: 3500, current_congestion_level: 'SEVERE', zone_tier: 'Core City' },
  { id: 2, geohash: 'qp02z1', junction_name: 'MG Road Junction', latitude: -5.48492, longitude: 90.66467, capacity: 2200, current_congestion_level: 'HIGH', zone_tier: 'Core City' },
  { id: 3, geohash: 'qp08bj', junction_name: 'Indiranagar 100ft Rd', latitude: -5.46295, longitude: 90.70862, capacity: 1800, current_congestion_level: 'MEDIUM', zone_tier: 'Mid-Ring' },
  { id: 4, geohash: 'qp08gt', junction_name: 'Hebbal Flyover', latitude: -5.46295, longitude: 90.86243, capacity: 3200, current_congestion_level: 'HIGH', zone_tier: 'Mid-Ring' },
  { id: 5, geohash: 'qp02zm', junction_name: 'Koramangala Sony World', latitude: -5.46295, longitude: 90.67566, capacity: 2000, current_congestion_level: 'SEVERE', zone_tier: 'Core City' },
  { id: 6, geohash: 'qp08fz', junction_name: 'Whitefield ITPL Hub', latitude: -5.45197, longitude: 90.82947, capacity: 2800, current_congestion_level: 'HIGH', zone_tier: 'Periphery' },
  { id: 7, geohash: 'qp02zp', junction_name: 'BTM Layout 2nd Stage', latitude: -5.45197, longitude: 90.66467, capacity: 1600, current_congestion_level: 'MEDIUM', zone_tier: 'Core City' },
  { id: 8, geohash: 'qp08by', junction_name: 'Marathahalli Bridge', latitude: -5.45746, longitude: 90.74158, capacity: 2500, current_congestion_level: 'SEVERE', zone_tier: 'Mid-Ring' },
  { id: 9, geohash: 'qp02ys', junction_name: 'Jayanagar 4th Block', latitude: -5.46844, longitude: 90.64270, capacity: 1500, current_congestion_level: 'LOW', zone_tier: 'Core City' },
  { id: 10, geohash: 'qp08gh', junction_name: 'Yeshwantpur Circle', latitude: -5.46844, longitude: 90.84045, capacity: 2400, current_congestion_level: 'MEDIUM', zone_tier: 'Mid-Ring' },
];

export const FALLBACK_KPIS = {
  total_active_zones: 50,
  average_city_speed_kmh: 21.4,
  city_wide_congestion_index: 68.2,
  active_incidents_count: 8,
  dispatched_units_count: 5,
  recent_forecasts_count: 96,
  highest_congestion_zone: 'Silk Board Chokepoint',
};

export const FALLBACK_INCIDENTS = [
  { id: 101, zone_id: 1, junction_name: 'Silk Board Chokepoint', geohash: 'qp02zt', violation_type: 'PARKING', severity: 'HIGH', status: 'DISPATCHED', occurred_at: '10 mins ago', officer_badge_number: 'KA-TP-1042', description: 'Double parking blocking outer bus lane' },
  { id: 102, zone_id: 8, junction_name: 'Marathahalli Bridge', geohash: 'qp08by', violation_type: 'SIGNAL', severity: 'HIGH', status: 'REPORTED', occurred_at: '15 mins ago', description: 'Traffic signal jump causing cross-intersection gridlock' },
  { id: 103, zone_id: 2, junction_name: 'MG Road Junction', geohash: 'qp02z1', violation_type: 'PARKING', severity: 'MEDIUM', status: 'DISPATCHED', occurred_at: '25 mins ago', officer_badge_number: 'KA-TP-2089', description: 'Commercial loading truck parked on footpath' },
  { id: 104, zone_id: 4, junction_name: 'Hebbal Flyover', geohash: 'qp08gt', violation_type: 'SPEEDING', severity: 'LOW', status: 'REPORTED', occurred_at: '40 mins ago', description: 'Vehicle breakdown on ramp' },
];

export async function fetchDashboardData() {
  try {
    const res = await fetch(`${API_BASE}/analytics/dashboard`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      return await res.json();
    }
  } catch (err) {
    // Backend offline / connecting
  }

  // Generate fallback heatmap points
  const heatmap = FALLBACK_ZONES.map(z => ({
    zone_id: z.id,
    junction_name: z.junction_name,
    geohash: z.geohash,
    latitude: z.latitude,
    longitude: z.longitude,
    weight: z.current_congestion_level === 'SEVERE' ? 0.95 : z.current_congestion_level === 'HIGH' ? 0.75 : z.current_congestion_level === 'MEDIUM' ? 0.50 : 0.25,
    congestion_level: z.current_congestion_level,
    active_violations: z.current_congestion_level === 'SEVERE' ? 3 : z.current_congestion_level === 'HIGH' ? 1 : 0
  }));

  return {
    kpis: FALLBACK_KPIS,
    heatmap,
    active_hotspots: [
      { zone_id: 1, junction_name: 'Silk Board Chokepoint', risk_score: 9, active_violation_count: 3 },
      { zone_id: 8, junction_name: 'Marathahalli Bridge', risk_score: 7, active_violation_count: 2 },
      { zone_id: 5, junction_name: 'Koramangala Sony World', risk_score: 6, active_violation_count: 2 },
    ]
  };
}

export async function fetchZones() {
  try {
    const res = await fetch(`${API_BASE}/zones`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      return await res.json();
    }
  } catch (err) {
    // Fallback
  }
  return FALLBACK_ZONES;
}

export async function fetchIncidents() {
  try {
    const res = await fetch(`${API_BASE}/incidents`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      const data = await res.json();
      if (Array.isArray(data) && data.length > 0) return data;
    }
  } catch (err) {
    // Fallback
  }
  return FALLBACK_INCIDENTS;
}

export async function reportIncident(payload) {
  try {
    const res = await fetch(`${API_BASE}/incidents`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (res.ok) {
      const data = await res.json();
      return data.incident;
    }
  } catch (err) {
    // Local fallback
  }

  return {
    id: Math.floor(Math.random() * 9000) + 1000,
    zone_id: payload.zone_id,
    junction_name: FALLBACK_ZONES.find(z => z.id === payload.zone_id)?.junction_name || 'Urban Zone',
    violation_type: payload.violation_type,
    severity: payload.severity,
    status: 'REPORTED',
    occurred_at: 'Just now',
    description: payload.description,
  };
}

export async function dispatchOfficer(incidentId, badgeNumber) {
  try {
    const res = await fetch(`${API_BASE}/dispatch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ incident_id: incidentId, officer_badge_number: badgeNumber }),
    });
    return res.ok;
  } catch (err) {
    return true; // Local simulation
  }
}

export async function fetchForecast(zoneId, minutesAhead = 15) {
  try {
    const res = await fetch(`${API_BASE}/forecast`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ zone_id: zoneId, time_window_minutes: minutesAhead }),
    });
    if (res.ok) {
      return await res.json();
    }
  } catch (err) {
    // Fallback
  }

  const zone = FALLBACK_ZONES.find(z => z.id === zoneId) || FALLBACK_ZONES[0];
  return {
    id: 1,
    zone_id: zone.id,
    junction_name: zone.junction_name,
    target_timestamp: `+${minutesAhead} mins`,
    predicted_volume: zone.current_congestion_level === 'SEVERE' ? 0.82 : 0.54,
    congestion_index: zone.current_congestion_level === 'SEVERE' ? 82.0 : 54.0,
    model_confidence: 0.9308,
    congestion_level: zone.current_congestion_level,
    recommendation: zone.current_congestion_level === 'SEVERE'
      ? 'CRITICAL: Deploy traffic warden to clear parking bottleneck; extend green phase by 30s.'
      : 'ADVISORY: Steady commuter flow; keep monitoring entry ramps.'
  };
}
