export type CongestionLevel = 'LOW' | 'MEDIUM' | 'HIGH' | 'SEVERE';
export type UserRole = 'admin' | 'operator' | 'citizen';
export type ViolationType = 'PARKING' | 'SPEEDING' | 'SIGNAL';
export type SeverityLevel = 'LOW' | 'MEDIUM' | 'HIGH';
export type IncidentStatus = 'REPORTED' | 'DISPATCHED' | 'RESOLVED' | 'DISMISSED';

export interface Zone {
  id: number;
  geohash: string;
  junction_name: string;
  latitude: number;
  longitude: number;
  capacity: number;
  current_congestion_level: CongestionLevel;
  zone_tier?: 'Core City' | 'Mid-Ring' | 'Periphery';
}

export interface HeatmapPoint {
  zone_id: number;
  junction_name: string;
  geohash: string;
  latitude: number;
  longitude: number;
  weight: number; // 0.0 to 1.0
  congestion_level: CongestionLevel;
  active_violations: number;
}

export interface CityKPIs {
  total_active_zones: number;
  average_city_speed_kmh: number;
  city_wide_congestion_index: number;
  active_incidents_count: number;
  dispatched_units_count: number;
  recent_forecasts_count: number;
  highest_congestion_zone: string;
}

export interface Incident {
  id: number;
  zone_id: number;
  junction_name: string;
  geohash?: string;
  violation_type: ViolationType;
  severity: SeverityLevel;
  status: IncidentStatus;
  occurred_at: string;
  officer_badge_number?: string;
  dispatch_time?: string;
  description?: string;
}

export interface ForecastResponse {
  id: number;
  zone_id: number;
  junction_name: string;
  target_timestamp: string;
  predicted_volume: number;
  congestion_index: number;
  model_confidence: number;
  congestion_level: CongestionLevel;
  recommendation: string;
}

export interface TimeSlot {
  slotIndex: number; // 0 to 95 (15 min intervals)
  timeString: string; // "09:15"
  displayLabel: string; // "09:15 AM"
  period: 'Morning Rush' | 'Midday' | 'Evening Rush' | 'Night';
}
