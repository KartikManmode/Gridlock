-- Traffic Congestion Forecasting Platform Database Schema (SQLite compatible)
-- Directly corresponds to ERD: Traffic Congestion Monitoring Architecture

PRAGMA foreign_keys = ON;

-- D1: USERS Table (P1: Authentication & RBAC)
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('admin', 'operator', 'citizen')),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- D2: ZONES Table (P2: Zone & Sensor Management)
CREATE TABLE IF NOT EXISTS zones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    geohash TEXT UNIQUE NOT NULL,
    junction_name TEXT NOT NULL,
    latitude REAL NOT NULL CHECK(latitude >= -90.0 AND latitude <= 90.0),
    longitude REAL NOT NULL CHECK(longitude >= -180.0 AND longitude <= 180.0),
    capacity INTEGER NOT NULL CHECK(capacity > 0),
    current_congestion_level TEXT NOT NULL DEFAULT 'LOW' CHECK(current_congestion_level IN ('LOW', 'MEDIUM', 'HIGH', 'SEVERE'))
);

-- D2: TRAFFIC_SENSOR_RECORDS Table (P2: Zone & Sensor Management)
CREATE TABLE IF NOT EXISTS traffic_sensor_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    zone_id INTEGER NOT NULL REFERENCES zones(id) ON DELETE CASCADE,
    recorded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    vehicle_count INTEGER NOT NULL CHECK(vehicle_count >= 0),
    average_speed REAL NOT NULL CHECK(average_speed >= 0.0),
    weather_condition TEXT DEFAULT 'CLEAR'
);

-- D3: PREDICTION_LOGS Table (P3: Predictive Congestion Forecasting)
CREATE TABLE IF NOT EXISTS prediction_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    zone_id INTEGER NOT NULL REFERENCES zones(id) ON DELETE CASCADE,
    requested_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    target_timestamp TIMESTAMP NOT NULL,
    predicted_volume REAL NOT NULL CHECK(predicted_volume >= 0.0),
    congestion_index REAL NOT NULL CHECK(congestion_index >= 0.0 AND congestion_index <= 100.0),
    model_confidence REAL NOT NULL CHECK(model_confidence >= 0.0 AND model_confidence <= 1.0),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- D4: INCIDENT_VIOLATIONS Table (P4: Incident & Enforcement Dispatch)
CREATE TABLE IF NOT EXISTS incident_violations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    zone_id INTEGER NOT NULL REFERENCES zones(id) ON DELETE CASCADE,
    reported_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    violation_type TEXT NOT NULL CHECK(violation_type IN ('SPEEDING', 'PARKING', 'SIGNAL')),
    severity TEXT NOT NULL CHECK(severity IN ('LOW', 'MEDIUM', 'HIGH')),
    status TEXT NOT NULL DEFAULT 'REPORTED' CHECK(status IN ('REPORTED', 'DISPATCHED', 'RESOLVED', 'DISMISSED')),
    occurred_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- D4: ENFORCEMENT_DISPATCHES Table (P4: Incident & Enforcement Dispatch)
CREATE TABLE IF NOT EXISTS enforcement_dispatches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id INTEGER UNIQUE NOT NULL REFERENCES incident_violations(id) ON DELETE CASCADE,
    officer_badge_number TEXT NOT NULL,
    dispatch_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolution_notes TEXT,
    resolved_at TIMESTAMP
);

-- Indexes for high-frequency queries
CREATE INDEX IF NOT EXISTS idx_sensor_records_zone_time ON traffic_sensor_records(zone_id, recorded_at);
CREATE INDEX IF NOT EXISTS idx_prediction_logs_zone_time ON prediction_logs(zone_id, target_timestamp);
CREATE INDEX IF NOT EXISTS idx_incidents_zone_status ON incident_violations(zone_id, status);
CREATE INDEX IF NOT EXISTS idx_zones_geohash ON zones(geohash);
