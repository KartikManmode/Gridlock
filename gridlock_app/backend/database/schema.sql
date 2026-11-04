-- Traffic Congestion Forecasting Platform Database Schema (PostgreSQL)
-- Directly corresponds to ERD: Traffic Congestion Monitoring Architecture

-- D1: USERS Table (P1: Authentication & RBAC)
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL CHECK (role IN ('admin', 'operator', 'citizen')),
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- D2: ZONES Table (P2: Zone & Sensor Management)
CREATE TABLE IF NOT EXISTS zones (
    id SERIAL PRIMARY KEY,
    geohash VARCHAR(12) UNIQUE NOT NULL,
    junction_name VARCHAR(255) NOT NULL,
    latitude DECIMAL(9, 6) NOT NULL CHECK (latitude >= -90 AND latitude <= 90),
    longitude DECIMAL(9, 6) NOT NULL CHECK (longitude >= -180 AND longitude <= 180),
    capacity INTEGER NOT NULL CHECK (capacity > 0),
    current_congestion_level VARCHAR(20) NOT NULL DEFAULT 'LOW' CHECK (current_congestion_level IN ('LOW', 'MEDIUM', 'HIGH', 'SEVERE'))
);

-- D2: TRAFFIC_SENSOR_RECORDS Table (P2: Zone & Sensor Management)
CREATE TABLE IF NOT EXISTS traffic_sensor_records (
    id BIGSERIAL PRIMARY KEY,
    zone_id INTEGER NOT NULL REFERENCES zones(id) ON DELETE CASCADE,
    recorded_at TIMESTAMP NOT NULL DEFAULT NOW(),
    vehicle_count INTEGER NOT NULL CHECK (vehicle_count >= 0),
    average_speed DECIMAL(5, 2) NOT NULL CHECK (average_speed >= 0.0),
    weather_condition VARCHAR(50) DEFAULT 'CLEAR'
);

-- D3: PREDICTION_LOGS Table (P3: Predictive Congestion Forecasting)
CREATE TABLE IF NOT EXISTS prediction_logs (
    id BIGSERIAL PRIMARY KEY,
    zone_id INTEGER NOT NULL REFERENCES zones(id) ON DELETE CASCADE,
    requested_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    target_timestamp TIMESTAMP NOT NULL,
    predicted_volume DECIMAL(8, 4) NOT NULL CHECK (predicted_volume >= 0.0),
    congestion_index DECIMAL(5, 2) NOT NULL CHECK (congestion_index >= 0.0 AND congestion_index <= 100.0),
    model_confidence DECIMAL(5, 4) NOT NULL CHECK (model_confidence >= 0.0 AND model_confidence <= 1.0),
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- D4: INCIDENT_VIOLATIONS Table (P4: Incident & Enforcement Dispatch)
CREATE TABLE IF NOT EXISTS incident_violations (
    id BIGSERIAL PRIMARY KEY,
    zone_id INTEGER NOT NULL REFERENCES zones(id) ON DELETE CASCADE,
    reported_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    violation_type VARCHAR(50) NOT NULL CHECK (violation_type IN ('SPEEDING', 'PARKING', 'SIGNAL')),
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH')),
    status VARCHAR(50) NOT NULL DEFAULT 'REPORTED' CHECK (status IN ('REPORTED', 'DISPATCHED', 'RESOLVED', 'DISMISSED')),
    occurred_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- D4: ENFORCEMENT_DISPATCHES Table (P4: Incident & Enforcement Dispatch)
CREATE TABLE IF NOT EXISTS enforcement_dispatches (
    id BIGSERIAL PRIMARY KEY,
    incident_id BIGINT UNIQUE NOT NULL REFERENCES incident_violations(id) ON DELETE CASCADE,
    officer_badge_number VARCHAR(100) NOT NULL,
    dispatch_time TIMESTAMP NOT NULL DEFAULT NOW(),
    resolution_notes TEXT,
    resolved_at TIMESTAMP CHECK (resolved_at IS NULL OR resolved_at >= dispatch_time)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_sensor_records_zone_time ON traffic_sensor_records(zone_id, recorded_at);
CREATE INDEX IF NOT EXISTS idx_prediction_logs_zone_time ON prediction_logs(zone_id, target_timestamp);
CREATE INDEX IF NOT EXISTS idx_incidents_zone_status ON incident_violations(zone_id, status);
CREATE INDEX IF NOT EXISTS idx_zones_geohash ON zones(geohash);
