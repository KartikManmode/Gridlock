"""KPI Calculation Engine for Traffic Forecasting Platform.

Aggregates metrics across D2 (Sensors), D3 (Predictions), and D4 (Incidents).
"""

from __future__ import annotations

from typing import Dict, Any
from backend.database.connection import db
from backend.p5_analytics_dashboard.models import CityKPIs


class KPICalculator:
    """Computes city-wide traffic metrics."""

    @classmethod
    def calculate_summary_kpis(cls) -> CityKPIs:
        # Total Zones (D2)
        zones = db.fetch_all("SELECT id, junction_name, current_congestion_level FROM zones")
        total_zones = len(zones)

        # Average speed from latest sensor logs (D2)
        speed_row = db.fetch_one(
            """SELECT AVG(average_speed) as avg_spd, AVG(vehicle_count) as avg_cnt 
               FROM (SELECT average_speed, vehicle_count FROM traffic_sensor_records ORDER BY recorded_at DESC LIMIT 100)"""
        )
        avg_speed = round(float(speed_row["avg_spd"]), 2) if speed_row and speed_row["avg_spd"] else 38.5

        # City-wide Congestion Index from latest prediction logs (D3)
        pred_row = db.fetch_one(
            """SELECT AVG(congestion_index) as avg_cong, COUNT(id) as pred_cnt 
               FROM (SELECT congestion_index, id FROM prediction_logs ORDER BY created_at DESC LIMIT 50)"""
        )
        avg_cong = round(float(pred_row["avg_cong"]), 2) if pred_row and pred_row["avg_cong"] else 42.0
        pred_cnt = int(pred_row["pred_cnt"]) if pred_row and pred_row["pred_cnt"] else 0

        # Incidents & Dispatches (D4)
        inc_row = db.fetch_one(
            """SELECT 
               SUM(CASE WHEN status IN ('REPORTED', 'DISPATCHED') THEN 1 ELSE 0 END) as active_inc,
               SUM(CASE WHEN status = 'DISPATCHED' THEN 1 ELSE 0 END) as dispatched_inc
               FROM incident_violations"""
        )
        active_inc = int(inc_row["active_inc"]) if inc_row and inc_row["active_inc"] else 0
        dispatched_inc = int(inc_row["dispatched_inc"]) if inc_row and inc_row["dispatched_inc"] else 0

        # Highest Congestion Zone
        highest_zone = db.fetch_one(
            """SELECT junction_name FROM zones 
               WHERE current_congestion_level IN ('SEVERE', 'HIGH')
               LIMIT 1"""
        )
        highest_name = highest_zone["junction_name"] if highest_zone else (zones[0]["junction_name"] if zones else "None")

        return CityKPIs(
            total_active_zones=total_zones,
            average_city_speed_kmh=avg_speed,
            city_wide_congestion_index=avg_cong,
            active_incidents_count=active_inc,
            dispatched_units_count=dispatched_inc,
            recent_forecasts_count=pred_cnt,
            highest_congestion_zone=highest_name
        )
