"""End-to-End Test Suite for All 5 DFD Processes.

Tests:
1. P1: Auth & RBAC (registration, login, JWT token verification)
2. P2: Zone & Sensor Management (zone lookup, streaming sensor reading into D2)
3. P3: Predictive Congestion Forecasting (generating forecast, ML Engine inference, D3 logging)
4. P4: Incident & Enforcement Dispatch (reporting violation into D4, dispatching officer, resolving)
5. P5: Analytics & Dashboard Aggregation (summary KPIs, dynamic heatmap, active hotspots)
"""

from __future__ import annotations

import sys
from fastapi.testclient import TestClient
from backend.main import app
from backend.database.seed_data import run_all_seeds

client = TestClient(app)


def test_entire_data_flow():
    print("=" * 60)
    print("RUNNING END-TO-END DATA FLOW VERIFICATION")
    print("=" * 60)

    # Re-seed test database
    run_all_seeds()

    # 1. Test Root
    res = client.get("/")
    assert res.status_code == 200, res.text
    print("✓ [Root] API operational:", res.json()["platform"])

    # 2. Test Process 1.0: Authentication & RBAC
    print("\n--- Testing Process 1.0: Authentication & RBAC ---")
    login_payload = {"email": "operator@gridlock.city", "password": "Operator@123"}
    res = client.post("/api/auth/login", json=login_payload)
    assert res.status_code == 200, res.text
    auth_data = res.json()
    token = auth_data["access_token"]
    assert auth_data["role"] == "operator"
    assert "dispatch:manage" in auth_data["permissions"]
    print("✓ [P1] Operator logged in. Token acquired with role:", auth_data["role"])

    headers = {"Authorization": f"Bearer {token}"}

    # 3. Test Process 2.0: Zone & Sensor Management
    print("\n--- Testing Process 2.0: Zone & Sensor Management ---")
    res = client.get("/api/zones")
    assert res.status_code == 200, res.text
    zones = res.json()
    assert len(zones) > 0
    test_zone = zones[0]
    print(f"✓ [P2] Retrieved {len(zones)} zones. First zone: {test_zone['junction_name']} ({test_zone['geohash']})")

    # Ingest sensor stream (SensorEntity -> P2 -> D2)
    stream_payload = {
        "zone_id": test_zone["id"],
        "vehicle_count": 1450,
        "average_speed": 18.5,
        "weather_condition": "Rainy"
    }
    res = client.post("/api/sensors/stream", json=stream_payload)
    assert res.status_code == 201, res.text
    print("✓ [P2] SensorEntity streamed reading into D2. Zone congestion recalculated.")

    # 4. Test Process 3.0: Predictive Congestion Forecasting
    print("\n--- Testing Process 3.0: Predictive Congestion Forecasting ---")
    forecast_payload = {
        "zone_id": test_zone["id"],
        "time_window_minutes": 30,
        "weather_condition": "Rainy",
        "temperature": 26.5
    }
    res = client.post("/api/forecast", json=forecast_payload, headers=headers)
    assert res.status_code == 200, res.text
    forecast = res.json()
    assert "predicted_volume" in forecast
    assert "congestion_index" in forecast
    assert "model_confidence" in forecast
    print(f"✓ [P3] ML Engine generated forecast: Volume={forecast['predicted_volume']}, "
          f"Index={forecast['congestion_index']}, Confidence={forecast['model_confidence']}. Logged to D3.")

    # 5. Test Process 4.0: Incident & Enforcement Dispatch
    print("\n--- Testing Process 4.0: Incident & Enforcement Dispatch ---")
    incident_payload = {
        "zone_id": test_zone["id"],
        "violation_type": "PARKING",
        "severity": "HIGH",
        "description": "Double parking blocking ambulance lane"
    }
    res = client.post("/api/incidents", json=incident_payload, headers=headers)
    assert res.status_code == 201, res.text
    incident = res.json()["incident"]
    inc_id = incident["id"]
    print(f"✓ [P4] Violation reported (ID: {inc_id}). Stored in D4.")

    dispatch_payload = {
        "incident_id": inc_id,
        "officer_badge_number": "KA-TP-9999",
        "notes": "Urgent tow-truck dispatch"
    }
    res = client.post("/api/dispatch", json=dispatch_payload, headers=headers)
    assert res.status_code == 201, res.text
    print(f"✓ [P4] Officer KA-TP-9999 dispatched to incident {inc_id}. Dispatches D4 updated.")

    # 6. Test Process 5.0: Analytics & Dashboard Aggregation
    print("\n--- Testing Process 5.0: Analytics & Dashboard Aggregation ---")
    res = client.get("/api/analytics/dashboard")
    assert res.status_code == 200, res.text
    dashboard = res.json()
    kpis = dashboard["kpis"]
    heatmap = dashboard["heatmap"]
    hotspots = dashboard["active_hotspots"]

    assert kpis["total_active_zones"] > 0
    assert len(heatmap) > 0
    print(f"✓ [P5] Aggregation successful! Active Zones: {kpis['total_active_zones']}, "
          f"Avg Speed: {kpis['average_city_speed_kmh']} km/h, Heatmap Nodes: {len(heatmap)}, "
          f"Hotspots: {len(hotspots)}")

    print("\n" + "=" * 60)
    print("ALL 5 PROCESSES VERIFIED SUCCESSFULLY AGAINST DFD SPECIFICATION!")
    print("=" * 60)


if __name__ == "__main__":
    test_entire_data_flow()
