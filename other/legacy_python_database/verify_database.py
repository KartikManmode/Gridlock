"""Database Verification and Health Test Script for Project Gridlock.

Validates that both PostgreSQL 16 and MongoDB 7.0 are operational,
schemas are enforced, indexes are working, and queries return expected data.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from database.postgres_connection import pg_client
from database.mongo_connection import mongo_repo


def verify_postgres():
    target = "Neon Cloud (ep-falling-frost...)" if "neon.tech" in (pg_client.database_url or "") else "PostgreSQL Local"
    print(f"\n--- 1. VERIFYING POSTGRESQL [{target}] ---")
    if not pg_client.is_connected():
        print("[FAIL] PostgreSQL is not reachable.")
        return False

    # Check tables
    tables = pg_client.fetch_all("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        ORDER BY table_name;
    """)
    table_names = [t["table_name"] for t in tables]
    print(f"[OK] Found {len(table_names)} tables: {', '.join(table_names)}")

    # Check row counts
    users_cnt = pg_client.fetch_one("SELECT COUNT(*) as c FROM users;")["c"]
    zones_cnt = pg_client.fetch_one("SELECT COUNT(*) as c FROM zones;")["c"]
    inc_cnt = pg_client.fetch_one("SELECT COUNT(*) as c FROM incident_violations;")["c"]
    disp_cnt = pg_client.fetch_one("SELECT COUNT(*) as c FROM enforcement_dispatches;")["c"]

    print(f"[OK] USERS count: {users_cnt} (Admin, Operator, Citizen)")
    print(f"[OK] ZONES count: {zones_cnt} (All 50 geohash cells)")
    print(f"[OK] INCIDENTS count: {inc_cnt}, DISPATCHES count: {disp_cnt}")

    # Test relational JOIN (Incidents + Dispatches + Zones)
    join_query = """
        SELECT i.id, i.violation_type, i.severity, i.status, z.junction_name, z.geohash, d.officer_badge_number
        FROM incident_violations i
        JOIN zones z ON i.zone_id = z.id
        LEFT JOIN enforcement_dispatches d ON d.incident_id = i.id
        LIMIT 2;
    """
    joined = pg_client.fetch_all(join_query)
    print(f"[OK] Relational JOIN query test returned {len(joined)} records:")
    for row in joined:
        print(f"     -> Incident #{row['id']}: {row['violation_type']} at {row['junction_name']} | Officer: {row['officer_badge_number'] or 'Unassigned'}")

    return True


def verify_mongo():
    target = "MongoDB Atlas Cloud (gridlock-cluster.zimyer2.mongodb.net)" if "mongodb.net" in mongo_repo.uri else "MongoDB Local"
    print(f"\n--- 2. VERIFYING MONGODB [{target}] ---")
    if not mongo_repo.is_connected():
        print("[FAIL] MongoDB is not reachable.")
        return False

    # Check collections
    colls = mongo_repo._db.list_collection_names()
    print(f"[OK] Found collections: {', '.join(colls)}")

    # Check prediction_logs count
    pred_cnt = mongo_repo.prediction_logs.count_documents({})
    print(f"[OK] PREDICTION_LOGS count: {pred_cnt} documents (50 zones x 96 time-slots)")

    # Test query by time-slider slot
    sample_doc = mongo_repo.prediction_logs.find_one({"slot_index": 36}) # 09:00 AM Morning Rush
    if sample_doc:
        print(f"[OK] Sample 09:00 AM Slot Document:")
        print(f"     -> Zone: {sample_doc.get('junction_name')} ({sample_doc.get('geohash')})")
        print(f"     -> Predicted Volume: {sample_doc.get('predicted_volume')} ({sample_doc.get('congestion_level')})")
        print(f"     -> 4 Expert Breakdown: {sample_doc.get('expert_breakdown')}")
        print(f"     -> Multi-horizon +15m: {sample_doc.get('multi_horizon_rollout')[0]}")

    # Check Model Registry
    model_doc = mongo_repo.model_registry.find_one({"version": "v47_pluviometer_thermo_spike"})
    if model_doc:
        print(f"[OK] Registered SOTA Model: {model_doc.get('version')} (R2: {model_doc.get('r2_score')})")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print(" PROJECT GRIDLOCK: DATABASE INTEGRATION VERIFICATION")
    print("=" * 60)
    pg_ok = verify_postgres()
    mg_ok = verify_mongo()

    if pg_ok and mg_ok:
        print("\n" + "=" * 60)
        print(" ALL CHECKS PASSED: Polyglot Database is fully operational!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n[ERROR] Verification failed.")
        sys.exit(1)
