"""Database Seeding Script.

Populates initial data into tables (USERS, ZONES, TRAFFIC_SENSOR_RECORDS,
INCIDENT_VIOLATIONS, ENFORCEMENT_DISPATCHES, PREDICTION_LOGS)
using the project's actual datasets (dataset/train.csv, geohash_city_zones.csv, Phase2 police violations).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import pandas as pd
from backend.database.connection import db

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def hash_password(password: str) -> str:
    """PBKDF2 SHA-256 secure hash."""
    salt = "gridlock_salt_secure_2026"
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000).hex()


def seed_users():
    print("Seeding USERS (D1)...")
    users = [
        ("Traffic Administrator", "admin@gridlock.city", hash_password("Admin@123"), "admin"),
        ("Control Room Operator", "operator@gridlock.city", hash_password("Operator@123"), "operator"),
        ("Bangalore Citizen", "citizen@gridlock.city", hash_password("Citizen@123"), "citizen"),
    ]
    for name, email, pwd, role in users:
        existing = db.fetch_one("SELECT id FROM users WHERE email = ?", (email,))
        if not existing:
            db.execute(
                "INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, ?)",
                (name, email, pwd, role)
            )
    print(f"  -> Total users: {len(db.fetch_all('SELECT id FROM users'))}")


def seed_zones():
    print("Seeding ZONES (D2)...")
    candidate_paths = [
        ROOT_DIR / "data" / "geohash_city_zones.csv",
        ROOT_DIR / "ml_research" / "experiments_gemini" / "geohash_city_zones.csv",
        ROOT_DIR / "ml_research" / "sota_submissions" / "extracted_93.087" / "scripts" / "geohash_city_zones.csv",
        ROOT_DIR / "gemini_codes" / "geohash_city_zones.csv",
        ROOT_DIR / "extracted_93.087" / "scripts" / "geohash_city_zones.csv",
    ]
    zones_file = None
    for p in candidate_paths:
        if p.exists():
            zones_file = p
            break
    
    if zones_file.exists():
        df_zones = pd.read_csv(zones_file)
        count = 0
        for _, row in df_zones.head(50).iterrows():
            geohash = str(row["geohash"])
            junction_name = f"{row.get('zone_name', 'Zone')} - {geohash}"
            lat = float(row.get("lat", row.get("latitude", -5.36)))
            lon = float(row.get("lon", row.get("longitude", 90.78)))
            capacity = int(row.get("capacity", 1500)) if "capacity" in row else 1500
            
            existing = db.fetch_one("SELECT id FROM zones WHERE geohash = ?", (geohash,))
            if not existing:
                db.execute(
                    """INSERT INTO zones (geohash, junction_name, latitude, longitude, capacity, current_congestion_level)
                       VALUES (?, ?, ?, ?, ?, 'LOW')""",
                    (geohash, junction_name, lat, lon, capacity)
                )
                count += 1
        print(f"  -> Inserted {count} zones from {zones_file.name}")
    else:
        # Fallback zones
        defaults = [
            ("qp02z1", "MG Road Junction", 12.9756, 77.6066, 2000, "HIGH"),
            ("qp02zt", "Silk Board Chokepoint", 12.9176, 77.6238, 3500, "SEVERE"),
            ("qp08bj", "Indiranagar 100ft Rd", 12.9784, 77.6408, 1800, "MEDIUM"),
            ("qp08gt", "Hebbal Flyover", 13.0358, 77.5970, 3000, "MEDIUM"),
        ]
        for gh, name, lat, lon, cap, cong in defaults:
            if not db.fetch_one("SELECT id FROM zones WHERE geohash = ?", (gh,)):
                db.execute(
                    "INSERT INTO zones (geohash, junction_name, latitude, longitude, capacity, current_congestion_level) VALUES (?, ?, ?, ?, ?, ?)",
                    (gh, name, lat, lon, cap, cong)
                )
        print("  -> Inserted fallback zones")


def seed_sensor_records():
    print("Seeding TRAFFIC_SENSOR_RECORDS (D2)...")
    candidate_paths = [
        ROOT_DIR / "data" / "dataset" / "train.csv",
        ROOT_DIR / "data" / "train.csv",
        ROOT_DIR / "dataset" / "train.csv",
    ]
    train_file = None
    for p in candidate_paths:
        if p.exists():
            train_file = p
            break
    if train_file and train_file.exists():
        # Read a sample of train data
        df_train = pd.read_csv(train_file, nrows=100)
        zones = {z["geohash"]: z["id"] for z in db.fetch_all("SELECT id, geohash FROM zones")}
        count = 0
        for _, row in df_train.iterrows():
            gh = str(row["geohash"])
            if gh in zones:
                zone_id = zones[gh]
                # Map demand (0-1) to realistic vehicle count & speed
                demand = float(row.get("demand", 0.1))
                vehicle_count = int(demand * 1200)
                speed = max(5.0, (1.0 - demand) * 60.0)
                weather = str(row.get("Weather", "CLEAR"))
                if pd.isna(weather) or weather == "nan":
                    weather = "CLEAR"
                
                db.execute(
                    """INSERT INTO traffic_sensor_records (zone_id, vehicle_count, average_speed, weather_condition)
                       VALUES (?, ?, ?, ?)""",
                    (zone_id, vehicle_count, round(speed, 2), weather)
                )
                count += 1
        print(f"  -> Inserted {count} sensor readings from train.csv")


def seed_incidents():
    print("Seeding INCIDENT_VIOLATIONS & ENFORCEMENT_DISPATCHES (D4)...")
    candidate_paths = [
        ROOT_DIR / "data" / "Phase2" / "jan to may police violation_anonymized791b166.csv",
        ROOT_DIR / "data" / "violations" / "jan to may police violation_anonymized791b166.csv",
        ROOT_DIR / "Phase2" / "jan to may police violation_anonymized791b166.csv",
    ]
    police_file = None
    for p in candidate_paths:
        if p.exists():
            police_file = p
            break
    zones = db.fetch_all("SELECT id FROM zones LIMIT 5")
    if not zones:
        return
    
    zone_ids = [z["id"] for z in zones]
    sample_violations = [
        (zone_ids[0], "PARKING", "HIGH", "DISPATCHED"),
        (zone_ids[1 % len(zone_ids)], "SPEEDING", "MEDIUM", "REPORTED"),
        (zone_ids[2 % len(zone_ids)], "SIGNAL", "HIGH", "DISPATCHED"),
        (zone_ids[3 % len(zone_ids)], "PARKING", "LOW", "RESOLVED"),
    ]
    for i, (zid, vtype, sev, stat) in enumerate(sample_violations):
        inc_id = db.execute(
            """INSERT INTO incident_violations (zone_id, violation_type, severity, status)
               VALUES (?, ?, ?, ?)""",
            (zid, vtype, sev, stat)
        )
        if stat in ("DISPATCHED", "RESOLVED"):
            badge = f"KA-TP-{1042 + i}"
            notes = "Officer deployed on-site to clear bottleneck" if stat == "DISPATCHED" else "Violation ticket issued and obstacle cleared"
            resolved_at = "CURRENT_TIMESTAMP" if stat == "RESOLVED" else None
            db.execute(
                """INSERT INTO enforcement_dispatches (incident_id, officer_badge_number, resolution_notes)
                   VALUES (?, ?, ?)""",
                (inc_id, badge, notes)
            )
    print("  -> Seeded violations and dispatches (D4)")


def seed_prediction_logs():
    print("Seeding PREDICTION_LOGS (D3)...")
    zones = db.fetch_all("SELECT id FROM zones LIMIT 5")
    user = db.fetch_one("SELECT id FROM users WHERE role = 'operator'")
    user_id = user["id"] if user else None
    
    from datetime import datetime, timedelta
    target_time = (datetime.utcnow() + timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S")

    for z in zones:
        db.execute(
            """INSERT INTO prediction_logs (zone_id, requested_by, target_timestamp, predicted_volume, congestion_index, model_confidence)
               VALUES (?, ?, ?, 0.4250, 42.50, 0.9308)""",
            (z["id"], user_id, target_time)
        )
    print("  -> Seeded prediction logs (D3)")


def run_all_seeds():
    db.init_db()
    seed_users()
    seed_zones()
    seed_sensor_records()
    seed_incidents()
    seed_prediction_logs()
    print("Seeding completed successfully!")


if __name__ == "__main__":
    run_all_seeds()
