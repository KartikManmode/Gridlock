"""MongoDB Client & Repository for Project Gridlock.

Handles high-throughput document persistence for ML prediction logs,
multi-horizon rollout vectors, and feature snapshots.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

# Load environment configuration from .env
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
MONGO_PORT = int(os.getenv("MONGO_PORT", "27017"))
MONGO_DB = os.getenv("MONGO_DB", "gridlock_predictions")
MONGO_USER = os.getenv("MONGO_USER", "mongo_admin")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD", "mongo_secret_2026")

DEFAULT_URI = f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/{MONGO_DB}?authSource=admin"
MONGO_URI = os.getenv("MONGODB_URI", DEFAULT_URI)


class MongoRepository:
    """Thread-safe client managing MongoDB collections and queries."""

    def __init__(self, uri: str = MONGO_URI, db_name: str = MONGO_DB):
        self.uri = uri
        self.db_name = db_name
        self._client: Optional[MongoClient] = None
        self._db: Optional[Database] = None
        self._connect()

    def _connect(self) -> None:
        try:
            self._client = MongoClient(self.uri, serverSelectionTimeoutMS=3000)
            self._db = self._client[self.db_name]
        except Exception:
            self._client = None
            self._db = None

    def is_connected(self) -> bool:
        """Returns True if MongoDB instance responds to ping."""
        if not self._client:
            self._connect()
        if not self._client:
            return False
        try:
            self._client.admin.command("ping")
            return True
        except Exception:
            return False

    def init_indexes(self) -> None:
        """Ensures compound spatial-temporal indexes and TTL expiration on MongoDB."""
        if self._db is None:
            self._connect()
        try:
            self.prediction_logs.create_index([("geohash", 1), ("target_timestamp", 1)], name="idx_geohash_target_time", background=True)
            self.prediction_logs.create_index([("zone_id", 1), ("target_timestamp", 1)], name="idx_zone_target_time", background=True)
            self.prediction_logs.create_index([("congestion_level", 1), ("target_timestamp", 1)], name="idx_congestion_target_time", background=True)
            self.prediction_logs.create_index([("created_at", 1)], expireAfterSeconds=2592000, name="idx_ttl_30d_cleanup")
            self.feature_snapshots.create_index([("zone_id", 1), ("timestamp", 1)], unique=True, name="idx_zone_timestamp_unique")
            self.model_registry.create_index([("version", 1)], unique=True, name="idx_model_version_unique")
        except Exception as err:
            print(f"[MongoRepository] Index init note: {err}")

    @property
    def prediction_logs(self) -> Collection:
        """Prediction logs collection accessor."""
        if self._db is None:
            self._connect()
        return self._db["prediction_logs"]

    @property
    def feature_snapshots(self) -> Collection:
        """Feature snapshots collection accessor."""
        if self._db is None:
            self._connect()
        return self._db["feature_snapshots"]

    @property
    def model_registry(self) -> Collection:
        """Model registry collection accessor."""
        if self._db is None:
            self._connect()
        return self._db["model_registry"]

    def insert_prediction(self, doc: Dict[str, Any]) -> str:
        """Inserts a single prediction log document and returns its ID."""
        if "created_at" not in doc:
            doc["created_at"] = datetime.utcnow()
        res = self.prediction_logs.insert_one(doc)
        return str(res.inserted_id)

    def insert_predictions_batch(self, docs: List[Dict[str, Any]]) -> int:
        """Inserts multiple prediction logs efficiently in batch."""
        if not docs:
            return 0
        now = datetime.utcnow()
        for doc in docs:
            if "created_at" not in doc:
                doc["created_at"] = now
        res = self.prediction_logs.insert_many(docs, ordered=False)
        return len(res.inserted_ids)

    def get_predictions_for_slot(self, target_timestamp: str) -> List[Dict[str, Any]]:
        """Queries all zone predictions for a specific time-slider slot."""
        cursor = self.prediction_logs.find(
            {"target_timestamp": target_timestamp},
            {"_id": 0}
        )
        return list(cursor)

    def get_zone_forecast_series(self, zone_id: int, limit: int = 96) -> List[Dict[str, Any]]:
        """Returns chronological forecast documents for a specific zone."""
        cursor = self.prediction_logs.find(
            {"zone_id": zone_id},
            {"_id": 0}
        ).sort("target_timestamp", 1).limit(limit)
        return list(cursor)

    def close(self) -> None:
        """Closes the MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None


# Global singleton instance
mongo_repo = MongoRepository()
