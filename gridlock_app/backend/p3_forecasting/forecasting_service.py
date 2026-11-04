"""Process 3.0: Predictive Congestion Forecasting Service.

Orchestrates:
1. UserEntity forecast requests (Zone ID, Time Window)
2. Fetching historical lag readings from D2
3. Forwarding feature vector to MLEngine
4. Receiving predicted volume and model confidence
5. Writing forecast record into D3 (prediction_logs Table)
6. Returning prediction response to UserEntity
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional
from backend.database.connection import db
from backend.ml_engine.inference import MLEngine
from backend.p3_forecasting.models import ForecastRequest, ForecastResponse
from backend.p3_forecasting.lag_extractor import LagExtractor
from backend.p2_zone_sensors.zone_service import ZoneService


class ForecastingService:
    """Predictive forecasting workflow corresponding to Process 3.0."""

    @classmethod
    def generate_forecast(
        cls,
        req: ForecastRequest,
        requested_by_user_id: Optional[int] = None
    ) -> ForecastResponse:
        zone = ZoneService.get_zone_by_id(req.zone_id)
        if not zone:
            raise ValueError(f"Zone ID {req.zone_id} does not exist.")

        # Target timestamp
        now = datetime.datetime.now(datetime.timezone.utc)
        target_dt = now + datetime.timedelta(minutes=req.time_window_minutes)
        target_str = target_dt.strftime("%Y-%m-%d %H:%M:%S")

        # 1. Select historical lag readings from D2 and build vector
        feature_vector = LagExtractor.build_feature_vector(
            zone_id=req.zone_id,
            target_dt=target_dt,
            weather=req.weather_condition or "CLEAR",
            temperature=req.temperature or 28.0
        )

        # 2. Forward lag feature vector to ML Engine
        predicted_vol, congestion_idx, confidence = MLEngine.predict(feature_vector)

        # 3. Categorize level & recommendation
        if congestion_idx >= 75.0:
            level = "SEVERE"
            recommendation = "CRITICAL: Reroute traffic via outer ring road; dispatch rapid enforcement."
        elif congestion_idx >= 50.0:
            level = "HIGH"
            recommendation = "ADVISORY: Extend green light timing on primary artery by 20%."
        elif congestion_idx >= 25.0:
            level = "MEDIUM"
            recommendation = "MODERATE: Normal commuter flow expected; monitor choke points."
        else:
            level = "LOW"
            recommendation = "CLEAR: Optimal free-flow traffic condition."

        # 4. Insert forecast record into D3 (prediction_logs in PostgreSQL)
        log_id = db.execute(
            """INSERT INTO prediction_logs 
               (zone_id, requested_by, target_timestamp, predicted_volume, congestion_index, model_confidence)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (req.zone_id, requested_by_user_id, target_str, predicted_vol, congestion_idx, confidence)
        )

        # 5. Dual-persistence: Write high-resolution forecast document to MongoDB Atlas
        try:
            from database.mongo_connection import mongo_repo
            if mongo_repo.is_connected():
                mongo_repo.insert_prediction({
                    "zone_id": req.zone_id,
                    "geohash": zone.get("geohash", ""),
                    "junction_name": zone.get("junction_name", ""),
                    "target_timestamp": target_str,
                    "predicted_volume": float(predicted_vol),
                    "congestion_index": float(congestion_idx),
                    "model_confidence": float(confidence),
                    "congestion_level": level,
                    "recommendation": recommendation,
                    "feature_vector": feature_vector,
                })
        except Exception:
            pass

        return ForecastResponse(
            id=log_id,
            zone_id=req.zone_id,
            junction_name=zone["junction_name"],
            target_timestamp=target_str,
            predicted_volume=predicted_vol,
            congestion_index=congestion_idx,
            model_confidence=confidence,
            congestion_level=level,
            recommendation=recommendation
        )

    @classmethod
    def get_recent_predictions(cls, limit: int = 50) -> List[Dict[str, Any]]:
        """Queries recent prediction logs from D3 for analytics."""
        return db.fetch_all(
            """SELECT p.*, z.junction_name, z.geohash, z.latitude, z.longitude
               FROM prediction_logs p
               JOIN zones z ON p.zone_id = z.id
               ORDER BY p.created_at DESC LIMIT ?""",
            (limit,)
        )
