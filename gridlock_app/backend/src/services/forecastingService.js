import { db } from '../config/postgres.js';
import { mongo } from '../config/mongo.js';
import { MLEngine } from '../ml_engine/inference.js';
import { LagExtractor } from './lagExtractor.js';
import { ZoneService } from './zoneService.js';

export class ForecastingService {
  static async generateForecast(req, requestedByUserId = null) {
    const zone = await ZoneService.getZoneById(req.zone_id);
    if (!zone) {
      throw new Error(`Zone ID ${req.zone_id} does not exist.`);
    }

    const timeWindow = req.time_window_minutes || 15;
    const now = new Date();
    const targetDt = new Date(now.getTime() + timeWindow * 60000);
    const targetStr = targetDt.toISOString().replace('T', ' ').substring(0, 19);

    // 1. Build Lag Feature Vector
    const featureVector = await LagExtractor.buildFeatureVector(
      req.zone_id,
      targetDt,
      req.weather_condition || 'CLEAR',
      req.temperature || 28.0
    );

    // 2. ML Engine Inference
    const { predictedVolume, congestionIndex, confidence, expertBreakdown } =
      MLEngine.predict(featureVector);

    // 3. Recommendation & Level
    let level = 'LOW';
    let recommendation = 'CLEAR: Optimal free-flow traffic condition.';
    if (congestionIndex >= 75.0) {
      level = 'SEVERE';
      recommendation = 'CRITICAL: Reroute traffic via outer ring road; dispatch rapid enforcement.';
    } else if (congestionIndex >= 50.0) {
      level = 'HIGH';
      recommendation = 'ADVISORY: Extend green light timing on primary artery by 20%.';
    } else if (congestionIndex >= 25.0) {
      level = 'MEDIUM';
      recommendation = 'MODERATE: Normal commuter flow expected; monitor choke points.';
    }

    // 4. Insert forecast record into D3 (prediction_logs in PostgreSQL)
    const insertedLog = await db.fetchOne(
      `INSERT INTO prediction_logs 
       (zone_id, requested_by, target_timestamp, predicted_volume, congestion_index, model_confidence)
       VALUES ($1, $2, $3, $4, $5, $6)
       RETURNING id`,
      [req.zone_id, requestedByUserId, targetDt, predictedVolume, congestionIndex, confidence]
    );

    // 5. Dual-persistence: Write high-resolution forecast document to MongoDB Atlas
    try {
      if (await mongo.isConnected()) {
        await mongo.insertPrediction({
          zone_id: req.zone_id,
          geohash: zone.geohash,
          junction_name: zone.junction_name,
          target_timestamp: targetStr,
          predicted_volume: predictedVolume,
          congestion_index: congestionIndex,
          model_confidence: confidence,
          congestion_level: level,
          recommendation,
          feature_vector: featureVector,
          expert_breakdown: expertBreakdown,
        });
      }
    } catch {
      // Optional logging failure should not block HTTP response
    }

    return {
      id: insertedLog ? insertedLog.id : null,
      zone_id: req.zone_id,
      junction_name: zone.junction_name,
      target_timestamp: targetStr,
      predicted_volume: predictedVolume,
      congestion_index: congestionIndex,
      model_confidence: confidence,
      congestion_level: level,
      recommendation,
    };
  }

  static async getRecentPredictions(limit = 50) {
    return await db.fetchAll(
      `SELECT p.*, z.junction_name, z.geohash, z.latitude, z.longitude
       FROM prediction_logs p
       JOIN zones z ON p.zone_id = z.id
       ORDER BY p.created_at DESC LIMIT $1`,
      [limit]
    );
  }
}
