import express from 'express';
import { ForecastingService } from '../services/forecastingService.js';
import { AuthService } from '../services/authService.js';
import { mongo } from '../config/mongo.js';

const router = express.Router();

router.post('/forecast', async (req, res) => {
  let userId = null;
  const authHeader = req.headers.authorization;
  if (authHeader && authHeader.startsWith('Bearer ')) {
    const token = authHeader.split(' ')[1];
    const payload = AuthService.verifyToken(token);
    if (payload) {
      userId = payload.sub;
    }
  }

  try {
    const forecast = await ForecastingService.generateForecast(req.body, userId);
    return res.json(forecast);
  } catch (err) {
    return res.status(400).json({ detail: err.message });
  }
});

router.get('/forecast/logs', async (req, res) => {
  try {
    const limit = parseInt(req.query.limit, 10) || 50;
    const logs = await ForecastingService.getRecentPredictions(limit);
    return res.json(logs);
  } catch (err) {
    return res.status(500).json({ detail: err.message });
  }
});

router.get('/forecast/atlas-rollouts', async (req, res) => {
  try {
    const isConnected = await mongo.isConnected();
    if (!isConnected) {
      return res.json({ connected: false, count: 0, rollouts: [] });
    }
    const targetTimestamp = req.query.target_timestamp;
    let data;
    if (targetTimestamp) {
      data = await mongo.getPredictionsForSlot(targetTimestamp);
    } else {
      const col = await mongo.getPredictionLogsCollection();
      data = col ? await col.find({}, { projection: { _id: 0 } }).limit(50).toArray() : [];
    }
    return res.json({ connected: true, count: data.length, rollouts: data });
  } catch (err) {
    return res.status(500).json({ detail: err.message });
  }
});

router.get('/forecast/atlas-series/:zone_id', async (req, res) => {
  try {
    const isConnected = await mongo.isConnected();
    const zoneId = parseInt(req.params.zone_id, 10);
    if (!isConnected) {
      return res.json({ connected: false, series: [] });
    }
    const limit = parseInt(req.query.limit, 10) || 96;
    const series = await mongo.getZoneForecastSeries(zoneId, limit);
    return res.json({
      connected: true,
      zone_id: zoneId,
      count: series.length,
      series,
    });
  } catch (err) {
    return res.status(500).json({ detail: err.message });
  }
});

export default router;
