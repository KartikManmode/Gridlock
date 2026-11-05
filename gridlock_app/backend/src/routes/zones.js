import express from 'express';
import { ZoneService } from '../services/zoneService.js';
import { SensorService } from '../services/sensorService.js';
import { AuthService } from '../services/authService.js';

const router = express.Router();

router.get('/zones', async (req, res) => {
  try {
    const zones = await ZoneService.getAllZones();
    return res.json(zones);
  } catch (err) {
    return res.status(500).json({ detail: err.message });
  }
});

router.get('/zones/:zone_id', async (req, res) => {
  try {
    const zone = await ZoneService.getZoneById(parseInt(req.params.zone_id, 10));
    if (!zone) {
      return res.status(404).json({ detail: 'Zone not found.' });
    }
    return res.json(zone);
  } catch (err) {
    return res.status(500).json({ detail: err.message });
  }
});

router.post('/zones', async (req, res) => {
  const authHeader = req.headers.authorization;
  if (authHeader && authHeader.startsWith('Bearer ')) {
    try {
      const token = authHeader.split(' ')[1];
      AuthService.checkPermission(token, 'zones:manage');
    } catch (err) {
      return res.status(403).json({ detail: err.message });
    }
  }

  try {
    const zone = await ZoneService.createZone(req.body);
    return res.status(201).json(zone);
  } catch (err) {
    return res.status(400).json({ detail: err.message });
  }
});

router.post('/sensors/stream', async (req, res) => {
  try {
    const inserted = await SensorService.ingestReading(req.body);
    return res.status(201).json({
      status: 'SUCCESS',
      message: 'Sensor reading recorded',
      record: inserted,
    });
  } catch (err) {
    return res.status(400).json({ detail: err.message });
  }
});

export default router;
