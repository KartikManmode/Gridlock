import express from 'express';
import { IncidentService } from '../services/incidentService.js';
import { DispatchService } from '../services/dispatchService.js';
import { AuthService } from '../services/authService.js';

const router = express.Router();

router.post('/incidents', async (req, res) => {
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
    const incident = await IncidentService.reportViolation(req.body, userId);
    return res.status(201).json({
      status: 'REPORTED',
      message: 'Incident violation registered successfully',
      incident,
    });
  } catch (err) {
    return res.status(400).json({ detail: err.message });
  }
});

router.get('/incidents', async (req, res) => {
  try {
    const status = req.query.status_filter || null;
    const limit = parseInt(req.query.limit, 10) || 50;
    const incidents = await IncidentService.getIncidents(status, limit);
    return res.json(incidents);
  } catch (err) {
    return res.status(500).json({ detail: err.message });
  }
});

router.post('/dispatch', async (req, res) => {
  const authHeader = req.headers.authorization;
  if (authHeader && authHeader.startsWith('Bearer ')) {
    try {
      const token = authHeader.split(' ')[1];
      AuthService.checkPermission(token, 'dispatch:manage');
    } catch (err) {
      return res.status(403).json({ detail: err.message });
    }
  }

  try {
    const dispatch = await DispatchService.dispatchUnit(req.body);
    return res.status(201).json({
      status: 'DISPATCHED',
      message: `Officer ${req.body.officer_badge_number} dispatched to incident ${req.body.incident_id}`,
      dispatch,
    });
  } catch (err) {
    return res.status(400).json({ detail: err.message });
  }
});

router.post('/dispatch/resolve', async (req, res) => {
  const authHeader = req.headers.authorization;
  if (authHeader && authHeader.startsWith('Bearer ')) {
    try {
      const token = authHeader.split(' ')[1];
      AuthService.checkPermission(token, 'dispatch:manage');
    } catch (err) {
      return res.status(403).json({ detail: err.message });
    }
  }

  try {
    const result = await DispatchService.resolveIncident(req.body);
    return res.json({
      status: 'RESOLVED',
      message: `Incident ${req.body.incident_id} successfully resolved.`,
      incident_id: req.body.incident_id,
    });
  } catch (err) {
    return res.status(400).json({ detail: err.message });
  }
});

export default router;
