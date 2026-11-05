import express from 'express';
import { AnalyticsService } from '../services/analyticsService.js';

const router = express.Router();

router.get('/dashboard', async (req, res) => {
  try {
    const summary = await AnalyticsService.getDashboardSummary();
    return res.json(summary);
  } catch (err) {
    return res.status(500).json({ detail: err.message });
  }
});

export default router;
