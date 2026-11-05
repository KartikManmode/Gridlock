import express from 'express';
import cors from 'cors';

import authRouter from './routes/auth.js';
import zonesRouter from './routes/zones.js';
import forecastingRouter from './routes/forecasting.js';
import incidentsRouter from './routes/incidents.js';
import analyticsRouter from './routes/analytics.js';

export const app = express();

app.use(cors({
  origin: '*',
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['*'],
}));

app.use(express.json());

// Mount API routes under /api
app.use('/api/auth', authRouter);
app.use('/api', zonesRouter);
app.use('/api', forecastingRouter);
app.use('/api', incidentsRouter);
app.use('/api/analytics', analyticsRouter);

// Health check / root entrypoint
app.get('/', (req, res) => {
  res.json({
    platform: 'Project Gridlock Traffic Forecasting System',
    status: 'OPERATIONAL',
    stack: 'Node.js + Express 2.0',
    architecture: {
      P1: '1.0 Authentication & RBAC',
      P2: '2.0 Zone & Sensor Management',
      P3: '3.0 Predictive Congestion Forecasting',
      P4: '4.0 Incident & Enforcement Dispatch',
      P5: '5.0 Analytics & Dashboard Aggregation',
      ML_Engine: 'SOTA 93.087% Ensemble (Signal, Spatial, Shock, Oracle + Pluviometer)',
      Data_Stores: [
        'D1: Neon PostgreSQL (users)',
        'D2: Neon PostgreSQL (zones/sensor records)',
        'D3: Neon & MongoDB Atlas (prediction_logs / rollouts)',
        'D4: Neon PostgreSQL (incidents/dispatches)',
      ],
    },
    docs_url: '/api/analytics/dashboard',
  });
});

// Fallback 404
app.use((req, res) => {
  res.status(404).json({ detail: `Cannot ${req.method} ${req.url}` });
});

// Global error handler
app.use((err, req, res, next) => {
  console.error('[Unhandled Error]', err);
  res.status(500).json({ detail: err.message || 'Internal Server Error' });
});
