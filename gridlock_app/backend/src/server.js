import { app } from './app.js';
import { config } from './config/env.js';
import { db } from './config/postgres.js';
import { mongo } from './config/mongo.js';

const PORT = config.PORT || 8000;

const server = app.listen(PORT, '0.0.0.0', async () => {
  console.log(`============================================================`);
  console.log(`🚀 GRIDLOCK EXPRESS BACKEND RUNNING ON http://0.0.0.0:${PORT}`);
  console.log(`============================================================`);

  const pgConnected = await db.isConnected();
  console.log(`  -> PostgreSQL (Neon Cloud): ${pgConnected ? 'CONNECTED' : 'DISCONNECTED'}`);

  const mongoConnected = await mongo.isConnected();
  console.log(`  -> MongoDB (Atlas Cloud):  ${mongoConnected ? 'CONNECTED' : 'DISCONNECTED'}`);
  console.log(`============================================================\n`);
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  console.log('SIGTERM signal received: closing HTTP server');
  server.close(async () => {
    if (db.pool) await db.pool.end();
    console.log('HTTP server closed');
  });
});
