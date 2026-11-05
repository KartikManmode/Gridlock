/**
 * Project Gridlock: Polyglot Database Verification (Node.js)
 * Verifies Neon Serverless PostgreSQL & MongoDB Atlas Cloud
 */
import { db } from '../backend/src/config/postgres.js';
import { mongo } from '../backend/src/config/mongo.js';

async function verifyPostgres() {
  console.log('--- 1. VERIFYING POSTGRESQL [Neon Cloud] ---');
  const isConnected = await db.isConnected();
  if (!isConnected) {
    console.error('❌ Could not connect to Neon PostgreSQL.');
    return false;
  }

  const tables = await db.fetchAll(`
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public' 
    ORDER BY table_name;
  `);
  console.log(`[OK] Found ${tables.length} tables:`, tables.map((t) => t.table_name).join(', '));

  const usersCount = (await db.fetchOne('SELECT COUNT(*)::int as count FROM users')).count;
  const zonesCount = (await db.fetchOne('SELECT COUNT(*)::int as count FROM zones')).count;
  const incCount = (await db.fetchOne('SELECT COUNT(*)::int as count FROM incident_violations')).count;
  const dispCount = (await db.fetchOne('SELECT COUNT(*)::int as count FROM enforcement_dispatches')).count;

  console.log(`[OK] USERS count: ${usersCount}`);
  console.log(`[OK] ZONES count: ${zonesCount}`);
  console.log(`[OK] INCIDENTS count: ${incCount}, DISPATCHES count: ${dispCount}`);

  const sampleJoin = await db.fetchAll(`
    SELECT i.id, i.violation_type, z.junction_name, z.zone_tier,
           COALESCE(d.officer_badge_number, 'Unassigned') as officer
    FROM incident_violations i
    JOIN zones z ON i.zone_id = z.id
    LEFT JOIN enforcement_dispatches d ON i.id = d.incident_id
    ORDER BY i.occurred_at DESC LIMIT 2;
  `);

  console.log(`[OK] Relational JOIN query test returned ${sampleJoin.length} records:`);
  sampleJoin.forEach((r, idx) => {
    console.log(`     -> Incident #${idx + 1}: ${r.violation_type} at ${r.junction_name} | Officer: ${r.officer}`);
  });

  return true;
}

async function verifyMongo() {
  console.log('\n--- 2. VERIFYING MONGODB [MongoDB Atlas Cloud] ---');
  const isConnected = await mongo.isConnected();
  if (!isConnected) {
    console.error('❌ Could not connect to MongoDB Atlas.');
    return false;
  }

  const database = await mongo.getDb();
  const cols = await database.listCollections().toArray();
  console.log('[OK] Found collections:', cols.map((c) => c.name).join(', '));

  const count = await database.collection('prediction_logs').countDocuments();
  console.log(`[OK] PREDICTION_LOGS count: ${count} documents`);

  const sampleDoc = await database.collection('prediction_logs').findOne(
    { target_timestamp: { $regex: '09:00' } },
    { projection: { _id: 0 } }
  );

  if (sampleDoc) {
    console.log('[OK] Sample 09:00 AM Slot Document:');
    console.log(`     -> Zone: ${sampleDoc.junction_name || sampleDoc.zone_id} (${sampleDoc.geohash || ''})`);
    console.log(`     -> Predicted Volume: ${sampleDoc.predicted_volume} (${sampleDoc.congestion_level || ''})`);
    if (sampleDoc.expert_breakdown) {
      console.log(`     -> 4 Expert Breakdown:`, sampleDoc.expert_breakdown);
    }
  }

  return true;
}

async function main() {
  console.log('============================================================');
  console.log(' PROJECT GRIDLOCK: DATABASE INTEGRATION VERIFICATION (NODE.JS)');
  console.log('============================================================\n');

  try {
    const pgOk = await verifyPostgres();
    const mgOk = await verifyMongo();

    console.log('\n============================================================');
    if (pgOk && mgOk) {
      console.log(' ALL CHECKS PASSED: Polyglot Database is fully operational!');
    } else {
      console.log(' ⚠️ Some database checks failed.');
    }
    console.log('============================================================\n');
  } finally {
    if (db.pool) await db.pool.end();
    process.exit(0);
  }
}

main().catch((err) => {
  console.error('Verification error:', err);
  process.exit(1);
});
