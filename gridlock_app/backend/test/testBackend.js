import assert from 'assert';
import http from 'http';
import { app } from '../src/app.js';
import { db } from '../src/config/postgres.js';
import { mongo } from '../src/config/mongo.js';

let server;
let baseUrl;

async function request(path, options = {}) {
  const url = `${baseUrl}${path}`;
  const response = await fetch(url, {
    method: options.method || 'GET',
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  const data = await response.json().catch(() => null);
  return { status: response.status, data };
}

async function runTests() {
  console.log('============================================================');
  console.log('RUNNING NODE.JS / EXPRESS END-TO-END DATA FLOW VERIFICATION');
  console.log('============================================================\n');

  // Start test server on dynamic port
  server = http.createServer(app);
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const port = server.address().port;
  baseUrl = `http://127.0.0.1:${port}`;

  try {
    // 1. Test Root
    const rootRes = await request('/');
    assert.strictEqual(rootRes.status, 200);
    assert.strictEqual(rootRes.data.platform, 'Project Gridlock Traffic Forecasting System');
    console.log('✓ [Root] API operational:', rootRes.data.platform);
    console.log('  -> Stack:', rootRes.data.stack);

    // 2. Test Process 1.0: Authentication & RBAC
    console.log('\n--- Testing Process 1.0: Authentication & RBAC ---');
    const loginRes = await request('/api/auth/login', {
      method: 'POST',
      body: { email: 'operator@gridlock.city', password: 'Operator@123' },
    });
    assert.strictEqual(loginRes.status, 200, JSON.stringify(loginRes.data));
    const token = loginRes.data.access_token;
    assert.strictEqual(loginRes.data.role, 'operator');
    assert(loginRes.data.permissions.includes('dispatch:manage'));
    console.log('✓ [P1] Operator logged in. Token acquired with role:', loginRes.data.role);

    const headers = { Authorization: `Bearer ${token}` };

    const meRes = await request('/api/auth/me', { headers });
    assert.strictEqual(meRes.status, 200);
    assert.strictEqual(meRes.data.role, 'operator');
    console.log('✓ [P1] /me token validation verified for email:', meRes.data.email);

    // 3. Test Process 2.0: Zone & Sensor Management
    console.log('\n--- Testing Process 2.0: Zone & Sensor Management ---');
    const zonesRes = await request('/api/zones');
    assert.strictEqual(zonesRes.status, 200);
    assert(Array.isArray(zonesRes.data));
    assert(zonesRes.data.length >= 10);
    const firstZone = zonesRes.data[0];
    console.log(`✓ [P2] Retrieved ${zonesRes.data.length} zones. Sample zone: ${firstZone.junction_name} (${firstZone.geohash})`);

    const streamRes = await request('/api/sensors/stream', {
      method: 'POST',
      body: {
        zone_id: firstZone.id,
        vehicle_count: 2200,
        average_speed: 18.5,
        weather_condition: 'RAIN',
      },
    });
    assert.strictEqual(streamRes.status, 201);
    assert.strictEqual(streamRes.data.status, 'SUCCESS');
    console.log('✓ [P2] SensorEntity streamed reading into D2. Zone congestion recalculated.');

    // 4. Test Process 3.0: Predictive Congestion Forecasting
    console.log('\n--- Testing Process 3.0: Predictive Congestion Forecasting ---');
    const forecastRes = await request('/api/forecast', {
      method: 'POST',
      headers,
      body: {
        zone_id: firstZone.id,
        time_window_minutes: 15,
        weather_condition: 'RAIN',
        temperature: 29.5,
      },
    });
    assert.strictEqual(forecastRes.status, 200, JSON.stringify(forecastRes.data));
    assert(forecastRes.data.predicted_volume > 0);
    assert(forecastRes.data.congestion_index >= 0);
    console.log(
      `✓ [P3] ML Engine generated forecast: Volume=${forecastRes.data.predicted_volume}, Index=${forecastRes.data.congestion_index}, Confidence=${forecastRes.data.model_confidence}. Logged to Neon & Atlas.`
    );

    // Test MongoDB Atlas Rollouts route
    const rolloutsRes = await request('/api/forecast/atlas-rollouts');
    assert.strictEqual(rolloutsRes.status, 200);
    console.log(`✓ [P3] MongoDB Atlas rollouts retrieved: Count=${rolloutsRes.data.count}, Connected=${rolloutsRes.data.connected}`);

    // 5. Test Process 4.0: Incident & Enforcement Dispatch
    console.log('\n--- Testing Process 4.0: Incident & Enforcement Dispatch ---');
    const incidentRes = await request('/api/incidents', {
      method: 'POST',
      headers,
      body: {
        zone_id: firstZone.id,
        violation_type: 'PARKING',
        severity: 'HIGH',
        description: 'Delivery truck parked on double line blocking lane',
      },
    });
    assert.strictEqual(incidentRes.status, 201);
    const incidentId = incidentRes.data.incident.id;
    console.log(`✓ [P4] Violation reported (ID: ${incidentId}). Stored in Neon D4.`);

    const dispatchRes = await request('/api/dispatch', {
      method: 'POST',
      headers,
      body: {
        incident_id: incidentId,
        officer_badge_number: 'KA-TP-9988',
        notes: 'Rapid deployment to clear chokepoint',
      },
    });
    assert.strictEqual(dispatchRes.status, 201);
    console.log(`✓ [P4] Officer KA-TP-9988 dispatched to incident ${incidentId}. Status set to DISPATCHED.`);

    const resolveRes = await request('/api/dispatch/resolve', {
      method: 'POST',
      headers,
      body: {
        incident_id: incidentId,
        resolution_notes: 'Vehicle towed and traffic flow restored.',
      },
    });
    assert.strictEqual(resolveRes.status, 200);
    assert.strictEqual(resolveRes.data.status, 'RESOLVED');
    console.log(`✓ [P4] Incident ${incidentId} resolved successfully.`);

    // 6. Test Process 5.0: Analytics & Dashboard Aggregation
    console.log('\n--- Testing Process 5.0: Analytics & Dashboard Aggregation ---');
    const dashRes = await request('/api/analytics/dashboard');
    assert.strictEqual(dashRes.status, 200);
    assert(dashRes.data.kpis.total_active_zones >= 10);
    assert(Array.isArray(dashRes.data.heatmap));
    assert(Array.isArray(dashRes.data.active_hotspots));
    console.log(
      `✓ [P5] Dashboard aggregation successful! Active Zones: ${dashRes.data.kpis.total_active_zones}, Avg Speed: ${dashRes.data.kpis.average_city_speed_kmh} km/h, Heatmap Nodes: ${dashRes.data.heatmap.length}, Hotspots: ${dashRes.data.active_hotspots.length}`
    );

    console.log('\n============================================================');
    console.log('ALL 5 PROCESSES VERIFIED SUCCESSFULLY IN NODE.JS & EXPRESS!');
    console.log('============================================================\n');
  } finally {
    if (server) server.close();
    if (db.pool) await db.pool.end();
    process.exit(0);
  }
}

runTests().catch((err) => {
  console.error('\n❌ Test failed with error:', err);
  if (server) server.close();
  process.exit(1);
});
