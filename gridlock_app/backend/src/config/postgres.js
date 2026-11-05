import pg from 'pg';
import { config } from './env.js';

const { Pool } = pg;

let pool = null;

if (config.DATABASE_URL) {
  try {
    pool = new Pool({
      connectionString: config.DATABASE_URL,
      ssl: {
        rejectUnauthorized: false,
      },
      max: 10,
      idleTimeoutMillis: 30000,
      connectionTimeoutMillis: 5000,
    });
  } catch (err) {
    console.error('[PostgreSQL] Failed to create connection pool:', err.message);
  }
}

/**
 * Normalizes numeric strings from PostgreSQL DECIMAL/NUMERIC to JS numbers
 */
function normalizeRow(row) {
  if (!row) return row;
  const out = { ...row };
  for (const [key, val] of Object.entries(out)) {
    if (typeof val === 'string' && /^-?\d+(\.\d+)?$/.test(val)) {
      // Check if it looks like a number (e.g., latitude, longitude, predicted_volume)
      if (
        key.includes('lat') ||
        key.includes('lon') ||
        key.includes('speed') ||
        key.includes('volume') ||
        key.includes('index') ||
        key.includes('confidence') ||
        key.includes('weight') ||
        key.includes('capacity') ||
        key.includes('count') ||
        key.includes('score')
      ) {
        out[key] = parseFloat(val);
      }
    }
  }
  return out;
}

export const db = {
  get pool() {
    return pool;
  },

  async query(text, params = []) {
    if (!pool) {
      throw new Error('PostgreSQL connection pool not initialized.');
    }
    const res = await pool.query(text, params);
    return res;
  },

  async fetchAll(text, params = []) {
    const res = await this.query(text, params);
    return res.rows.map(normalizeRow);
  },

  async fetchOne(text, params = []) {
    const res = await this.query(text, params);
    return res.rows.length > 0 ? normalizeRow(res.rows[0]) : null;
  },

  async isConnected() {
    if (!pool) return false;
    try {
      const res = await pool.query('SELECT 1;');
      return res.rowCount === 1;
    } catch {
      return false;
    }
  },
};
