import { MongoClient } from 'mongodb';
import { config } from './env.js';

let client = null;
let database = null;
let isConnecting = false;

const DB_NAME = 'gridlock_predictions';

function getFallbackDirectUri(srvUri) {
  const match = srvUri.match(/^mongodb\+srv:\/\/([^:]+):([^@]+)@([^\/]+)(\/.*)?$/);
  if (!match) return srvUri;
  const [, user, pass, host, pathPart] = match;
  if (host.includes('gridlock-cluster.zimyer2.mongodb.net')) {
    const dbPath = pathPart ? pathPart.split('?')[0] : `/${DB_NAME}`;
    return `mongodb://${user}:${pass}@ac-jd8jtvp-shard-00-00.zimyer2.mongodb.net:27017,ac-jd8jtvp-shard-00-01.zimyer2.mongodb.net:27017,ac-jd8jtvp-shard-00-02.zimyer2.mongodb.net:27017${dbPath}?ssl=true&replicaSet=atlas-wo2oxa-shard-0&authSource=admin&retryWrites=true&w=majority`;
  }
  return srvUri;
}

async function connectMongo() {
  if (database) return database;
  if (!config.MONGODB_URI) return null;
  if (isConnecting) return null;

  try {
    isConnecting = true;
    let uriToUse = config.MONGODB_URI;

    try {
      client = new MongoClient(uriToUse, {
        serverSelectionTimeoutMS: 3000,
      });
      await client.connect();
    } catch (err) {
      if (err.message && err.message.includes('querySrv')) {
        // Fallback for macOS Node c-ares DNS SRV resolution
        uriToUse = getFallbackDirectUri(config.MONGODB_URI);
        client = new MongoClient(uriToUse, {
          serverSelectionTimeoutMS: 4000,
        });
        await client.connect();
      } else {
        throw err;
      }
    }

    database = client.db(DB_NAME);
    return database;
  } catch (err) {
    client = null;
    database = null;
    return null;
  } finally {
    isConnecting = false;
  }
}

export const mongo = {
  async getDb() {
    if (!database) {
      await connectMongo();
    }
    return database;
  },

  async isConnected() {
    try {
      const db = await this.getDb();
      if (!db) return false;
      const res = await db.command({ ping: 1 });
      return res.ok === 1;
    } catch {
      return false;
    }
  },

  async getPredictionLogsCollection() {
    const db = await this.getDb();
    return db ? db.collection('prediction_logs') : null;
  },

  async insertPrediction(doc) {
    try {
      const col = await this.getPredictionLogsCollection();
      if (!col) return null;
      if (!doc.created_at) {
        doc.created_at = new Date();
      }
      const res = await col.insertOne(doc);
      return res.insertedId.toString();
    } catch (err) {
      console.error('[MongoDB] Insert prediction note:', err.message);
      return null;
    }
  },

  async getPredictionsForSlot(targetTimestamp) {
    try {
      const col = await this.getPredictionLogsCollection();
      if (!col) return [];
      const cursor = col.find(
        { target_timestamp: targetTimestamp },
        { projection: { _id: 0 } }
      );
      return await cursor.toArray();
    } catch {
      return [];
    }
  },

  async getZoneForecastSeries(zoneId, limit = 96) {
    try {
      const col = await this.getPredictionLogsCollection();
      if (!col) return [];
      const cursor = col
        .find({ zone_id: parseInt(zoneId, 10) }, { projection: { _id: 0 } })
        .sort({ target_timestamp: 1 })
        .limit(limit);
      return await cursor.toArray();
    } catch {
      return [];
    }
  },
};
