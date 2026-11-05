import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Potential locations for .env
const candidates = [
  path.resolve(__dirname, '../../../.env'),      // gridlock_app/.env
  path.resolve(__dirname, '../../../../.env'),   // gridlock/.env
  path.resolve(process.cwd(), '.env'),           // Current working directory .env
];

let loadedPath = null;
for (const envPath of candidates) {
  if (fs.existsSync(envPath)) {
    dotenv.config({ path: envPath });
    loadedPath = envPath;
    break;
  }
}

if (!loadedPath) {
  dotenv.config(); // fallback to default dotenv lookup
}

export const config = {
  PORT: process.env.PORT || 8000,
  DATABASE_URL: process.env.DATABASE_URL || '',
  MONGODB_URI: process.env.MONGODB_URI || '',
  JWT_SECRET: process.env.JWT_SECRET || 'gridlock-secret-traffic-forecasting-token-key-2026',
  SALT: process.env.SALT || 'gridlock_salt_secure_2026',
};
