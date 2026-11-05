import crypto from 'crypto';
import jwt from 'jsonwebtoken';
import { db } from '../config/postgres.js';
import { config } from '../config/env.js';

export const UserRole = {
  ADMIN: 'admin',
  OPERATOR: 'operator',
  CITIZEN: 'citizen',
};

export const ROLE_PERMISSIONS = {
  [UserRole.ADMIN]: [
    'users:manage',
    'zones:manage',
    'sensors:stream',
    'forecast:read',
    'forecast:generate',
    'incidents:report',
    'dispatch:manage',
    'analytics:read',
    'admin:all',
  ],
  [UserRole.OPERATOR]: [
    'zones:read',
    'sensors:stream',
    'forecast:read',
    'forecast:generate',
    'incidents:report',
    'dispatch:manage',
    'analytics:read',
  ],
  [UserRole.CITIZEN]: [
    'forecast:read',
    'incidents:report',
    'analytics:read',
  ],
};

export class AuthService {
  static hashPassword(password) {
    return crypto
      .pbkdf2Sync(password, config.SALT, 100000, 32, 'sha256')
      .toString('hex');
  }

  static verifyPassword(password, hashed) {
    const computed = AuthService.hashPassword(password);
    return crypto.timingSafeEqual(Buffer.from(computed), Buffer.from(hashed));
  }

  static generateToken(userId, role, email) {
    return jwt.sign(
      {
        sub: userId,
        role,
        email,
      },
      config.JWT_SECRET,
      { expiresIn: '24h' }
    );
  }

  static verifyToken(token) {
    try {
      return jwt.verify(token, config.JWT_SECRET);
    } catch {
      return null;
    }
  }

  static async registerUser({ name, email, password, role = UserRole.CITIZEN }) {
    const existing = await db.fetchOne('SELECT id FROM users WHERE email = $1', [email]);
    if (existing) {
      throw new Error(`Email ${email} is already registered.`);
    }

    const validRole = Object.values(UserRole).includes(role) ? role : UserRole.CITIZEN;
    const passwordHash = AuthService.hashPassword(password);

    const inserted = await db.fetchOne(
      'INSERT INTO users (name, email, password_hash, role) VALUES ($1, $2, $3, $4) RETURNING id, name, email, role, created_at',
      [name, email, passwordHash, validRole]
    );
    return inserted;
  }

  static async loginUser({ email, password }) {
    const user = await db.fetchOne('SELECT * FROM users WHERE email = $1', [email]);
    if (!user || !AuthService.verifyPassword(password, user.password_hash)) {
      throw new Error('Invalid email or password.');
    }

    const token = AuthService.generateToken(user.id, user.role, user.email);
    const permissions = ROLE_PERMISSIONS[user.role] || [];

    return {
      access_token: token,
      token_type: 'bearer',
      role: user.role,
      permissions,
    };
  }

  static checkPermission(token, requiredPermission) {
    const payload = AuthService.verifyToken(token);
    if (!payload) {
      throw new Error('Unauthorized or invalid token.');
    }
    const perms = ROLE_PERMISSIONS[payload.role] || [];
    if (!perms.includes(requiredPermission) && !perms.includes('admin:all')) {
      throw new Error(`Forbidden: missing required permission '${requiredPermission}'.`);
    }
    return payload;
  }
}
