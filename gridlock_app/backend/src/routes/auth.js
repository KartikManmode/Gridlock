import express from 'express';
import { AuthService } from '../services/authService.js';

const router = express.Router();

router.post('/register', async (req, res) => {
  try {
    const user = await AuthService.registerUser(req.body);
    return res.status(201).json(user);
  } catch (err) {
    return res.status(400).json({ detail: err.message });
  }
});

router.post('/login', async (req, res) => {
  try {
    const data = await AuthService.loginUser(req.body);
    return res.json(data);
  } catch (err) {
    return res.status(401).json({ detail: err.message });
  }
});

router.get('/me', (req, res) => {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ detail: 'Missing or invalid Bearer token.' });
  }
  const token = authHeader.split(' ')[1];
  const payload = AuthService.verifyToken(token);
  if (!payload) {
    return res.status(401).json({ detail: 'Session expired or invalid.' });
  }
  return res.json(payload);
});

export default router;
