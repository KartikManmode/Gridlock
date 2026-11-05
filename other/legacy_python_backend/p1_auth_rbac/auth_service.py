"""Process 1.0: Authentication & RBAC Service.

Implements:
- User registration and password hashing (PBKDF2 SHA-256)
- User credential verification against D1 (users Table)
- JWT-style signed token issuance and RBAC permission checks
- UserEntity interaction handling
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any, Dict, List, Optional
from backend.database.connection import db
from backend.p1_auth_rbac.models import UserRegisterRequest, UserLoginRequest, TokenResponse, UserRole

SECRET_KEY = "gridlock-secret-traffic-forecasting-token-key-2026"
SALT = "gridlock_salt_secure_2026"

ROLE_PERMISSIONS: Dict[UserRole, List[str]] = {
    UserRole.ADMIN: [
        "users:manage", "zones:manage", "sensors:stream",
        "forecast:read", "forecast:generate",
        "incidents:report", "dispatch:manage",
        "analytics:read", "admin:all"
    ],
    UserRole.OPERATOR: [
        "zones:read", "sensors:stream",
        "forecast:read", "forecast:generate",
        "incidents:report", "dispatch:manage",
        "analytics:read"
    ],
    UserRole.CITIZEN: [
        "forecast:read",
        "incidents:report",
        "analytics:read"
    ],
}


class AuthService:
    """Authentication and Role-Based Access Control logic."""

    @staticmethod
    def hash_password(password: str) -> str:
        return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), SALT.encode("utf-8"), 100000).hex()

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        return hmac.compare_digest(AuthService.hash_password(password), hashed)

    @staticmethod
    def generate_token(user_id: int, role: str, email: str) -> str:
        payload = {
            "sub": user_id,
            "role": role,
            "email": email,
            "exp": int(time.time()) + 86400  # 24 hours
        }
        raw_json = json.dumps(payload, separators=(',', ':'))
        b64_payload = base64.urlsafe_b64encode(raw_json.encode()).decode().rstrip("=")
        signature = hmac.new(SECRET_KEY.encode(), b64_payload.encode(), hashlib.sha256).hexdigest()
        return f"{b64_payload}.{signature}"

    @staticmethod
    def verify_token(token: str) -> Optional[Dict[str, Any]]:
        try:
            parts = token.split(".")
            if len(parts) != 2:
                return None
            b64_payload, signature = parts
            expected_sig = hmac.new(SECRET_KEY.encode(), b64_payload.encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(signature, expected_sig):
                return None
            # Fix base64 padding
            rem = len(b64_payload) % 4
            if rem > 0:
                b64_payload += "=" * (4 - rem)
            payload = json.loads(base64.urlsafe_b64decode(b64_payload).decode())
            if payload.get("exp", 0) < time.time():
                return None
            return payload
        except Exception:
            return None

    @classmethod
    def register_user(cls, req: UserRegisterRequest) -> Dict[str, Any]:
        existing = db.fetch_one("SELECT id FROM users WHERE email = ?", (req.email,))
        if existing:
            raise ValueError(f"User with email '{req.email}' already exists.")
        
        pwd_hash = cls.hash_password(req.password)
        user_id = db.execute(
            "INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, ?)",
            (req.name, req.email, pwd_hash, req.role.value)
        )
        return db.fetch_one("SELECT id, name, email, role, created_at FROM users WHERE id = ?", (user_id,))

    @classmethod
    def login_user(cls, req: UserLoginRequest) -> TokenResponse:
        user = db.fetch_one("SELECT * FROM users WHERE email = ?", (req.email,))
        if not user or not cls.verify_password(req.password, user["password_hash"]):
            raise ValueError("Invalid email or password.")
        
        role = UserRole(user["role"])
        token = cls.generate_token(user["id"], user["role"], user["email"])
        permissions = ROLE_PERMISSIONS.get(role, [])

        return TokenResponse(
            access_token=token,
            token_type="bearer",
            role=role,
            user_id=user["id"],
            name=user["name"],
            permissions=permissions
        )

    @classmethod
    def check_permission(cls, token: str, required_permission: str) -> Dict[str, Any]:
        payload = cls.verify_token(token)
        if not payload:
            raise PermissionError("Invalid or expired session token.")
        role = UserRole(payload["role"])
        permissions = ROLE_PERMISSIONS.get(role, [])
        if "admin:all" in permissions or required_permission in permissions:
            return payload
        raise PermissionError(f"Role '{role.value}' lacks required permission: {required_permission}")
