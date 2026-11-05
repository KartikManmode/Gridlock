"""Process 1.0: Authentication & RBAC Models.

Data Transfer Objects and Enums for User Authentication and Role-Based Access Control.
Corresponds to D1: users Table.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class UserRole(str, Enum):
    ADMIN = "admin"
    OPERATOR = "operator"
    CITIZEN = "citizen"


class UserRegisterRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=6)
    role: UserRole = UserRole.CITIZEN


class UserLoginRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=255)
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: UserRole
    user_id: int
    name: str
    permissions: list[str]


class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    role: UserRole
    created_at: str
