"""API Routes for Process 1.0: Authentication & RBAC."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Header, status
from typing import Optional
from backend.p1_auth_rbac.models import UserRegisterRequest, UserLoginRequest, TokenResponse, UserResponse
from backend.p1_auth_rbac.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["1.0 Authentication & RBAC"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(req: UserRegisterRequest):
    try:
        user = AuthService.register_user(req)
        return user
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/login", response_model=TokenResponse)
def login(req: UserLoginRequest):
    try:
        return AuthService.login_user(req)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))


@router.get("/me")
def get_current_user_profile(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid Bearer token.")
    token = authorization.split(" ")[1]
    payload = AuthService.verify_token(token)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired or invalid.")
    return payload
