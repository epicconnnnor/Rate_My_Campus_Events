"""

Authentication and Security utilities for RateMyCampusEvents v0.1.0.

"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from app.core.config import ACCESS_TOKEN_EXPIRE_MINUTES, ALGORITHM, SECRET_KEY
from jose import ExpiredSignatureError, JWTError, jwt
from passlib.context import CryptContext

# =============================================================================
# PASSWORD HASHING
# =============================================================================

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


# =============================================================================
# USER AUTHENTICATION
# =============================================================================

def authenticate_user(email: str, password: str, get_user_by_email_func) -> Optional[dict]:
    """Authenticate user with email and password"""
    user = get_user_by_email_func(email)
    if not user:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return user


# =============================================================================
# JWT TOKEN OPERATIONS
# =============================================================================

def create_access_token(data: dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()

    expire = (
        datetime.now(timezone.utc) + expires_delta
        if expires_delta
        else datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    to_encode.update({"exp": expire})

    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> Optional[dict[str, Any]]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except (ExpiredSignatureError, JWTError):
        return None


# =============================================================================
# SESSION HELPERS
# =============================================================================

def get_email_from_token(token: str) -> Optional[str]:
    payload = decode_access_token(token)
    return payload.get("sub") if payload else None


def is_token_expired(token: str) -> bool:
    return decode_access_token(token) is None


def get_user_from_session(session: Optional[str], get_user_by_email_func) -> Optional[dict]:
    """Get user from session token"""
    if not session:
        return None

    email = get_email_from_token(session)
    if not email:
        return None

    return get_user_by_email_func(email)
