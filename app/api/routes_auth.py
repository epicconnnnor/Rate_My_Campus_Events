"""
Authentication Routes

This module defines all authentication-related routes for RateMyCampusEvents v0.1.0.

"""

from datetime import timedelta
from typing import Optional

import logging

from app.core.oauth import (GITHUB, GOOGLE, PROVIDERS, configured_providers,
                            is_configured, link_or_create_user, oauth,
                            profile_from_github, profile_from_google)
from app.core.config import COOKIE_SECURE
from app.core.security import (ACCESS_TOKEN_EXPIRE_MINUTES, authenticate_user,
                               create_access_token, hash_password)
from app.db import database as db
from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.templating import Jinja2Templates

log = logging.getLogger("auth")

router = APIRouter()
templates = Jinja2Templates(directory="templates")
AUTH_COOKIE = "access_token"


def login_redirect(token: str) -> RedirectResponse:
    """Start a browser session without putting its bearer credential in a URL."""
    response = RedirectResponse("/", status_code=303)
    response.set_cookie(
        AUTH_COOKIE,
        token,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        path="/",
    )
    return response


# =============================================================================
# HTML LOGIN PAGE
# =============================================================================

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: Optional[str] = None):
    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "error": error,
            "providers": configured_providers(),
        },
    )


# =============================================================================
# HTML LOGIN SUBMIT
# =============================================================================

@router.post("/login")
async def login_html(
    email: str = Form(...),
    password: str = Form(...),
):
    user = authenticate_user(email, password, db.get_user_by_email)
    if not user:
        return RedirectResponse("/login?error=invalid", status_code=303)

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    token = create_access_token(
        data={"sub": user["email"]},
        expires_delta=access_token_expires,
    )

    return login_redirect(token)


# =============================================================================
# OAuth2
# =============================================================================

@router.post("/token")
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
):
    email = form_data.username

    user = authenticate_user(email, form_data.password, db.get_user_by_email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    token = create_access_token(
        data={"sub": user["email"]},
        expires_delta=access_token_expires,
    )

    return {
        "access_token": token,
        "token_type": "bearer",
    }


# =============================================================================
# HTML REGISTER PAGE
# =============================================================================

@router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request, error: Optional[str] = None):
    return templates.TemplateResponse(
        "register.html",
        {
            "request": request,
            "error": error,
            "providers": configured_providers(),
        },
    )


# =============================================================================
# HTML REGISTER SUBMIT
# =============================================================================

@router.post("/register")
async def register(
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form("student"),
):
    if db.get_user_by_email(email) is not None:
        return RedirectResponse("/register?error=exists", status_code=303)

    user = db.create_user(
        {
            "name": name,
            "email": email,
            "password_hash": hash_password(password),
        }
    )

    # Auto-login
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    token = create_access_token(
        data={"sub": email},
        expires_delta=access_token_expires,
    )

    return login_redirect(token)


# =============================================================================
# LOGOUT
# =============================================================================

@router.get("/logout")
async def logout():
    response = RedirectResponse("/", status_code=303)
    response.delete_cookie(AUTH_COOKIE, path="/")
    return response


# =============================================================================
# SIGN IN WITH A PROVIDER
# =============================================================================

@router.get("/auth/{provider}")
async def oauth_start(provider: str, request: Request):
    """Hand off to Google or GitHub.

    Authlib puts the state in the session cookie and checks it on the way back,
    which is what stops someone else's callback being replayed at us.
    """
    if provider not in PROVIDERS or not is_configured(provider):
        return RedirectResponse("/login?error=provider", status_code=303)

    client = oauth.create_client(provider)
    redirect_uri = request.url_for("oauth_callback", provider=provider)
    return await client.authorize_redirect(request, str(redirect_uri))


@router.get("/auth/{provider}/callback", name="oauth_callback")
async def oauth_callback(provider: str, request: Request):
    if provider not in PROVIDERS or not is_configured(provider):
        return RedirectResponse("/login?error=provider", status_code=303)

    client = oauth.create_client(provider)

    try:
        token = await client.authorize_access_token(request)
    except Exception:
        log.exception("%s: could not exchange the code", provider)
        return RedirectResponse("/login?error=oauth", status_code=303)

    try:
        if provider == GOOGLE:
            profile = profile_from_google(token.get("userinfo"))
        else:
            user = (await client.get("user", token=token)).json()
            emails = (await client.get("user/emails", token=token)).json()
            profile = profile_from_github(user, emails)
    except Exception:
        log.exception("%s: could not read the profile", provider)
        return RedirectResponse("/login?error=oauth", status_code=303)

    if profile is None:
        # Almost always an unverified address, which we will not link on.
        return RedirectResponse("/login?error=email", status_code=303)

    account = link_or_create_user(
        profile,
        get_user_by_email=db.get_user_by_email,
        create_user=db.create_user,
    )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    jwt_token = create_access_token(
        data={"sub": account["email"]},
        expires_delta=access_token_expires,
    )

    return login_redirect(jwt_token)
