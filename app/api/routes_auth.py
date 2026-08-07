"""
Authentication Routes

This module defines all authentication-related routes for RateMyCampusEvents v0.1.0.

"""

from datetime import timedelta
from typing import Optional

from app.core.security import (ACCESS_TOKEN_EXPIRE_MINUTES, authenticate_user,
                               create_access_token, hash_password)
from app.db import database as db
from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="templates")


# =============================================================================
# HTML LOGIN PAGE
# =============================================================================

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


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

    return RedirectResponse(f"/?token={token}", status_code=303)


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
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


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

    return RedirectResponse(f"/?token={token}", status_code=303)


# =============================================================================
# LOGOUT
# =============================================================================

@router.get("/logout")
async def logout():
    return RedirectResponse("/", status_code=303)
