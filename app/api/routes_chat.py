"""
The chat box, and the health check the sync workflow leans on.

/chat is a plain blocking request: you ask, a spinner turns, an answer lands in
a div. Not streamed. Streaming is a later polish pass and it is not worth an
SSE endpoint to save two seconds.
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import text

from app.api.routes_events import get_current_user
from app.core.config import CHAT_DAILY_LIMIT
from app.db import database as db
from app.db.database import engine
from app.rag.answer import answer_question
from app.rag.retriever import CAMPUS_TZ

log = logging.getLogger("chat")

router = APIRouter()
templates = Jinja2Templates(directory="templates")

OVER_LIMIT = (
    f"You've used all {CHAT_DAILY_LIMIT} of today's questions. "
    "The counter resets at midnight."
)

BROKEN = (
    "Something went wrong reaching the assistant. Try again in a minute."
)


def _for_display(match):
    """Flatten a match into what the template needs, with the time already in
    campus time -- Jinja is the wrong place to be doing timezone maths."""
    event = match.event
    starts_at = event.get("starts_at")
    when = None
    if starts_at:
        local = starts_at.astimezone(CAMPUS_TZ)
        # Built by hand rather than with %-I, which is a glibc extension and
        # not portable.
        when = f"{local:%a %b %d}, {local.hour % 12 or 12}:{local:%M %p}"

    return {
        "event_id": event.get("event_id"),
        "title": event.get("title"),
        "location": event.get("location"),
        "when": when,
    }


# =============================================================================
# HEALTH
# =============================================================================


@router.get("/healthz")
async def healthz():
    """Cheap enough to hit every day, honest enough to be worth hitting.

    A bare 200 would stay green with the database on fire, and this is the only
    uptime monitoring there is, so it costs one round trip to find out.
    """
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception:
        log.exception("healthz: database unreachable")
        return JSONResponse({"status": "error", "database": False},
                            status_code=503)

    return {"status": "ok", "database": True}


# =============================================================================
# CHAT PAGE
# =============================================================================


@router.get("/chat", response_class=HTMLResponse)
async def chat_page(
    request: Request,
    current_user: Optional[dict] = Depends(get_current_user),
    token: Optional[str] = Query(None),
):
    if not current_user:
        return RedirectResponse("/login", status_code=303)

    today = datetime.now(CAMPUS_TZ).date()
    used = db.get_chat_request_count(current_user["user_id"], today)

    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "user": current_user,
            "token": token,
            "used": used,
            "limit": CHAT_DAILY_LIMIT,
        },
    )


# =============================================================================
# ASK
# =============================================================================


@router.post("/chat", response_class=HTMLResponse)
async def chat(
    request: Request,
    question: str = Form(...),
    current_user: Optional[dict] = Depends(get_current_user),
    token: Optional[str] = Query(None),
):
    if not current_user:
        return templates.TemplateResponse(
            "fragments/chat_reply.html",
            {
                "request": request,
                "error": "Log in to ask a question.",
                "used": 0,
                "limit": CHAT_DAILY_LIMIT,
            },
            status_code=401,
        )

    question = question.strip()
    if not question:
        return templates.TemplateResponse(
            "fragments/chat_reply.html",
            {
                "request": request,
                "error": "Ask me something first.",
                "used": db.get_chat_request_count(
                    current_user["user_id"], datetime.now(CAMPUS_TZ).date()
                ),
                "limit": CHAT_DAILY_LIMIT,
            },
        )

    # Counted before the answer, so a slow or failed model still costs a turn.
    # Otherwise the cap is trivially avoidable by asking things that break it.
    today = datetime.now(CAMPUS_TZ).date()
    used = db.record_chat_request(current_user["user_id"], today)

    if used > CHAT_DAILY_LIMIT:
        return templates.TemplateResponse(
            "fragments/chat_reply.html",
            {
                "request": request,
                "error": OVER_LIMIT,
                "used": CHAT_DAILY_LIMIT,
                "limit": CHAT_DAILY_LIMIT,
            },
            status_code=429,
        )

    try:
        answer = answer_question(question)
    except Exception:
        log.exception("chat: answering %r failed", question[:100])
        return templates.TemplateResponse(
            "fragments/chat_reply.html",
            {
                "request": request,
                "error": BROKEN,
                "used": used,
                "limit": CHAT_DAILY_LIMIT,
            },
            status_code=502,
        )

    return templates.TemplateResponse(
        "fragments/chat_reply.html",
        {
            "request": request,
            "question": question,
            "answer": answer,
            "events": [_for_display(match) for match in answer.matches],
            "used": used,
            "limit": CHAT_DAILY_LIMIT,
            "token": token,
        },
    )
