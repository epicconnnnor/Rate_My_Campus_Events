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
from app.core.config import CHAT_DAILY_LIMIT, DEMO_DATE
from app.db import database as db
from app.db.database import engine
from app.rag.answer import answer_question
from app.rag.providers import ProviderNotConfigured
from app.rag.retriever import (CAMPUS_TZ, FilterOverrides,
                                current_campus_date)

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

# A misconfiguration is not weather, and telling somebody to try again in a
# minute is advice that cannot work: it will fail identically forever until a
# person changes a setting. So the actual reason is shown.
#
# What gets shown is the exception's own message, which is written for this --
# the name of the variable that is missing and what to set it to. No key, no
# path, no traceback: nothing here is worth hiding, and nothing here is a
# secret. The generic message stays for everything else, where the reason is
# genuinely a stack trace nobody outside the log wants.
MISCONFIGURED_PREFIX = "This install is not configured to answer questions yet."


# How many category chips to offer. Enough to be useful, few enough that the
# row does not wrap into a wall above the input.
CATEGORY_CHIPS = 6


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

    groups = event.get("groups") or []

    return {
        "event_id": event.get("event_id"),
        "title": event.get("title"),
        "location": event.get("location"),
        "when": when,
        # One organizer on the card. The feed lists co-hosts and a card is not
        # the place to read all of them.
        "organizer": groups[0] if groups else None,
        "is_free": event.get("is_free"),
        "experience": event.get("experience"),
    }


def _categories(limit=CATEGORY_CHIPS):
    """The calendar's own most common event types.

    Read from the data rather than written down here, because a chip for a
    category nothing is filed under is a chip that always returns nothing.
    Empty on a database that is not up yet -- the page still works, it just
    offers fewer chips.
    """
    # unnest belongs in FROM, not in the select list. A set-returning function
    # in the target list is expanded after grouping, so the obvious
    # "SELECT unnest(...) ... GROUP BY" does not mean what it reads like.
    statement = text(
        "SELECT category, count(*) AS total "
        "FROM events, unnest(event_types) AS category "
        "GROUP BY category ORDER BY total DESC, category LIMIT :limit"
    )
    try:
        with engine.connect() as connection:
            return [row.category
                    for row in connection.execute(statement, {"limit": limit})]
    except Exception:
        log.exception("chat: could not read the category list")
        return []


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
            "categories": _categories(),
            # Shown in the page when it is set, so a demo that is standing in
            # September 2026 says so rather than quietly lying about "today".
            "demo_date": DEMO_DATE,
        },
    )


# =============================================================================
# ASK
# =============================================================================


@router.post("/chat", response_class=HTMLResponse)
async def chat(
    request: Request,
    question: str = Form(...),
    # The chips. Absent means "not asked for", which is not the same as false:
    # an unticked "free" must not turn into a search for paid events only.
    free: Optional[str] = Form(None),
    experience: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
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

    overrides = FilterOverrides.from_form(
        free=True if free else None,
        experience=experience,
        category=category,
    )

    try:
        answer = answer_question(question, overrides=overrides)
    except ProviderNotConfigured as error:
        # Logged as an error rather than an exception: the traceback is six
        # frames of constructor and the one useful line is the message.
        log.error("chat: not configured -- %s", error)
        return templates.TemplateResponse(
            "fragments/chat_reply.html",
            {
                "request": request,
                "error": MISCONFIGURED_PREFIX,
                "detail": str(error),
                "used": used,
                "limit": CHAT_DAILY_LIMIT,
            },
            status_code=503,
        )
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
