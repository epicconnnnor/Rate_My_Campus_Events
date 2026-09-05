"""
This module defines all event-related routes for RateMyCampusEvents.
"""

from datetime import date, datetime
from typing import Optional

import logging

from app.core.security import get_user_from_session
from app.db import database as db
from app.db.database import engine
from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

log = logging.getLogger("events")

router = APIRouter()
templates = Jinja2Templates(directory="templates")


# =============================================================================
# AUTH HELPERS
# =============================================================================


async def get_current_user(
    token: Optional[str] = Query(None),
) -> Optional[dict]:
    if not token:
        return None

    user = get_user_from_session(token, db.get_user_by_email)
    return user


# =============================================================================
# LANDING PAGE
# =============================================================================


def when_to_show(event: dict) -> Optional[str]:
    """When an event says it is on, in campus time.

    Two fields, for two kinds of event. Anything imported from the calendar
    has a real timestamp in starts_at; date_time is the free-text field a
    person types when they create one by hand, and it is NULL on every
    imported row -- which is why these pages had been printing the word "None"
    once per event.
    """
    from app.rag.retriever import CAMPUS_TZ

    starts_at = event.get("starts_at")
    if starts_at:
        local = starts_at.astimezone(CAMPUS_TZ)
        # Built by hand rather than with %-I, which is a glibc extension.
        return f"{local:%a %b %d}, {local.hour % 12 or 12}:{local:%M %p}"
    return event.get("date_time") or None


def _landing_stats() -> dict:
    """Real numbers for the front page, read from the calendar we actually
    hold.

    Written down nowhere: a landing page quoting a number that was true once is
    worse than a landing page quoting none. If the database is not up, the
    counts come back None and the template leaves those blocks out rather than
    showing a confident zero.
    """
    from sqlalchemy import text

    try:
        with engine.connect() as connection:
            events = connection.execute(
                text("SELECT count(*) FROM events")
            ).scalar_one()
            categories = connection.execute(text(
                "SELECT count(DISTINCT category) "
                "FROM events, unnest(event_types) AS category"
            )).scalar_one()
        return {"events": events, "categories": categories}
    except Exception:
        log.exception("landing: could not read the counts")
        return {"events": None, "categories": None}


@router.get("/", response_class=HTMLResponse)
async def landing(
    request: Request,
    current_user: Optional[dict] = Depends(get_current_user),
    token: Optional[str] = Query(None),
):
    """The first screen, and the only one that has to work signed out.

    Nothing here needs a session. current_user only decides whether the
    call to action says "Get started" or "Open the chat".
    """
    return templates.TemplateResponse(
        "landing.html",
        {
            "request": request,
            "user": current_user,
            "token": token,
            "stats": _landing_stats(),
        },
    )


# =============================================================================
# EVENT LIST
# =============================================================================


@router.get("/events", response_class=HTMLResponse)
async def events_index(
    request: Request,
    current_user: Optional[dict] = Depends(get_current_user),
    token: Optional[str] = Query(None),
    q: str = Query("", max_length=200),
    category: str = Query("", max_length=200),
    from_date: str = Query("", max_length=10),
    free: bool = Query(False),
):
    try:
        first_date = date.fromisoformat(from_date) if from_date else None
    except ValueError:
        raise HTTPException(status_code=422, detail="Use YYYY-MM-DD for the date")
    all_events = db.list_events()
    categories = sorted({kind for event in all_events
                         for kind in (event.get("event_types") or [])})
    query = q.strip().casefold()
    all_events = [event for event in all_events
                  if (not category or category in (event.get("event_types") or []))
                  and (not free or event.get("is_free") is True)
                  and (not query or query in " ".join(
                      str(event.get(field) or "")
                      for field in ("title", "description", "location")
                  ).casefold())]

    from app.rag.retriever import CAMPUS_TZ

    for event in all_events:
        event["when"] = when_to_show(event)

        starts = event.get("starts_at")
        if not starts and event.get("date_time"):
            try:
                starts = datetime.fromisoformat(event["date_time"])
            except ValueError:
                pass
        if starts:
            if starts.tzinfo is None:
                starts = starts.replace(tzinfo=CAMPUS_TZ)
            starts = starts.astimezone(CAMPUS_TZ)
        event["display_start"] = starts
        event["month_label"] = starts.strftime("%B %Y") if starts else "Dates to be announced"

    if first_date:
        all_events = [event for event in all_events
                      if event["display_start"] and event["display_start"].date() >= first_date]

    all_events.sort(key=lambda event: (
        event["display_start"] is None,
        event["display_start"].timestamp() if event["display_start"] else 0,
        event["title"].casefold(),
    ))

    # Calculate ratings
    for event in all_events:
        event_reactions = db.get_reactions_for_event(event["event_id"])
        if event_reactions:
            avg = sum(r["value"]
                      for r in event_reactions) / len(event_reactions)
            event["avg_rating"] = round(avg, 1)
            event["total_reactions"] = len(event_reactions)
        else:
            event["avg_rating"] = 0
            event["total_reactions"] = 0

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "user": current_user,
            "events": all_events,
            "categories": categories,
            "q": q,
            "category": category,
            "from_date": from_date,
            "free": free,
            "token": token,
        },
    )


# =============================================================================
# CREATE EVENT PAGE
# =============================================================================


@router.get("/event/create", response_class=HTMLResponse)
async def create_event_page(
    request: Request,
    current_user: Optional[dict] = Depends(get_current_user),
    token: Optional[str] = Query(None),
):
    if not current_user:
        return RedirectResponse("/login", status_code=303)

    return templates.TemplateResponse(
        "create_event.html",
        {
            "request": request,
            "user": current_user,
            "token": token,
        },
    )


# =============================================================================
# CREATE EVENT SUBMIT
# =============================================================================


@router.post("/event/create")
async def create_event(
    current_user: Optional[dict] = Depends(get_current_user),
    token: Optional[str] = Query(None),
    title: str = Form(...),
    description: str = Form(...),
    date_time: str = Form(...),
    location: str = Form(...),
):
    if not current_user:
        return RedirectResponse("/login", status_code=303)

    event = db.create_event(
        {
            "title": title,
            "description": description,
            "date_time": date_time,
            "location": location,
            "created_by": current_user["user_id"],
        }
    )

    if token:
        return RedirectResponse(f"/?token={token}", status_code=303)
    return RedirectResponse("/events", status_code=303)


# =============================================================================
# EVENT DETAIL (PUBLIC)
# =============================================================================


@router.get("/event/{event_id}", response_class=HTMLResponse)
async def event_detail(
    request: Request,
    event_id: int,
    current_user: Optional[dict] = Depends(get_current_user),
    token: Optional[str] = Query(None),
):
    event = db.get_event(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    event["when"] = when_to_show(event)

    creator = db.get_user_by_id(event["created_by"])

    event_reactions = db.get_reactions_for_event(event_id)
    if event_reactions:
        avg_rating = sum(r["value"]
                         for r in event_reactions) / len(event_reactions)
        event["avg_rating"] = round(avg_rating, 1)
        event["total_reactions"] = len(event_reactions)

        rating_counts = {1: 0, -1: 0}
        for r in event_reactions:
            rating_counts[r["value"]] = rating_counts.get(r["value"], 0) + 1
        event["thumbs_up"] = rating_counts.get(1, 0)
        event["thumbs_down"] = rating_counts.get(-1, 0)
    else:
        event["avg_rating"] = 0
        event["total_reactions"] = 0
        event["thumbs_up"] = 0
        event["thumbs_down"] = 0

    user_reaction = None
    if current_user:
        user_reaction = next(
            (
                r
                for r in event_reactions
                if r.get("user_id") == current_user["user_id"]
            ),
            None,
        )

    event_comments = db.get_comments_for_event(event_id)

    for comment in event_comments:
        comment_user = db.get_user_by_id(comment.get("user_id"))
        comment["user_name"] = comment_user["name"] if comment_user else "Anonymous"

    return templates.TemplateResponse(
        "event.html",
        {
            "request": request,
            "user": current_user,
            "event": event,
            "creator": creator,
            "user_reaction": user_reaction,
            "comments": event_comments,
            "token": token,
        },
    )


# =============================================================================
# REACT TO EVENT
# =============================================================================


@router.post("/event/{event_id}/react")
async def react_to_event(
    event_id: int,
    value: int = Form(...),
    current_user: Optional[dict] = Depends(get_current_user),
    token: Optional[str] = Query(None),
):
    if not current_user:
        return RedirectResponse("/login", status_code=303)

    if not db.get_event(event_id):
        raise HTTPException(status_code=404, detail="Event not found")

    if value not in [1, -1]:
        raise HTTPException(status_code=400, detail="Value must be 1 or -1")

    db.upsert_reaction(event_id, current_user["user_id"], value)

    if token:
        return RedirectResponse(f"/event/{event_id}?token={token}", status_code=303)
    return RedirectResponse(f"/event/{event_id}", status_code=303)


# =============================================================================
# HTMX ENDPOINT: REACT TO EVENT
# =============================================================================


@router.post("/htmx/event/{event_id}/react", response_class=HTMLResponse)
async def htmx_react_to_event(
    request: Request,
    event_id: int,
    value: int = Form(...),
    current_user: Optional[dict] = Depends(get_current_user),
    token: Optional[str] = Query(None),
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    event = db.get_event(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    if value not in [1, -1]:
        raise HTTPException(status_code=400, detail="Value must be 1 or -1")

    db.upsert_reaction(event_id, current_user["user_id"], value)

    # Recalculate ratings
    event_reactions = db.get_reactions_for_event(event_id)
    if event_reactions:
        avg_rating = sum(r["value"]
                         for r in event_reactions) / len(event_reactions)
        event["avg_rating"] = round(avg_rating, 1)
        event["total_reactions"] = len(event_reactions)

        rating_counts = {1: 0, -1: 0}
        for r in event_reactions:
            rating_counts[r["value"]] = rating_counts.get(r["value"], 0) + 1
        event["thumbs_up"] = rating_counts.get(1, 0)
        event["thumbs_down"] = rating_counts.get(-1, 0)
    else:
        event["avg_rating"] = 0
        event["total_reactions"] = 0
        event["thumbs_up"] = 0
        event["thumbs_down"] = 0

    user_reaction = next(
        (r for r in event_reactions if r.get(
            "user_id") == current_user["user_id"]),
        None,
    )

    return templates.TemplateResponse(
        "fragments/rating_section.html",
        {
            "request": request,
            "event": event,
            "user": current_user,
            "user_reaction": user_reaction,
            "token": token,
        },
    )


# =============================================================================
# ADD COMMENT
# =============================================================================


@router.post("/event/{event_id}/comment")
async def add_comment(
    event_id: int,
    text: str = Form(...),
    current_user: Optional[dict] = Depends(get_current_user),
    token: Optional[str] = Query(None),
):
    if not current_user:
        return RedirectResponse("/login", status_code=303)

    if not db.get_event(event_id):
        raise HTTPException(status_code=404, detail="Event not found")

    db.create_comment(event_id, current_user["user_id"], text)

    if token:
        return RedirectResponse(f"/event/{event_id}?token={token}", status_code=303)
    return RedirectResponse(f"/event/{event_id}", status_code=303)


@router.post("/htmx/event/{event_id}/comment", response_class=HTMLResponse)
async def htmx_add_comment(
    request: Request,
    event_id: int,
    text: str = Form(...),
    current_user: Optional[dict] = Depends(get_current_user),
    token: Optional[str] = Query(None),
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not db.get_event(event_id):
        raise HTTPException(status_code=404, detail="Event not found")

    db.create_comment(event_id, current_user["user_id"], text)

    # Get all comments for this event
    event_comments = db.get_comments_for_event(event_id)

    for comment in event_comments:
        comment_user = db.get_user_by_id(comment.get("user_id"))
        comment["user_name"] = comment_user["name"] if comment_user else "Anonymous"

    return templates.TemplateResponse(
        "fragments/comments_list.html",
        {
            "request": request,
            "comments": event_comments,
            "user": current_user,
            "token": token,
        },
    )


# =============================================================================
# DELETE COMMENT
# =============================================================================


@router.delete("/htmx/comment/{comment_id}", response_class=HTMLResponse)
async def htmx_delete_comment(
    request: Request,
    comment_id: int,
    current_user: Optional[dict] = Depends(get_current_user),
    token: Optional[str] = Query(None),
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Get the comment to check ownership
    # Only allow the comment author to delete
    comment = db.get_comment(comment_id)
    if not comment:
        raise HTTPException(status_code=404, detail="Comment not found")

    if comment["user_id"] != current_user["user_id"]:
        raise HTTPException(
            status_code=403, detail="Not authorized to delete this comment")

    db.delete_comment(comment_id)

    event_comments = db.get_comments_for_event(comment["event_id"])

    for comment in event_comments:
        comment_user = db.get_user_by_id(comment.get("user_id"))
        comment["user_name"] = comment_user["name"] if comment_user else "Anonymous"

    return templates.TemplateResponse(
        "fragments/comments_list.html",
        {
            "request": request,
            "comments": event_comments,
            "user": current_user,
            "token": token,
        },
    )
