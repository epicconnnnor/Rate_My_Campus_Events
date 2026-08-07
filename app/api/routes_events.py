"""
This module defines all event-related routes for RateMyCampusEvents.
"""

from datetime import datetime
from typing import Optional

from app.core.security import get_user_from_session
from app.db import database as db
from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

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
# HOME PAGE
# =============================================================================


@router.get("/", response_class=HTMLResponse)
async def home(
    request: Request,
    current_user: Optional[dict] = Depends(get_current_user),
    token: Optional[str] = Query(None),
):
    all_events = db.list_events()

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
    return RedirectResponse("/", status_code=303)


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
