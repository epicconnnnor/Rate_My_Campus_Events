"""
Builds and stores the document that gets embedded for each event.

What goes in: title, description, organizing groups, event types, location.

What stays out: the date. Dates are handled in SQL, where they can be filtered
exactly. Putting "October 30" in the vector only pulls the event toward other
October events that have nothing to do with it.
"""

import logging
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Iterable, List, Optional

from sqlmodel import Session, select

from app.db.database import engine
from app.models.event import Event, EventEmbedding
from app.rag.providers import EmbeddingProvider, get_embedding_provider

log = logging.getLogger("indexer")

# Long enough that nothing in the UMass feed is currently touched (the longest
# description is ~3.4k), short enough to stay inside the embedding model's
# input limit if someone posts an essay.
MAX_DESCRIPTION_CHARS = 4000

# How many documents to send per embedding call.
BATCH_SIZE = 32

# Google's free tier allows 100 embed_content requests per minute, and it
# counts documents rather than HTTP calls -- one batch of 32 spends 32 of them.
# This is that published quota, not a number picked for feel. Exceeding it is a
# hard 429: the first eval run died on document 101 of 102.
FREE_TIER_DOCUMENTS_PER_MINUTE = 100

# Slack so a slow clock or a shared project cannot land us exactly on the line.
PACING_HEADROOM_SECONDS = 2


# =============================================================================
# DOCUMENT
# =============================================================================


def build_content(event: Event) -> str:
    """The text that represents this event in the vector space."""
    description = (event.description or "").strip()
    if len(description) > MAX_DESCRIPTION_CHARS:
        description = description[:MAX_DESCRIPTION_CHARS].rstrip()

    lines = [
        (event.title or "").strip(),
        description,
    ]
    if event.groups:
        lines.append("Organized by: " + ", ".join(event.groups))
    if event.event_types:
        lines.append("Event type: " + ", ".join(event.event_types))
    if event.location:
        lines.append("Location: " + event.location)

    return "\n".join(line for line in lines if line)


def _needs_embedding(event: Event, existing: Optional[EventEmbedding],
                     content: str) -> bool:
    """Whether this event is worth spending an embedding call on.

    The trigger the spec asks for is a change in external_updated_at. Comparing
    the rendered document as well catches the cases that misses: a user-created
    event, which has no external_updated_at at all, and any hand-edit.
    """
    if existing is None:
        return True
    if existing.source_updated_at != event.external_updated_at:
        return True
    return existing.content != content


class _Pacer:
    """Keeps embedded documents under the per-minute ceiling.

    The SDK has its own retry, and it exhausted itself against this quota
    rather than riding it out, so waiting has to happen before the request
    rather than after the refusal. This runs once per backfill, so simply
    sleeping out the rest of the minute is fine and beats anything cleverer.
    """

    def __init__(self, limit: int = FREE_TIER_DOCUMENTS_PER_MINUTE) -> None:
        self.limit = limit
        self._window_started = time.monotonic()
        self._spent = 0

    def reserve(self, documents: int) -> None:
        """Block until `documents` more can be sent without breaking the quota."""
        elapsed = time.monotonic() - self._window_started
        if elapsed >= 60:
            self._window_started = time.monotonic()
            self._spent = 0
            elapsed = 0

        if self._spent + documents > self.limit:
            wait = max(0.0, 60 - elapsed) + PACING_HEADROOM_SECONDS
            log.info(
                "%d/%d documents used this minute; waiting %.0fs for the quota "
                "window to roll", self._spent, self.limit, wait,
            )
            time.sleep(wait)
            self._window_started = time.monotonic()
            self._spent = 0

        self._spent += documents


def _store(session: Session, event: Event, content: str,
           vector: List[float]) -> None:
    existing = session.exec(
        select(EventEmbedding).where(EventEmbedding.event_id == event.event_id)
    ).first()

    if existing is None:
        existing = EventEmbedding(event_id=event.event_id)
        session.add(existing)

    existing.content = content
    existing.embedding = vector
    existing.source_updated_at = event.external_updated_at
    existing.embedded_at = datetime.now(timezone.utc)


# =============================================================================
# INDEXING
# =============================================================================


def rebuild_event_doc(event_id: int,
                      provider: Optional[EmbeddingProvider] = None) -> bool:
    """Re-embed one event if its document changed.

    Returns True if it was embedded, False if it was already up to date.
    """
    with Session(engine) as session:
        event = session.get(Event, event_id)
        if event is None:
            raise LookupError(f"no event with id {event_id}")

        content = build_content(event)
        existing = session.exec(
            select(EventEmbedding).where(EventEmbedding.event_id == event_id)
        ).first()

        if not _needs_embedding(event, existing, content):
            return False

        provider = provider or get_embedding_provider()
        vector = provider.embed_documents([content])[0]
        _store(session, event, content, vector)
        session.commit()
        return True


def index_events(event_ids: Optional[Iterable[int]] = None,
                 provider: Optional[EmbeddingProvider] = None) -> Counter:
    """Embed whatever is out of date, in batches.

    Pass event_ids to limit it; pass nothing to walk the whole table.
    """
    stats = Counter()
    provider = provider or get_embedding_provider()

    with Session(engine) as session:
        statement = select(Event)
        if event_ids is not None:
            statement = statement.where(Event.event_id.in_(list(event_ids)))

        pending = []
        pacer = _Pacer()

        def flush():
            if not pending:
                return
            texts = [item[1] for item in pending]
            # Not len(texts): a cached document is not a request, and pacing
            # for it means sleeping out a quota window for work that is not
            # going to happen.
            pacer.reserve(provider.chargeable(texts))
            vectors = provider.embed_documents(texts)
            for (event, content), vector in zip(pending, vectors):
                _store(session, event, content, vector)
            session.commit()
            pending.clear()

        for event in session.exec(statement):
            content = build_content(event)
            existing = session.exec(
                select(EventEmbedding).where(
                    EventEmbedding.event_id == event.event_id
                )
            ).first()

            if not _needs_embedding(event, existing, content):
                stats["skipped"] += 1
                continue

            stats["embedded"] += 1
            pending.append((event, content))
            if len(pending) >= BATCH_SIZE:
                flush()

        flush()

    return stats
