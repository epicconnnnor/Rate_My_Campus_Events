"""
Shared fixtures for the evaluation suite.

The eval runs the real pipeline -- real embeddings, real retrieval, real
answers -- against a frozen slice of the calendar, so it needs a live database
with pgvector and a real API key. Without either it skips rather than pretends,
because an eval that passes on stub vectors is measuring nothing.

The unit tests under app/test do not touch any of this; they never ask for
these fixtures.
"""

import json
import os
import pathlib
from datetime import date, datetime

import pytest

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
EVENTS_FILE = FIXTURES / "events.json"
QUESTIONS_FILE = FIXTURES / "golden_questions.json"

# The frozen events run from 2026-09-02 to 2026-11-12. Every question is asked
# as if it were the first of those days, so "this weekend" means one fixed
# weekend forever and the answers stay comparable between runs.
EVAL_TODAY = date(2026, 9, 2)


def _live_key() -> bool:
    return bool(os.getenv("GEMINI_API_KEY")) and os.getenv(
        "RAG_PROVIDER", "gemini"
    ) != "fake"


requires_live_model = pytest.mark.skipif(
    not _live_key(),
    reason="needs GEMINI_API_KEY and a non-fake RAG_PROVIDER; "
           "an eval on stub vectors measures nothing",
)


@pytest.fixture(scope="session")
def frozen_events():
    """The ~100 events the golden questions are written against."""
    with EVENTS_FILE.open(encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="session")
def golden_questions():
    """Connor's questions, written against the frozen events above."""
    if not QUESTIONS_FILE.exists():
        return []
    with QUESTIONS_FILE.open(encoding="utf-8") as handle:
        return json.load(handle).get("questions", [])


@pytest.fixture(scope="session")
def eval_db(frozen_events):
    """Load the frozen events into the database and embed them.

    Replaces whatever is in `events` -- point DATABASE_NAME at a throwaway
    database, never at anything you care about.
    """
    from sqlmodel import Session

    from app.db.database import engine
    from app.models.event import Event, EventEmbedding
    from app.rag.indexer import index_events

    def parse(value):
        return datetime.fromisoformat(value) if value else None

    with Session(engine) as session:
        session.query(EventEmbedding).delete()
        session.query(Event).filter(Event.source == "localist").delete()
        session.commit()

        for row in frozen_events:
            session.add(Event(
                source="localist",
                created_by=None,
                external_id=row["external_id"],
                title=row["title"],
                description=row["description"],
                location=row["location"],
                starts_at=parse(row["starts_at"]),
                ends_at=parse(row["ends_at"]),
                status=row["status"],
                is_free=row["is_free"],
                experience=row["experience"],
                keywords=row["keywords"],
                groups=row["groups"],
                event_types=row["event_types"],
                localist_url=row["localist_url"],
                external_updated_at=parse(row["external_updated_at"]),
            ))
        session.commit()

    index_events()
    return frozen_events
