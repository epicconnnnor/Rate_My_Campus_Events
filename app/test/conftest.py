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


# Which variable has to be set depends on who is answering. Naming the wrong
# one is how a secret called GEMINIAPI produced a green tick that had judged
# nothing; naming none of them would do it again the first time somebody
# switched provider.
KEY_FOR_PROVIDER = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def _required_key() -> str:
    """The variable this run needs, or "" if no real model is selected."""
    provider = os.getenv("RAG_PROVIDER") or "openai"
    return KEY_FOR_PROVIDER.get(provider, "")


def _have_live_model() -> bool:
    variable = _required_key()
    return bool(variable) and bool(os.getenv(variable))


def _key_is_required() -> bool:
    """Whether a missing key should fail rather than skip.

    Set by CI. A pull request from a fork cannot read the repository's secrets,
    so it skips; a push or a pull request from a branch in this repository has
    no excuse.
    """
    return os.getenv("EVAL_REQUIRE_LIVE_MODEL", "").strip().lower() == "true"


@pytest.fixture(scope="session")
def live_model():
    """Gate for the judged tests.

    A skip here used to be indistinguishable from a pass, which is how a secret
    named GEMINIAPI instead of GEMINI_API_KEY produced a green tick that had
    judged nothing. Where a key is meant to exist, its absence now fails.
    """
    if _have_live_model():
        return True

    if _key_is_required():
        variable = _required_key() or "an API key"
        pytest.fail(
            f"{variable} is unset, or RAG_PROVIDER names no real model, so the "
            "judge cannot run -- and this is a push or a same-repo pull "
            "request, where a key is expected. Skipping here would be a green "
            "tick certifying nothing. Check that the repository secret is "
            f"named exactly {variable}, matching .github/workflows/eval.yml.",
            pytrace=False,
        )

    pytest.skip(
        f"no {_required_key() or 'API key'}: expected on a fork's pull "
        "request, which cannot read repository secrets"
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
