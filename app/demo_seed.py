"""
Fill a local database with the frozen events, so the chat UI has something to
answer with.

    python -m app.demo_seed

The alternative is `python -m app.ingest.localist`, which pulls the real UMass
calendar. That is the right thing for anything deployed and the wrong thing for
a first local run: it needs the network, it returns a different calendar every
week, and none of the golden questions mean anything against it.

These are the same 102 events the eval judges against -- app/test/fixtures --
which is why they live under a test directory and get loaded from here rather
than copied. They run from 2026-09-02 to 2026-11-12, so pair this with

    DEMO_DATE=2026-09-02

or every question searches a calendar the clock has already walked past.

Replaces the events it owns and nothing else: rows with source "localist" go,
anything a person created through the UI stays.
"""

import argparse
import json
import logging
import pathlib
from datetime import datetime
from typing import List, Optional

from sqlmodel import Session

from app.db.database import engine
from app.models.event import Event, EventEmbedding

log = logging.getLogger("demo_seed")

FIXTURE = (pathlib.Path(__file__).parent / "test" / "fixtures" / "events.json")


def _parse(value):
    return datetime.fromisoformat(value) if value else None


def load_events(path: pathlib.Path = FIXTURE) -> int:
    """Put the frozen events in the database. Returns how many."""
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)

    with Session(engine) as session:
        # The embeddings go first: they point at the events by id, and the
        # events are about to be replaced by rows with different ones.
        session.query(EventEmbedding).delete()
        session.query(Event).filter(Event.source == "localist").delete()
        session.commit()

        for row in rows:
            session.add(Event(
                source="localist",
                created_by=None,
                external_id=row["external_id"],
                title=row["title"],
                description=row["description"],
                location=row["location"],
                starts_at=_parse(row["starts_at"]),
                ends_at=_parse(row["ends_at"]),
                status=row["status"],
                is_free=row["is_free"],
                experience=row["experience"],
                keywords=row["keywords"],
                groups=row["groups"],
                event_types=row["event_types"],
                localist_url=row["localist_url"],
                external_updated_at=_parse(row["external_updated_at"]),
            ))
        session.commit()

    return len(rows)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Load the frozen demo events into the database."
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="embed them afterwards, the same work as python -m app.rag.backfill",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="override RAG_PROVIDER while embedding, e.g. 'fake'",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    loaded = load_events()
    log.info("loaded %d demo events from %s", loaded, FIXTURE)

    if not args.embed:
        log.info("not embedded yet -- run: python -m app.rag.backfill")
        return

    # Imported here rather than at the top: loading events needs no API key,
    # and a missing one should not stop somebody who only wanted the rows.
    from app.rag.indexer import index_events
    from app.rag.providers import get_embedding_provider

    stats = index_events(provider=get_embedding_provider(args.provider))
    log.info("embedded %d, already current %d",
             stats["embedded"], stats["skipped"])


if __name__ == "__main__":
    main()
