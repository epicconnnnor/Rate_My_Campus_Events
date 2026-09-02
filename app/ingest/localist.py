"""
Localist ingest for RateMyCampusEvents.

Pulls the UMass campus calendar into the events table:

    python -m app.ingest.localist

Imported events are keyed on the Localist event id (external_id), carry
source='localist', and have no author. Nothing is ever hard-deleted: an event
that stops coming back from the feed keeps its row and simply goes stale, which
is what last_seen_at records.
"""

import argparse
import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlmodel import Session, select

from app.db.database import engine
from app.models.event import Event

log = logging.getLogger("localist")

API_URL = "https://events.umass.edu/api/2/events"

# The API caps a range at 370 days and pp at 100.
MAX_DAYS = 370
DEFAULT_DAYS = 90
PER_PAGE = 100

REQUEST_TIMEOUT = 30

# Columns owned by the feed. Every one of them is overwritten on each sync, and
# a difference in any of them is what counts as an event having changed.
# created_at / created_by / date_time are deliberately absent: date_time belongs
# to user-created events, and an imported event has no author.
FEED_COLUMNS = (
    "title",
    "description",
    "location",
    "starts_at",
    "ends_at",
    "status",
    "is_free",
    "experience",
    "keywords",
    "localist_url",
    "external_updated_at",
)


# =============================================================================
# FETCH
# =============================================================================


def fetch_page(page: int, days: int, http: requests.Session) -> Dict:
    """One page of the feed.

    distinct=true is what keeps this to one row per event: without it a
    recurring event comes back once per occurrence.
    """
    response = http.get(
        API_URL,
        params={"days": days, "pp": PER_PAGE, "page": page, "distinct": "true"},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def fetch_all(days: int) -> List[Dict]:
    """Every event in the window, paginating until page.total is exhausted."""
    raw_events: List[Dict] = []
    with requests.Session() as http:
        page = 1
        total_pages = 1
        while page <= total_pages:
            payload = fetch_page(page, days, http)
            total_pages = payload.get("page", {}).get("total", 1)
            batch = payload.get("events", [])
            raw_events.extend(item["event"] for item in batch)
            log.info("page %d/%d: %d events", page, total_pages, len(batch))
            page += 1
    return raw_events


# =============================================================================
# PARSE
# =============================================================================


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    """Localist returns offset-aware ISO 8601, e.g. 2026-09-02T14:00:00-04:00."""
    if not value:
        return None
    return datetime.fromisoformat(value)


def _location(raw: Dict) -> Optional[str]:
    parts = [
        (raw.get("location_name") or "").strip(),
        (raw.get("room_number") or "").strip(),
    ]
    return ", ".join(part for part in parts if part) or None


def parse_event(raw: Dict) -> Dict:
    """Map one Localist event onto our columns."""
    instances = raw.get("event_instances") or []
    # distinct=true means the first instance is the one occurrence we track.
    first = instances[0]["event_instance"] if instances else {}

    return {
        "external_id": str(raw["id"]),
        "title": raw.get("title"),
        # description_text, not description -- the latter is HTML.
        "description": raw.get("description_text"),
        "location": _location(raw),
        "starts_at": _parse_timestamp(first.get("start")),
        "ends_at": _parse_timestamp(first.get("end")),
        "status": raw.get("status"),
        "is_free": raw.get("free"),
        "experience": raw.get("experience"),
        "keywords": raw.get("keywords") or None,
        "localist_url": raw.get("localist_url"),
        "external_updated_at": _parse_timestamp(raw.get("updated_at")),
    }


# =============================================================================
# UPSERT
# =============================================================================


def _classify(session: Session, parsed: List[Dict]) -> Counter:
    """Count new / changed / unchanged before writing, for the run summary."""
    external_ids = [row["external_id"] for row in parsed]
    existing = {
        event.external_id: event
        for event in session.exec(
            select(Event).where(Event.external_id.in_(external_ids))
        )
    }

    stats = Counter()
    for row in parsed:
        prior = existing.get(row["external_id"])
        if prior is None:
            stats["new"] += 1
        elif any(getattr(prior, column) != row[column] for column in FEED_COLUMNS):
            stats["changed"] += 1
        else:
            stats["unchanged"] += 1
    return stats


def upsert(session: Session, parsed: List[Dict], synced_at: datetime) -> None:
    """Insert or update on external_id.

    The unique index from migration 0002 is what makes this safe to repeat and
    impossible to duplicate, even if two syncs overlap.
    """
    rows = [
        dict(
            row,
            source="localist",
            created_by=None,
            created_at=synced_at,
            last_seen_at=synced_at,
        )
        for row in parsed
    ]

    statement = pg_insert(Event).values(rows)
    statement = statement.on_conflict_do_update(
        index_elements=["external_id"],
        set_={
            column: getattr(statement.excluded, column)
            for column in FEED_COLUMNS + ("source", "last_seen_at")
        },
    )
    session.execute(statement)


# =============================================================================
# SYNC
# =============================================================================


def sync(days: int = DEFAULT_DAYS, dry_run: bool = False) -> Counter:
    if not 1 <= days <= MAX_DAYS:
        raise ValueError(f"days must be between 1 and {MAX_DAYS}, got {days}")

    raw_events = fetch_all(days)
    parsed = [parse_event(raw) for raw in raw_events]
    log.info("fetched %d events over %d days", len(parsed), days)

    synced_at = datetime.now(timezone.utc)

    with Session(engine) as session:
        stats = _classify(session, parsed)
        if dry_run:
            log.info("dry run: nothing written")
        else:
            upsert(session, parsed, synced_at)
            session.commit()

    return stats


# =============================================================================
# CLI
# =============================================================================


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Sync UMass Localist events into the events table."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help=f"size of the window to pull, 1-{MAX_DAYS} (default {DEFAULT_DAYS})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="fetch and report, but write nothing",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    stats = sync(days=args.days, dry_run=args.dry_run)
    log.info(
        "done: %d new, %d changed, %d unchanged",
        stats["new"],
        stats["changed"],
        stats["unchanged"],
    )


if __name__ == "__main__":
    main()
