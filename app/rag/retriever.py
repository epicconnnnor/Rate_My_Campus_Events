"""
Turning a question into events.

One path, every time: the model reads four things out of the question, Python
checks them, SQL narrows the table down, and the vector search ranks what is
left. There is no router deciding between strategies and the model never writes
SQL -- it hands back four fields and Python builds the query.

Filtering happens before the vector search, never after. Searching first and
filtering the results afterwards would let a Friday question come back with
nothing simply because the ten nearest events were all on Tuesday.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

from sqlmodel import Session, select

from app.core.config import RETRIEVAL_MAX_DISTANCE, RETRIEVAL_TOP_K
from app.db.database import engine
from app.models.event import Event, EventEmbedding
from app.rag.providers import (
    ChatProvider,
    EmbeddingProvider,
    get_chat_provider,
    get_embedding_provider,
)

log = logging.getLogger("retriever")

CAMPUS_TZ = ZoneInfo("America/New_York")

# Used when the question says nothing about when. Matches the ingest window, so
# it means "everything we know about".
DEFAULT_WINDOW_DAYS = 90

# No question should be able to ask for more than the calendar we hold.
MAX_WINDOW_DAYS = 370

# How far either side of the asked-for dates to look when nothing matched.
WIDEN_DAYS = 14

# A redirect that lists half the calendar is not a redirect.
MAX_ALTERNATIVES = 5

VALID_EXPERIENCES = ("inperson", "virtual", "hybrid")

NOTHING_FOUND = (
    "I couldn't find anything on the calendar for that. Try a different date "
    "range, or ask about something broader."
)


# =============================================================================
# SHAPES
# =============================================================================


@dataclass
class QueryFilters:
    """What the question asked for, after Python has checked it."""

    start: datetime
    end: datetime  # exclusive
    free: Optional[bool] = None
    experience: Optional[str] = None
    keywords: List[str] = field(default_factory=list)

    def widened(self, days: int) -> "QueryFilters":
        return QueryFilters(
            start=self.start - timedelta(days=days),
            end=self.end + timedelta(days=days),
            free=self.free,
            experience=self.experience,
            keywords=self.keywords,
        )

    def search_text(self, question: str) -> str:
        """What actually gets embedded.

        The keywords are the question with the scaffolding taken off, so they
        make a cleaner query vector than the raw sentence. With none extracted,
        the sentence is all there is.
        """
        return ", ".join(self.keywords) if self.keywords else question


@dataclass
class Match:
    event: Dict
    distance: float


@dataclass
class RetrievalResult:
    # "matches"      -- real answers to the question as asked
    # "alternatives" -- nothing matched, here is what is nearby instead
    # "empty"        -- nothing at all, and no model was asked to dress it up
    outcome: str
    matches: List[Match]
    filters: QueryFilters
    message: Optional[str] = None


# =============================================================================
# EXTRACTION
# =============================================================================

EXTRACTION_PROMPT = """\
You are reading a question about events on the UMass Amherst campus and \
pulling out the search filters it implies.

Today is {today} ({weekday}). The campus timezone is America/New_York.

Return one JSON object and nothing else, with exactly these four keys:

  "date_range": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}} or null
  "free":       true, false, or null
  "experience": "inperson", "virtual", "hybrid", or null
  "keywords":   a list of short strings, possibly empty

Rules:
- Dates must be absolute calendar dates in YYYY-MM-DD form. Never return a \
relative phrase like "this weekend" or "tomorrow" -- work out the actual dates \
from today's date above.
- "start" and "end" are both inclusive. A single day is the same date twice.
- Use null for anything the question does not ask about. Do not guess.
- "keywords" is what the person is actually looking for, not the scaffolding \
of the sentence. "something fun on Friday that isn't a lecture" gives \
["fun", "social"], not ["Friday"].
- Do not write SQL. Do not explain. Return only the JSON object.

Question: {question}
"""


def _default_window(today: date) -> QueryFilters:
    start = datetime.combine(today, time.min, tzinfo=CAMPUS_TZ)
    return QueryFilters(
        start=start, end=start + timedelta(days=DEFAULT_WINDOW_DAYS)
    )


def _parse_day(value) -> Optional[date]:
    """An ISO date, or nothing. Anything else the model invented is discarded."""
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def parse_json_object(raw: str) -> Dict:
    """The model was told to return bare JSON; be forgiving if it fenced it."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        parsed = json.loads(text)
    except (ValueError, IndexError):
        log.warning("extraction did not return JSON: %r", raw[:200])
        return {}
    return parsed if isinstance(parsed, dict) else {}


def build_extraction_prompt(question: str, today: date) -> str:
    return EXTRACTION_PROMPT.format(
        today=today.isoformat(),
        weekday=today.strftime("%A"),
        question=question,
    )


def extract_filters(question: str, today: date,
                    chat: ChatProvider) -> QueryFilters:
    """Ask the model for four fields, then decide what to believe.

    Everything here is defensive on purpose: a bad extraction should quietly
    become a broad search, never an exception and never a bad query.
    """
    raw = chat.complete(build_extraction_prompt(question, today))
    parsed = parse_json_object(raw)

    filters = _default_window(today)

    date_range = parsed.get("date_range")
    if isinstance(date_range, dict):
        start_day = _parse_day(date_range.get("start"))
        end_day = _parse_day(date_range.get("end"))
        if start_day and end_day:
            # A model that returns the days backwards meant the span between
            # them, so read it that way rather than throwing the question out.
            if end_day < start_day:
                start_day, end_day = end_day, start_day
            span = (end_day - start_day).days
            if span > MAX_WINDOW_DAYS:
                end_day = start_day + timedelta(days=MAX_WINDOW_DAYS)
            filters.start = datetime.combine(start_day, time.min, tzinfo=CAMPUS_TZ)
            # end is inclusive in the model's answer, exclusive in the query.
            filters.end = datetime.combine(
                end_day + timedelta(days=1), time.min, tzinfo=CAMPUS_TZ
            )

    if isinstance(parsed.get("free"), bool):
        filters.free = parsed["free"]

    experience = parsed.get("experience")
    if isinstance(experience, str) and experience.lower() in VALID_EXPERIENCES:
        filters.experience = experience.lower()

    keywords = parsed.get("keywords")
    if isinstance(keywords, list):
        filters.keywords = [
            word.strip() for word in keywords
            if isinstance(word, str) and word.strip()
        ]

    return filters


# =============================================================================
# SEARCH
# =============================================================================


def search_events(filters: QueryFilters, query_vector: List[float], *,
                  limit: int,
                  max_distance: Optional[float]) -> List[Match]:
    """Narrow the table down, then rank what survives by distance.

    One statement, so Postgres applies the WHERE before it ever looks at a
    vector. Only starts_at is filtered on -- ends_at is missing on a good
    fraction of the feed.

    max_distance of None keeps everything in range, however unrelated.
    """
    with Session(engine) as session:
        distance = EventEmbedding.embedding.cosine_distance(query_vector)
        statement = (
            select(Event, distance.label("distance"))
            .join(EventEmbedding, EventEmbedding.event_id == Event.event_id)
            .where(Event.starts_at >= filters.start)
            .where(Event.starts_at < filters.end)
        )
        if filters.free is not None:
            statement = statement.where(Event.is_free == filters.free)
        if filters.experience is not None:
            statement = statement.where(Event.experience == filters.experience)
        if max_distance is not None:
            statement = statement.where(distance <= max_distance)

        statement = statement.order_by(distance).limit(limit)

        return [
            Match(event=event.model_dump(), distance=float(value))
            for event, value in session.exec(statement)
        ]


# =============================================================================
# REDIRECT
# =============================================================================

REDIRECT_PROMPT = """\
Someone asked about events on the UMass Amherst campus and nothing matched \
what they asked for. Below is everything you are allowed to mention.

Their question: {question}

The only events available to you:
{alternatives}

Write two or three sentences telling them nothing matched, then point them at \
these. Mention only events from the list above, with their real dates. Do not \
invent an event, a date, or a detail that is not written above.
"""


def describe_match(match: Match) -> str:
    event = match.event
    when = event.get("starts_at")
    when = when.astimezone(CAMPUS_TZ).strftime("%A %B %d, %I:%M %p") if when else "date unknown"
    where = event.get("location") or "location unknown"
    return f"- {event.get('title')} ({when}, {where})"


def phrase_redirect(question: str, alternatives: List[Match],
                    chat: ChatProvider) -> str:
    """Let the model write the apology, but only from the list it is given."""
    prompt = REDIRECT_PROMPT.format(
        question=question,
        alternatives="\n".join(describe_match(match) for match in alternatives),
    )
    return chat.complete(prompt).strip()


# =============================================================================
# RETRIEVAL
# =============================================================================

SearchFn = Callable[..., List[Match]]


def retrieve(question: str, *,
             today: Optional[date] = None,
             chat: Optional[ChatProvider] = None,
             embedder: Optional[EmbeddingProvider] = None,
             search: Optional[SearchFn] = None,
             top_k: int = RETRIEVAL_TOP_K,
             max_distance: float = RETRIEVAL_MAX_DISTANCE) -> RetrievalResult:
    """Question in, events out, with three fallbacks before giving up."""
    chat = chat or get_chat_provider()
    search = search or search_events
    today = today or datetime.now(CAMPUS_TZ).date()

    filters = extract_filters(question, today, chat)

    embedder = embedder or get_embedding_provider()
    query_vector = embedder.embed_query(filters.search_text(question))

    matches = search(filters, query_vector, limit=top_k,
                     max_distance=max_distance)
    if matches:
        return RetrievalResult("matches", matches, filters)

    # 1. Nothing on those dates. Look either side of them.
    widened = filters.widened(WIDEN_DAYS)
    alternatives = search(widened, query_vector, limit=MAX_ALTERNATIVES,
                          max_distance=max_distance)

    # 2. Still nothing, so stop asking for relevance and take what is there.
    if not alternatives:
        alternatives = search(widened, query_vector, limit=MAX_ALTERNATIVES,
                              max_distance=None)

    # 3. The calendar is genuinely empty here. Say so without paying a model to
    #    say it, and without giving one the chance to invent an event.
    if not alternatives:
        return RetrievalResult("empty", [], filters, message=NOTHING_FOUND)

    return RetrievalResult(
        "alternatives",
        alternatives,
        filters,
        message=phrase_redirect(question, alternatives, chat),
    )
