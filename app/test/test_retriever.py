"""
Tests for the retrieval path.

Nothing here talks to a database or a model. The chat provider is a stub that
returns whatever the test queued, and the search function is injected, so the
fallback ladder can be walked one rung at a time.
"""

import json
from datetime import date, datetime, timedelta

import pytest

from app.rag.retriever import (
    CAMPUS_TZ,
    DEFAULT_WINDOW_DAYS,
    MAX_ALTERNATIVES,
    MAX_WINDOW_DAYS,
    NOTHING_FOUND,
    WIDEN_DAYS,
    Match,
    QueryFilters,
    build_extraction_prompt,
    extract_filters,
    retrieve,
)

# 2026-09-06 is a Sunday. The tests below depend on that, so they check it.
SUNDAY = date(2026, 9, 6)


class StubChat:
    """Returns queued replies and remembers every prompt it was given."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else "{}"


def extraction(**fields):
    """A model reply containing the four fields, with sane defaults."""
    payload = {
        "date_range": None,
        "free": None,
        "experience": None,
        "keywords": [],
    }
    payload.update(fields)
    return json.dumps(payload)


def midnight(day):
    return datetime.combine(day, datetime.min.time(), tzinfo=CAMPUS_TZ)


def fake_event(event_id=1, title="Something", starts_at=None):
    return {
        "event_id": event_id,
        "title": title,
        "starts_at": starts_at or midnight(SUNDAY),
        "location": "Student Union",
    }


# =============================================================================
# DATE EXTRACTION
# =============================================================================


def test_sunday_is_actually_a_sunday():
    assert SUNDAY.strftime("%A") == "Sunday"


def test_prompt_carries_today_and_the_timezone():
    """The model can only return absolute dates if it is told where it stands."""
    prompt = build_extraction_prompt("anything on tonight?", SUNDAY)
    assert "2026-09-06" in prompt
    assert "Sunday" in prompt
    assert "America/New_York" in prompt


def test_this_weekend_on_a_sunday_is_just_today():
    """Asked on a Sunday, the weekend has one day left in it."""
    chat = StubChat(extraction(
        date_range={"start": "2026-09-06", "end": "2026-09-06"}
    ))
    filters = extract_filters("what's happening this weekend?", SUNDAY, chat)

    assert filters.start == midnight(SUNDAY)
    # end is exclusive, so a single day runs to the following midnight.
    assert filters.end == midnight(date(2026, 9, 7))
    assert (filters.end - filters.start) == timedelta(days=1)


def test_this_weekend_on_a_sunday_may_mean_the_next_one():
    """The other honest reading of the same question, and it still works."""
    chat = StubChat(extraction(
        date_range={"start": "2026-09-12", "end": "2026-09-13"}
    ))
    filters = extract_filters("what's happening this weekend?", SUNDAY, chat)

    assert filters.start == midnight(date(2026, 9, 12))
    assert filters.end == midnight(date(2026, 9, 14))


def test_backwards_range_is_read_as_the_span_between_the_dates():
    """A Sunday makes this easy to get wrong -- 'Saturday to Sunday' anchored on
    today comes back inverted. Reading it backwards beats returning nothing."""
    chat = StubChat(extraction(
        date_range={"start": "2026-09-06", "end": "2026-09-05"}
    ))
    filters = extract_filters("this weekend?", SUNDAY, chat)

    assert filters.start == midnight(date(2026, 9, 5))
    assert filters.end == midnight(date(2026, 9, 7))
    assert filters.start < filters.end


def test_no_date_in_the_question_searches_the_whole_calendar():
    chat = StubChat(extraction(keywords=["music"]))
    filters = extract_filters("any live music?", SUNDAY, chat)

    assert filters.start == midnight(SUNDAY)
    assert filters.end == midnight(SUNDAY) + timedelta(days=DEFAULT_WINDOW_DAYS)


@pytest.mark.parametrize("bad", [
    {"start": "next friday", "end": "next sunday"},
    {"start": "09/06/2026", "end": "09/07/2026"},
    {"start": "2026-13-45", "end": "2026-13-46"},
    {"start": None, "end": None},
    "this weekend",
])
def test_dates_that_are_not_iso_are_discarded(bad):
    """The model was told absolute ISO only. Anything else is not parsed by
    hand, it is dropped and the search goes broad."""
    chat = StubChat(extraction(date_range=bad))
    filters = extract_filters("when?", SUNDAY, chat)

    assert filters.start == midnight(SUNDAY)
    assert filters.end == midnight(SUNDAY) + timedelta(days=DEFAULT_WINDOW_DAYS)


def test_absurd_range_is_clamped():
    chat = StubChat(extraction(
        date_range={"start": "2026-09-06", "end": "2999-01-01"}
    ))
    filters = extract_filters("everything ever", SUNDAY, chat)

    assert (filters.end - filters.start).days == MAX_WINDOW_DAYS + 1


def test_free_and_experience_and_keywords_come_through():
    chat = StubChat(extraction(
        date_range={"start": "2026-09-06", "end": "2026-09-06"},
        free=True,
        experience="virtual",
        keywords=["music", " concert "],
    ))
    filters = extract_filters("free online music today?", SUNDAY, chat)

    assert filters.free is True
    assert filters.experience == "virtual"
    assert filters.keywords == ["music", "concert"]


def test_experience_outside_the_three_known_values_is_ignored():
    chat = StubChat(extraction(experience="outdoors"))
    assert extract_filters("outside?", SUNDAY, chat).experience is None


def test_free_only_accepts_a_real_boolean():
    chat = StubChat(extraction(free="yes"))
    assert extract_filters("free stuff?", SUNDAY, chat).free is None


def test_a_reply_that_is_not_json_falls_back_to_a_broad_search():
    chat = StubChat("Sure! Here are some events you might like:")
    filters = extract_filters("anything on?", SUNDAY, chat)

    assert filters.start == midnight(SUNDAY)
    assert filters.keywords == []


def test_json_in_a_code_fence_is_still_read():
    chat = StubChat("```json\n" + extraction(free=True) + "\n```")
    assert extract_filters("free?", SUNDAY, chat).free is True


def test_keywords_are_what_gets_embedded_when_there_are_any():
    filters = QueryFilters(start=midnight(SUNDAY), end=midnight(SUNDAY),
                           keywords=["fun", "social"])
    assert filters.search_text("something fun on friday?") == "fun, social"


def test_the_question_is_embedded_when_no_keywords_survived():
    filters = QueryFilters(start=midnight(SUNDAY), end=midnight(SUNDAY))
    assert filters.search_text("something fun?") == "something fun?"


# =============================================================================
# THE FALLBACK LADDER
# =============================================================================


class RecordingSearch:
    """Returns a queued result per call and records how it was asked."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def __call__(self, filters, query_vector, *, limit, max_distance):
        self.calls.append({
            "filters": filters,
            "limit": limit,
            "max_distance": max_distance,
        })
        return self.results.pop(0) if self.results else []


class StubEmbedder:
    def embed_documents(self, texts):
        return [[0.0] for _ in texts]

    def embed_query(self, text):
        return [0.0]


def run(search, chat=None, question="anything on friday?"):
    chat = chat or StubChat(extraction(
        date_range={"start": "2026-09-11", "end": "2026-09-11"}
    ), "Nothing then, but here is what is nearby.")
    return retrieve(
        question,
        today=SUNDAY,
        chat=chat,
        embedder=StubEmbedder(),
        search=search,
    ), chat


def test_a_direct_hit_never_reaches_the_fallbacks():
    search = RecordingSearch([Match(fake_event(), 0.1)])
    result, chat = run(search)

    assert result.outcome == "matches"
    assert len(search.calls) == 1
    # Only the extraction prompt. Nothing asked a model to phrase an apology.
    assert len(chat.prompts) == 1


def test_first_fallback_widens_the_dates_by_two_weeks():
    search = RecordingSearch([], [Match(fake_event(title="Nearby"), 0.3)])
    result, _ = run(search)

    assert result.outcome == "alternatives"
    assert len(search.calls) == 2

    asked, widened = search.calls[0]["filters"], search.calls[1]["filters"]
    assert widened.start == asked.start - timedelta(days=WIDEN_DAYS)
    assert widened.end == asked.end + timedelta(days=WIDEN_DAYS)
    # Still asking for relevance at this rung.
    assert search.calls[1]["max_distance"] is not None


def test_first_fallback_caps_how_many_alternatives_it_offers():
    search = RecordingSearch([], [Match(fake_event(), 0.3)])
    run(search)
    assert search.calls[1]["limit"] == MAX_ALTERNATIVES


def test_second_fallback_drops_the_keyword_filter():
    """Two empty rungs, so relevance is abandoned and anything in range will
    do -- same widened window, no distance cutoff."""
    search = RecordingSearch([], [], [Match(fake_event(title="Whatever"), 1.9)])
    result, _ = run(search)

    assert result.outcome == "alternatives"
    assert len(search.calls) == 3
    assert search.calls[2]["max_distance"] is None
    assert search.calls[2]["filters"].start == search.calls[1]["filters"].start
    assert search.calls[2]["limit"] == MAX_ALTERNATIVES


def test_third_fallback_is_canned_and_costs_nothing():
    search = RecordingSearch([], [], [])
    result, chat = run(search)

    assert result.outcome == "empty"
    assert result.matches == []
    assert result.message == NOTHING_FOUND
    assert len(search.calls) == 3
    # Extraction only. No model was asked to write the dead end.
    assert len(chat.prompts) == 1


def test_the_redirect_only_ever_sees_the_alternatives():
    """The model can't invent an event it was never shown."""
    offered = [
        Match(fake_event(1, "Open Mic Night"), 0.3),
        Match(fake_event(2, "Ceramics Sale"), 0.4),
    ]
    chat = StubChat(
        extraction(date_range={"start": "2026-09-11", "end": "2026-09-11"}),
        "Nothing on Friday, but there's an open mic.",
    )
    result, _ = run(RecordingSearch([], offered), chat=chat)

    redirect_prompt = chat.prompts[1]
    assert "Open Mic Night" in redirect_prompt
    assert "Ceramics Sale" in redirect_prompt
    assert "Homecoming Parade" not in redirect_prompt
    assert result.message == "Nothing on Friday, but there's an open mic."


def test_filters_survive_into_the_result():
    search = RecordingSearch([Match(fake_event(), 0.1)])
    result, _ = run(search)

    assert result.filters.start == midnight(date(2026, 9, 11))
    assert result.filters.end == midnight(date(2026, 9, 12))


def test_free_and_experience_are_carried_into_every_rung():
    chat = StubChat(
        extraction(free=True, experience="inperson"),
        "here is something else",
    )
    search = RecordingSearch([], [], [Match(fake_event(), 1.5)])
    run(search, chat=chat)

    for call in search.calls:
        assert call["filters"].free is True
        assert call["filters"].experience == "inperson"
