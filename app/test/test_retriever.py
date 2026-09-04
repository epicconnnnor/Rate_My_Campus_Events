"""
Tests for the retrieval path.

Nothing here talks to a database or a model. The chat provider is a stub that
returns whatever the test queued, and the search function is injected, so the
fallback ladder can be walked one rung at a time.
"""

import json
from datetime import date, datetime, timedelta, timezone

import pytest

from app.rag.retriever import (
    CAMPUS_TZ,
    DEFAULT_WINDOW_DAYS,
    MAX_ALTERNATIVES,
    MAX_WINDOW_DAYS,
    NOTHING_FOUND,
    WIDEN_DAYS,
    FilterOverrides,
    Match,
    QueryFilters,
    build_extraction_prompt,
    current_campus_date,
    describe_match,
    extract_filters,
    format_event_time,
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


# =============================================================================
# HOW A TIME IS WRITTEN
# =============================================================================

# Stored times are UTC. Campus is four hours behind in September, so an event
# stored at 20:00 is a 4pm event and must read as one everywhere.
STORED_UTC = datetime(2026, 9, 5, 20, 0, tzinfo=timezone.utc)
AS_WRITTEN = "Saturday September 05, 04:00 PM"


def test_a_stored_time_is_written_in_campus_time():
    assert format_event_time(STORED_UTC) == AS_WRITTEN


def test_an_event_with_no_start_says_so_rather_than_guessing():
    assert format_event_time(None) == "date unknown"


def test_the_answer_and_the_judge_are_shown_the_same_time():
    """The one that matters.

    The eval marks an answer against a context built separately from the same
    events. While the two rendered times differently -- the answer in campus
    time, the judge from the stored UTC value -- the judge read a correct 4:00 PM
    as a four-hour invention and failed six of nine golden questions for it.

    Both sides go through format_event_time now, so this holds by construction.
    It is asserted anyway: the failure was silent, cost a day of quota to see,
    and looked exactly like a hallucinating model.
    """
    from app.test.test_hallucinations import _context

    match = Match(fake_event(starts_at=STORED_UTC), distance=0.1)

    answer_side = describe_match(match)
    judge_side = _context([match])

    assert AS_WRITTEN in answer_side
    assert AS_WRITTEN in judge_side

    # And neither of them leaks the stored value it was rendered from.
    assert "20:00" not in answer_side
    assert "20:00" not in judge_side


# =============================================================================
# THE CHIPS
# =============================================================================
#
# The filter chips above the chat box are not a second filter applied to the
# answer. They are handed to retrieval and beat whatever the model read out of
# the sentence, which is the only way "free" can mean free rather than mean a
# hint that a model is free to ignore.


def test_an_override_beats_what_the_model_read():
    filters = QueryFilters(start=midnight(SUNDAY), end=midnight(SUNDAY),
                           free=False)
    FilterOverrides(free=True).apply(filters)
    assert filters.free is True


def test_an_unset_override_leaves_the_extraction_alone():
    """Tapping "virtual" must not throw away the dates the sentence asked for,
    nor un-set a "free" the model correctly read."""
    filters = QueryFilters(start=midnight(SUNDAY), end=midnight(SUNDAY),
                           free=True, keywords=["music"])
    FilterOverrides(experience="virtual").apply(filters)

    assert filters.experience == "virtual"
    assert filters.free is True
    assert filters.keywords == ["music"]


def test_no_chips_at_all_changes_nothing():
    filters = QueryFilters(start=midnight(SUNDAY), end=midnight(SUNDAY),
                           free=True, experience="inperson")
    FilterOverrides().apply(filters)
    assert (filters.free, filters.experience) == (True, "inperson")


def test_an_empty_set_of_overrides_is_falsy():
    """retrieve() skips the whole step on this, so it has to be honest."""
    assert not FilterOverrides()
    assert FilterOverrides(free=True)
    assert FilterOverrides(category="Lecture")


def test_the_chips_reach_every_rung_of_the_ladder():
    """Not just the first search. A widened retry that quietly dropped the
    chips would answer a question nobody asked."""
    search = RecordingSearch([], [], [Match(fake_event(), 1.5)])
    chat = StubChat(extraction(), "here is something else")

    retrieve("anything?", today=SUNDAY, chat=chat, embedder=StubEmbedder(),
             search=search, overrides=FilterOverrides(free=True,
                                                      category="Lecture"))

    assert search.calls
    for call in search.calls:
        assert call["filters"].free is True
        assert call["filters"].category == "Lecture"


# -- what arrives from the browser is not trusted --------------------------


def test_an_unticked_box_is_not_a_search_for_paid_events():
    """Absent means "not asked for". Reading it as False would quietly hide
    every free event the moment somebody did not tick the box."""
    assert FilterOverrides.from_form(free=None).free is None
    assert FilterOverrides.from_form().free is None


def test_a_ticked_box_asks_for_free_events():
    assert FilterOverrides.from_form(free=True).free is True


@pytest.mark.parametrize("value", ["online", "", "   ", "INPERSON; DROP"])
def test_an_experience_outside_the_three_known_values_is_dropped(value):
    """Same posture as extract_filters: junk widens the search, never breaks
    the page and never reaches the query."""
    assert FilterOverrides.from_form(experience=value).experience is None


def test_a_known_experience_comes_through_in_any_case():
    assert FilterOverrides.from_form(experience="Virtual").experience == "virtual"
    assert FilterOverrides.from_form(experience=" hybrid ").experience == "hybrid"


def test_the_any_category_chip_posts_an_empty_string():
    """The default radio has value="", which has to read as no category rather
    than as a category nothing is filed under."""
    assert FilterOverrides.from_form(category="").category is None
    assert FilterOverrides.from_form(category="   ").category is None


def test_a_category_is_taken_as_written():
    """Matched against the calendar's own event_types, so it is not lowercased
    or otherwise tidied on the way through."""
    assert FilterOverrides.from_form(category="Film Screening").category == (
        "Film Screening"
    )


def test_a_category_survives_the_widening():
    """widened() rebuilds the filters field by field, so anything added to
    QueryFilters and forgotten there vanishes on the first fallback."""
    filters = QueryFilters(start=midnight(SUNDAY), end=midnight(SUNDAY),
                           category="Lecture", free=True,
                           experience="virtual", keywords=["talk"])
    widened = filters.widened(WIDEN_DAYS)

    assert widened.category == "Lecture"
    assert widened.free is True
    assert widened.experience == "virtual"
    assert widened.keywords == ["talk"]


# =============================================================================
# WHAT COUNTS AS TODAY
# =============================================================================


def test_today_is_the_real_date_when_demo_date_is_unset(monkeypatch):
    import app.rag.retriever as retriever

    monkeypatch.setattr(retriever, "DEMO_DATE", None)
    assert current_campus_date() == datetime.now(CAMPUS_TZ).date()


def test_demo_date_moves_what_the_app_calls_today(monkeypatch):
    """A clone started outside the frozen window searches a calendar it has
    already walked past, and every question comes back empty."""
    import app.rag.retriever as retriever

    monkeypatch.setattr(retriever, "DEMO_DATE", date(2026, 9, 2))
    assert current_campus_date() == date(2026, 9, 2)


def test_an_explicit_today_still_wins_over_demo_date(monkeypatch):
    """The eval passes EVAL_TODAY in. Nothing about a demo setting may move
    what the judged run is standing on."""
    import app.rag.retriever as retriever

    monkeypatch.setattr(retriever, "DEMO_DATE", date(2026, 10, 1))
    search = RecordingSearch([Match(fake_event(), 0.1)])
    result, _ = run(search)

    assert result.filters.start == midnight(date(2026, 9, 11))


def test_the_distance_cutoff_belongs_to_the_embedding_model():
    """0.70 was measured against text-embedding-3-small, not chosen for feel.

    Pinned so that changing EMBEDDING_MODEL without re-measuring shows up here
    rather than as every question quietly falling through to the near-miss
    branch. Under the previous model the right answer was 0.6; the same number
    under this one threw away a Libraries event asked for by the word
    "library".
    """
    from app.core.config import EMBEDDING_MODEL, RETRIEVAL_MAX_DISTANCE

    assert RETRIEVAL_MAX_DISTANCE == 0.70
    assert EMBEDDING_MODEL == "text-embedding-3-small"
