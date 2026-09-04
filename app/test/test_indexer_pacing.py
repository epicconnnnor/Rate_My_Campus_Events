"""
Tests for staying inside the per-minute quotas.

One Pacer serves both the embedding backfill and the chat calls, against
different limits, so the pacing tests cover the shared class and the two
callers' numbers separately.

Nothing here sleeps for real: the clock is a fake so a minute costs nothing.
"""

import pytest

from app.core.config import (CHAT_REQUESTS_PER_MINUTE, CHAT_RPM_BY_PROVIDER,
                             EMBEDDING_DPM_BY_PROVIDER)
from app.rag import pacing
from app.rag.indexer import _document_pacer

# The document tests pin Google's number rather than inheriting whichever
# provider the environment names, so they mean the same thing on any machine.
GEMINI_DOCUMENTS_PER_MINUTE = EMBEDDING_DPM_BY_PROVIDER["gemini"]
from app.rag.pacing import Pacer
from app.rag.providers import (SERVER_BACKOFF_BASE_SECONDS,
                               SERVER_BACKOFF_CAP_SECONDS,
                               _is_daily_quota, _quota_ids,
                               SERVER_RETRY_ATTEMPTS, _is_quota_error,
                               _is_server_error, _quota_delay,
                               _retry_after, _server_delay, chat_pacer)


class FakeClock:
    """Stands in for the time module. Sleeping just moves the clock."""

    def __init__(self):
        self.now = 1000.0
        self.slept = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(pacing, "time", fake)
    return fake


# =============================================================================
# PACING
# =============================================================================


def test_the_document_limit_is_googles_number_not_ours():
    """100 embed_content requests a minute on the free tier, metered per
    document. Not a number picked for feel: exceeding it is a hard 429, and
    the first eval run died on document 101 of 102."""
    assert GEMINI_DOCUMENTS_PER_MINUTE == 100


def test_gemini_paces_under_its_published_ceiling():
    """The Flash-Lite models publish 15 a minute, and this is deliberately not
    that number. Pacing to exactly 15 meant arriving at 14 and 15 of 15 every
    minute, leaving nothing for what a single process cannot see: another job
    on the same key, a local run, a clock behind the server's."""
    assert CHAT_RPM_BY_PROVIDER["gemini"] == 12
    assert CHAT_RPM_BY_PROVIDER["gemini"] < 15


def test_openai_paces_higher_but_still_paces():
    """OpenAI's limits are not the binding constraint -- gpt-4o-mini allows
    hundreds a minute on the smallest paid tier. This is a brake on a runaway
    loop spending real money, not a quota dodge, so it stays a real number."""
    assert CHAT_RPM_BY_PROVIDER["openai"] == 60
    assert CHAT_RPM_BY_PROVIDER["openai"] > CHAT_RPM_BY_PROVIDER["gemini"]


def test_every_real_provider_has_both_limits():
    """A provider added to one map and not the other silently falls back to
    Gemini's numbers, which would pace an OpenAI run into the ground."""
    for provider in ("openai", "gemini", "fake"):
        assert CHAT_RPM_BY_PROVIDER[provider] > 0
        assert EMBEDDING_DPM_BY_PROVIDER[provider] > 0


def test_the_resolved_limit_is_one_of_them():
    assert CHAT_REQUESTS_PER_MINUTE in CHAT_RPM_BY_PROVIDER.values()


def test_staying_under_the_limit_never_waits(clock):
    pacer = _document_pacer(GEMINI_DOCUMENTS_PER_MINUTE)
    for _ in range(3):
        pacer.reserve(32)
    assert clock.slept == []


def test_a_cached_batch_costs_nothing(clock):
    """The indexer reserves what will actually be sent, not what it was asked
    for. A run answered entirely from the embedding cache used to reach 102
    documents and sit out a quota window for requests nobody was making."""
    pacer = _document_pacer(GEMINI_DOCUMENTS_PER_MINUTE)
    for _ in range(10):
        pacer.reserve(0)
    assert clock.slept == []

    # And the window is still untouched, so real work after it is not paced
    # against documents that were never sent.
    pacer.reserve(GEMINI_DOCUMENTS_PER_MINUTE)
    assert clock.slept == []


def test_crossing_the_limit_waits_out_the_minute(clock):
    """96 documents through, then a batch that would make 102 -- which is
    exactly what killed the first eval run."""
    pacer = _document_pacer(GEMINI_DOCUMENTS_PER_MINUTE)
    for _ in range(3):
        pacer.reserve(32)
    pacer.reserve(6)

    assert len(clock.slept) == 1
    # Nothing has taken any time on the fake clock, so it waits out the window.
    assert clock.slept[0] >= 60


def test_the_window_resets_after_the_wait(clock):
    pacer = _document_pacer(GEMINI_DOCUMENTS_PER_MINUTE)
    for _ in range(3):
        pacer.reserve(32)
    pacer.reserve(6)
    clock.slept.clear()

    # A fresh window: another 94 should pass without waiting again.
    for _ in range(2):
        pacer.reserve(32)
    pacer.reserve(30)
    assert clock.slept == []


def test_a_minute_passing_on_its_own_resets_the_window(clock):
    pacer = _document_pacer(GEMINI_DOCUMENTS_PER_MINUTE)
    for _ in range(3):
        pacer.reserve(32)

    clock.now += 61  # the eval spent a minute doing something else
    pacer.reserve(32)
    assert clock.slept == []


def test_only_the_overflowing_batch_waits(clock):
    pacer = _document_pacer(GEMINI_DOCUMENTS_PER_MINUTE)
    for _ in range(6):
        pacer.reserve(32)
    # 192 documents is two windows' worth, so exactly one wait.
    assert len(clock.slept) == 1


# =============================================================================
# CHAT CALLS ARE PACED PER MODEL
# =============================================================================


def test_a_model_keeps_one_pacer_across_providers():
    """The point of holding these in the module rather than on the provider.

    answer_question builds a fresh chat provider for every question, so a pacer
    living on the instance would start a new minute each time and pace nothing.
    """
    assert chat_pacer("some-model") is chat_pacer("some-model")


def test_chat_and_judge_do_not_share_a_count():
    """Quotas are per project per model. Counting them together would pace two
    separate allowances as though they were one."""
    assert chat_pacer("chat-model") is not chat_pacer("judge-model")


def test_chat_stays_under_its_limit_a_minute(clock):
    pacer = Pacer(CHAT_REQUESTS_PER_MINUTE, "requests")
    for _ in range(CHAT_REQUESTS_PER_MINUTE):
        pacer.reserve(1)
    assert clock.slept == []

    # The one after that is where the window has to roll.
    pacer.reserve(1)
    assert len(clock.slept) == 1
    assert clock.slept[0] >= 60


def test_an_eval_run_of_two_calls_a_question_waits_once_on_gemini(clock):
    """Nine golden questions, two chat calls each -- reading the question and
    writing the answer -- is 18 in a burst against Gemini's limit of 12. One
    waited window, the same as it cost at 15."""
    pacer = Pacer(CHAT_RPM_BY_PROVIDER["gemini"], "requests")
    for _ in range(18):
        pacer.reserve(1)
    assert len(clock.slept) == 1


def test_the_same_eval_run_never_waits_on_openai(clock):
    """Which is the point of the retune: the pacer stops being the thing the
    eval is waiting on, without stopping being a pacer."""
    pacer = Pacer(CHAT_RPM_BY_PROVIDER["openai"], "requests")
    for _ in range(18):
        pacer.reserve(1)
    assert clock.slept == []


# =============================================================================
# THE 429 BACKSTOP
# =============================================================================

REAL_429 = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded "
    "your current quota', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': "
    "'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '14.044580725s'}]}}"
)


class FakeError(Exception):
    def __init__(self, message, code=429, details=None):
        super().__init__(message)
        self.code = code
        self.details = details


def test_a_quota_error_is_recognised_by_code_and_by_text():
    assert _is_quota_error(FakeError("boom", code=429))
    assert _is_quota_error(FakeError("RESOURCE_EXHAUSTED", code=None))


def test_other_failures_are_not_treated_as_quota():
    assert not _is_quota_error(FakeError("500 INTERNAL", code=500))


def test_the_servers_own_retry_delay_is_used():
    """It tells us how long to wait. Guessing shorter just gets refused again."""
    assert _retry_after(FakeError(REAL_429)) == pytest.approx(14.044580725)


def test_retry_delay_is_read_from_structured_details_too():
    error = FakeError("429", details=[
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "7s"},
    ])
    assert _retry_after(error) == pytest.approx(7.0)


def test_no_retry_delay_means_no_guess_from_the_parser():
    assert _retry_after(FakeError("429 quota exceeded, no advice given")) is None


# =============================================================================
# THE 5XX BACKSTOP
# =============================================================================

REAL_503 = (
    "503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is "
    "currently experiencing high demand. Spikes in demand are usually "
    "temporary. Please try again later.', 'status': 'UNAVAILABLE'}}"
)


def test_a_503_is_recognised_as_a_server_error():
    assert _is_server_error(FakeError(REAL_503, code=503))


def test_server_errors_are_recognised_by_status_when_there_is_no_code():
    for status in ("UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED"):
        assert _is_server_error(FakeError(status, code=None)), status


def test_a_quota_refusal_is_not_a_server_error():
    """They get separate budgets, so they must not both match."""
    error = FakeError(REAL_429, code=429)
    assert _is_quota_error(error)
    assert not _is_server_error(error)


def test_a_404_is_neither_and_must_not_be_retried():
    dead_model = FakeError(
        "404 NOT_FOUND. This model models/gemini-2.5-flash is no longer "
        "available to new users.", code=404,
    )
    assert not _is_quota_error(dead_model)
    assert not _is_server_error(dead_model)


def test_server_backoff_doubles():
    delays = [_server_delay(FakeError(REAL_503, code=503), n) for n in range(1, 5)]
    assert delays == [2, 4, 8, 16]


def test_server_backoff_is_capped():
    assert _server_delay(FakeError(REAL_503, code=503), 20) == SERVER_BACKOFF_CAP_SECONDS


def test_the_server_budget_is_about_eight_minutes_of_patience():
    """Long enough to outlast a demand spike, short enough that a real outage
    still fails the run rather than hanging CI.

    It was a minute, and a minute kept losing runs to spikes that passed. The
    upper bound is the half of that trade worth guarding: this is spent per
    call, so it is the number that decides how long a dead model hangs the job.
    """
    total = sum(
        _server_delay(FakeError(REAL_503, code=503), n)
        for n in range(1, SERVER_RETRY_ATTEMPTS)
    )
    assert 420 <= total <= 600


def test_the_cap_is_what_keeps_the_last_attempts_from_running_away():
    """Without it the ninth wait alone would be over eight minutes, and the
    budget above would be nearer twenty than eight."""
    uncapped = SERVER_BACKOFF_BASE_SECONDS * (2 ** (SERVER_RETRY_ATTEMPTS - 2))
    assert uncapped > SERVER_BACKOFF_CAP_SECONDS
    assert _server_delay(
        FakeError(REAL_503, code=503), SERVER_RETRY_ATTEMPTS - 1
    ) == SERVER_BACKOFF_CAP_SECONDS


def test_the_two_policies_do_not_share_a_delay_calculation():
    """A 503 carries no retryDelay, so honouring one would mean guessing."""
    server = FakeError(REAL_503, code=503)
    assert _retry_after(server) is None
    assert _server_delay(server, 1) == 2

    quota = FakeError(REAL_429, code=429)
    assert _quota_delay(quota, 1) == pytest.approx(15.044580725)


# =============================================================================
# PER-DAY QUOTAS ARE NOT WORTH WAITING FOR
# =============================================================================

# Both taken verbatim from real CI failures.
DAILY_429 = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded "
    "your current quota. * Quota exceeded for metric: "
    "generativelanguage.googleapis.com/generate_content_free_tier_requests, "
    "limit: 20, model: gemini-3.6-flash. Please retry in 58.993147613s.', "
    "'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': "
    "'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaId': "
    "'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}]}, {'@type': "
    "'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '58s'}]}}"
)

PER_MINUTE_429 = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'details': [{'@type': "
    "'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaId': "
    "'EmbedContentRequestsPerMinutePerUserPerProjectPerModel-FreeTier'}]}, "
    "{'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': "
    "'14.044580725s'}]}}"
)


def test_the_daily_quota_id_is_read_from_the_message():
    assert _quota_ids(FakeError(DAILY_429, code=429)) == [
        "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
    ]


def test_the_daily_quota_id_is_read_from_structured_details():
    error = FakeError("429", details=[{
        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
        "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}],
    }])
    assert _is_daily_quota(error)


def test_a_per_day_refusal_is_fatal():
    assert _is_daily_quota(FakeError(DAILY_429, code=429))


def test_a_per_minute_refusal_is_not():
    """This one is worth sitting out -- it is the embedding quota that pacing
    already handles, and it really does come back within the minute."""
    assert not _is_daily_quota(FakeError(PER_MINUTE_429, code=429))


def test_a_daily_refusal_still_looks_like_a_quota_error():
    """It is still a 429; it just is not one worth waiting on."""
    assert _is_quota_error(FakeError(DAILY_429, code=429))


def test_the_misleading_retry_delay_is_ignored_for_daily_quotas():
    """The refusal advises 58 seconds for an allowance that returns tomorrow.
    We can still read it -- we simply must not act on it."""
    error = FakeError(DAILY_429, code=429)
    assert _retry_after(error) == pytest.approx(58.0)
    assert _is_daily_quota(error)


def test_an_unknown_quota_id_is_not_assumed_daily():
    assert not _is_daily_quota(FakeError("429 RESOURCE_EXHAUSTED, no detail", code=429))
