"""
Tests for staying inside the embedding quota.

Nothing here sleeps for real: the clock is a fake so a minute costs nothing.
"""

import pytest

from app.rag import indexer
from app.rag.indexer import FREE_TIER_DOCUMENTS_PER_MINUTE, _Pacer
from app.rag.providers import _is_quota_error, _retry_after


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
    monkeypatch.setattr(indexer, "time", fake)
    return fake


# =============================================================================
# PACING
# =============================================================================


def test_the_limit_is_googles_number_not_ours():
    assert FREE_TIER_DOCUMENTS_PER_MINUTE == 100


def test_staying_under_the_limit_never_waits(clock):
    pacer = _Pacer()
    for _ in range(3):
        pacer.reserve(32)
    assert clock.slept == []


def test_crossing_the_limit_waits_out_the_minute(clock):
    """96 documents through, then a batch that would make 102 -- which is
    exactly what killed the first eval run."""
    pacer = _Pacer()
    for _ in range(3):
        pacer.reserve(32)
    pacer.reserve(6)

    assert len(clock.slept) == 1
    # Nothing has taken any time on the fake clock, so it waits out the window.
    assert clock.slept[0] >= 60


def test_the_window_resets_after_the_wait(clock):
    pacer = _Pacer()
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
    pacer = _Pacer()
    for _ in range(3):
        pacer.reserve(32)

    clock.now += 61  # the eval spent a minute doing something else
    pacer.reserve(32)
    assert clock.slept == []


def test_only_the_overflowing_batch_waits(clock):
    pacer = _Pacer()
    for _ in range(6):
        pacer.reserve(32)
    # 192 documents is two windows' worth, so exactly one wait.
    assert len(clock.slept) == 1


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
