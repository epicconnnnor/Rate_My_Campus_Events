"""
DEMO_DATE, the setting that lets a local run stand inside the frozen events.

Only the parsing is here. What uses it lives in the retriever, and is tested in
test_retriever.py where the rest of the retrieval path already is.
"""

import importlib
from datetime import date

import pytest

from app.core import config


def reload_with(monkeypatch, value):
    """config reads the environment at import, so the module is reloaded."""
    if value is None:
        monkeypatch.delenv("DEMO_DATE", raising=False)
    else:
        monkeypatch.setenv("DEMO_DATE", value)
    return importlib.reload(config)


@pytest.fixture(autouse=True)
def restore_config(monkeypatch):
    """Put the module back however this test found it.

    Everything else importing app.core.config gets the reloaded module, so a
    test that leaves DEMO_DATE set would quietly move another test's today.
    """
    yield
    monkeypatch.delenv("DEMO_DATE", raising=False)
    importlib.reload(config)


def test_unset_means_the_real_date():
    """The deployed case. Nothing should have to be set to get today."""
    assert config.DEMO_DATE is None or isinstance(config.DEMO_DATE, date)


def test_an_unset_variable_leaves_it_none(monkeypatch):
    assert reload_with(monkeypatch, None).DEMO_DATE is None


def test_an_empty_string_is_treated_as_unset(monkeypatch):
    """An unset CI or compose variable arrives as "" rather than as absent."""
    assert reload_with(monkeypatch, "").DEMO_DATE is None
    assert reload_with(monkeypatch, "   ").DEMO_DATE is None


def test_an_iso_date_is_read(monkeypatch):
    assert reload_with(monkeypatch, "2026-09-02").DEMO_DATE == date(2026, 9, 2)


def test_surrounding_space_is_forgiven(monkeypatch):
    """Copied out of the README with a trailing space is not a typo worth
    refusing to start over."""
    assert reload_with(monkeypatch, " 2026-09-02 ").DEMO_DATE == date(2026, 9, 2)


@pytest.mark.parametrize("value", [
    "2026-13-01",     # no thirteenth month
    "02/09/2026",     # not ISO
    "next tuesday",   # not a date at all
    "2026-09",        # not a day
])
def test_something_that_is_not_a_date_says_so_loudly(monkeypatch, value):
    """Silently ignoring a typo is the cruellest option: the app comes up,
    every question returns nothing, and the calendar looks broken rather than
    the setting looking wrong."""
    with pytest.raises(RuntimeError) as raised:
        reload_with(monkeypatch, value)

    message = str(raised.value)
    assert "DEMO_DATE" in message
    assert "YYYY-MM-DD" in message


def test_the_complaint_names_a_date_that_would_work(monkeypatch):
    """A refusal that does not say what to type instead is half a message."""
    with pytest.raises(RuntimeError) as raised:
        reload_with(monkeypatch, "nonsense")

    assert "2026-09-02" in str(raised.value)
