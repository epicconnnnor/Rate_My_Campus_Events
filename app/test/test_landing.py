"""
The front page.

It is the one screen that has to work with no session, no token and no
database. Somebody arriving cold sees this before anything else, so a 500
here is worse than a 500 anywhere in the app.
"""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    """The app with no user and no database.

    SECRET_KEY is patched onto the config module rather than into the
    environment: config reads it once at import and require_secret_key()
    checks that module global, so setenv would be too late. Nothing here is
    signed -- no request in this file carries a token.
    """
    import app.core.config as config
    monkeypatch.setattr(config, "SECRET_KEY", "not-a-real-key-nothing-is-signed")

    from app.main import app
    return TestClient(app)


@pytest.fixture
def no_database(monkeypatch):
    """A database that refuses to connect, which is the honest local default."""
    import app.api.routes_events as routes

    def refuse():
        raise OSError("no database here")

    monkeypatch.setattr(routes, "engine", SimpleNamespace(connect=refuse))


# =============================================================================
# IT WORKS SIGNED OUT
# =============================================================================


def test_the_front_page_answers_without_a_session(client, no_database):
    response = client.get("/")
    assert response.status_code == 200
    assert "RateMyCampusEvents" in response.text


def test_a_visitor_with_no_account_is_pointed_at_one(client, no_database):
    """The call to action is the whole reason the page exists."""
    body = client.get("/").text
    assert 'href="/register"' in body
    assert "Get started" in body
    assert 'href="/login"' in body


def test_a_signed_in_visitor_is_pointed_at_the_chat(client, no_database,
                                                    monkeypatch):
    from app.api.routes_events import get_current_user
    from app.main import app

    app.dependency_overrides[get_current_user] = lambda: {"name": "Someone"}
    try:
        body = client.get("/").text
    finally:
        app.dependency_overrides.pop(get_current_user, None)

    assert "Open the chat" in body
    assert "Get started" not in body


def test_the_page_never_asks_for_a_login_first(client, no_database):
    """A redirect to /login here would put a wall in front of the first
    screen anybody sees."""
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 200


# =============================================================================
# THE NUMBERS ARE REAL OR THEY ARE ABSENT
# =============================================================================


def test_real_counts_are_shown_when_the_calendar_can_be_read(client,
                                                             monkeypatch):
    import app.api.routes_events as routes

    monkeypatch.setattr(routes, "_landing_stats",
                        lambda: {"events": 102, "categories": 17})
    body = client.get("/").text

    assert "102" in body
    assert "events indexed" in body
    assert "17" in body
    assert "event categories" in body


def test_an_unreadable_count_is_left_out_rather_than_shown_as_zero(client,
                                                                   no_database):
    """A confident 0 reads as "this thing indexes nothing". Saying nothing at
    all is the honest failure."""
    body = client.get("/").text

    assert "events indexed" not in body
    assert "event categories" not in body
    # The claim that needs no database still stands.
    assert "grounded in the official calendar" in body


def test_a_broken_database_does_not_break_the_page(client, no_database):
    assert client.get("/").status_code == 200


# =============================================================================
# THE LIST MOVED, AND EVERYTHING STILL POINTS AT IT
# =============================================================================


def test_the_event_list_is_reachable_from_the_front_page(client, no_database):
    assert 'href="/events' in client.get("/").text


def test_the_front_page_is_not_the_event_list(client, no_database):
    """They were the same URL. Somebody arriving now gets the pitch, and the
    list is one click away rather than the first thing they parse."""
    body = client.get("/").text
    assert "event-grid" not in body
