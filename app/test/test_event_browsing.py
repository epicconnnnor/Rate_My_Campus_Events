from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    import app.core.config as config
    monkeypatch.setattr(config, "SECRET_KEY", "test-only-signing-key")
    from app.main import app
    from app.api import routes_events as routes
    events = [
        dict(event_id=1, title="Later concert", starts_at=datetime(2026, 10, 2, tzinfo=timezone.utc),
             event_types=["Music"], is_free=True),
        dict(event_id=2, title="Early concert", date_time="2026-09-04T19:00",
             event_types=["Music"], is_free=False),
        dict(event_id=3, title="Undated workshop", date_time="TBA",
             event_types=["Workshop"], is_free=True),
    ]
    monkeypatch.setattr(routes.db, "list_events", lambda: [dict(event) for event in events])
    monkeypatch.setattr(routes.db, "get_reactions_for_event", lambda _: [])
    return TestClient(app)


def test_chronological_order_and_missing_metadata(client):
    response = client.get("/events")
    assert response.status_code == 200
    events = response.context["events"]
    assert [event["event_id"] for event in events] == [2, 1, 3]
    assert events[1]["month_label"] == "October 2026"
    assert "Location to be announced" in response.text
    assert "Dates to be announced" in response.text


def test_combined_filters_and_empty_date_submission(client):
    response = client.get("/events", params=dict(q=" CONCERT ", category="Music", free="1", from_date=""))
    assert response.status_code == 200
    assert [event["event_id"] for event in response.context["events"]] == [1]
    assert response.context["categories"] == ["Music", "Workshop"]


def test_date_filter_uses_campus_day_and_excludes_unknown_dates(client):
    # The October event is October 1 in Amherst.
    response = client.get("/events?from_date=2026-10-02")
    assert response.status_code == 200
    assert response.context["events"] == []
    assert "No events found" in response.text
    assert client.get("/events?from_date=not-a-date").status_code == 422


def test_public_navigation_and_signed_in_links(client):
    from app.main import app
    from app.api.routes_events import get_current_user
    assert 'href="/chat"' in client.get("/events").text
    app.dependency_overrides[get_current_user] = lambda: {"name": "Test"}
    try:
        body = client.get("/events").text
        assert 'href="/"' in body
        assert 'href="/chat"' in body
        assert "?token=" not in body
    finally:
        app.dependency_overrides.pop(get_current_user, None)
