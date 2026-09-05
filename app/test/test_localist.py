from app.ingest.localist import parse_event
from app.core.event_description import render_event_description


def test_localist_descriptions_keep_paragraph_breaks():
    event = parse_event({
        "id": 1,
        "title": "Sample event",
        "description": "<p>First paragraph.</p><p>Second paragraph.</p>",
        "event_instances": [],
    })

    assert event["description"] == "First paragraph.\n\nSecond paragraph."


def test_localist_descriptions_fall_back_to_plain_text():
    event = parse_event({
        "id": 1,
        "title": "Sample event",
        "description_text": "A plain description",
        "event_instances": [],
    })

    assert event["description"] == "A plain description"


def test_localist_descriptions_keep_embedded_links_clickable():
    event = parse_event({
        "id": 1,
        "title": "Sample event",
        "description": '<p><a href="https://example.com/signup">Sign up here</a></p>',
        "event_instances": [],
    })

    assert event["description"] == "[Sign up here](https://example.com/signup)"
    rendered = render_event_description(event["description"])
    assert 'href="https://example.com/signup"' in rendered
    assert "Sign up here</a>" in rendered
