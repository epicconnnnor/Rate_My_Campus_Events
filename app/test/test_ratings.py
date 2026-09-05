from app.api.routes_events import add_rating_summary


def test_upvotes_are_shown_as_a_score_out_of_ten():
    event = {}

    add_rating_summary(event, [{"value": 1}, {"value": 1}, {"value": -1}])

    assert event["rating_score"] == 6.7
    assert event["thumbs_up_percent"] == 67
    assert event["thumbs_down_percent"] == 33


def test_an_unrated_event_has_no_misleading_score():
    event = {}

    add_rating_summary(event, [])

    assert event["rating_score"] is None
