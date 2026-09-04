"""
What the chat box says when it cannot answer.

There are two ways to fail and they want opposite handling. A model that is
busy or broken is weather: say something calm, suggest waiting. A missing key
is answerable by whoever is running the app and by nobody else, so telling
them to try again in a minute is advice that cannot work -- it will fail
identically forever.

That distinction is the whole test. Six identical "something went wrong
reaching the assistant" pages, every one of them a RuntimeError raised in a
provider constructor before a single request was sent, is what these lock down.

No database and no model: the auth dependency is overridden, the usage counter
is a stub, and answer_question is replaced with something that raises.
"""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.routes_chat import BROKEN, MISCONFIGURED_PREFIX
from app.api.routes_events import get_current_user
from app.rag.providers import ProviderNotConfigured

USER = {"user_id": 1, "name": "Test Person", "email": "test@example.com"}


@pytest.fixture
def client(monkeypatch):
    """The app with its edges stubbed: signed in, under quota, no model.

    app.main calls require_secret_key() at import, so importing it without one
    raises before any test body runs. The unit job deliberately sets no
    secrets -- it is the suite that needs neither a key nor a database -- so
    the value is supplied here rather than added to the workflow. It signs
    nothing: no request in this file carries a token.

    Patched on the config module rather than in the environment. config reads
    SECRET_KEY once at import and require_secret_key() checks that module
    global, so by the time any test runs the value has already been read and
    setenv would be shouting at a closed door.
    """
    import app.core.config as config
    monkeypatch.setattr(config, "SECRET_KEY",
                        "not-a-real-key-nothing-here-is-signed")

    import app.api.routes_chat as routes
    from app.main import app

    monkeypatch.setattr(routes, "db", SimpleNamespace(
        get_chat_request_count=lambda user_id, day: 1,
        record_chat_request=lambda user_id, day: 2,
    ))
    # The chips read the calendar's own event types. No database here.
    monkeypatch.setattr(routes, "_categories", lambda *a, **k: [])

    app.dependency_overrides[get_current_user] = lambda: USER
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_current_user, None)


def ask(client, question="anything on friday?"):
    return client.post("/chat", data={"question": question})


# =============================================================================
# A MISSING KEY NAMES ITSELF
# =============================================================================


def test_a_missing_key_is_reported_rather_than_hidden(client, monkeypatch):
    import app.api.routes_chat as routes

    def unconfigured(*args, **kwargs):
        raise ProviderNotConfigured(
            "OPENAI_API_KEY is not set. Export it, or set "
            "RAG_PROVIDER=gemini with a GEMINI_API_KEY, or "
            "RAG_PROVIDER=fake for canned replies."
        )

    monkeypatch.setattr(routes, "answer_question", unconfigured)
    response = ask(client)

    assert response.status_code == 503
    body = response.text
    assert MISCONFIGURED_PREFIX in body
    # The part that was missing before: which variable, and what to do.
    assert "OPENAI_API_KEY" in body
    assert "RAG_PROVIDER=fake" in body
    # And not the advice that cannot work.
    assert BROKEN not in body


def test_a_configuration_problem_is_not_dressed_up_as_weather(client,
                                                              monkeypatch):
    """503 rather than 502: nothing upstream misbehaved, we never asked it."""
    import app.api.routes_chat as routes

    monkeypatch.setattr(routes, "answer_question", _raises(
        ProviderNotConfigured("RAG_PROVIDER is 'opanai', which is not a provider.")
    ))
    response = ask(client)

    assert response.status_code == 503
    assert "opanai" in response.text


# =============================================================================
# EVERYTHING ELSE STAYS VAGUE
# =============================================================================


def test_a_real_failure_still_says_nothing_specific(client, monkeypatch):
    """The detail here is a stack trace, and the log is where it belongs.
    Whatever a model put in an exception string is not for a stranger."""
    import app.api.routes_chat as routes

    monkeypatch.setattr(routes, "answer_question", _raises(
        RuntimeError("connection reset by peer at 10.0.0.4:5432")
    ))
    response = ask(client)

    assert response.status_code == 502
    assert BROKEN in response.text
    assert "10.0.0.4" not in response.text
    assert MISCONFIGURED_PREFIX not in response.text


def test_a_server_error_from_the_model_is_still_weather(client, monkeypatch):
    """A 503 from the provider has already been retried to exhaustion by the
    time it reaches here. It is the case the calm message was written for."""
    import app.api.routes_chat as routes

    monkeypatch.setattr(routes, "answer_question", _raises(
        Exception("503 UNAVAILABLE. The model is overloaded.")
    ))
    response = ask(client)

    assert response.status_code == 502
    assert BROKEN in response.text


def test_an_empty_question_is_neither_kind_of_failure(client):
    response = ask(client, question="   ")
    assert response.status_code == 200
    assert "Ask me something first." in response.text
    assert MISCONFIGURED_PREFIX not in response.text
    assert BROKEN not in response.text


def _raises(error):
    def fail(*args, **kwargs):
        raise error
    return fail
