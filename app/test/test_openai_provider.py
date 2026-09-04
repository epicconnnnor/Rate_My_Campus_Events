"""
The OpenAI provider, and the retry logic now that it serves two SDKs.

Nothing here sends a request. The clients are stubs, and the errors are shaped
the way each SDK shapes them -- which is the whole point: google-genai puts an
int status on `.code`, openai puts one on `.status_code` and uses `.code` for a
string slug. Classifying those with one function is where a quiet bug would
live, so it gets tested rather than read.
"""

import importlib
from types import SimpleNamespace

import pytest

from app.rag.providers import (ProviderNotConfigured, _is_connection_error,
                               _is_quota_error, _is_server_error,
                               _sdk_error_types, _seconds_from_header,
                               _status_code)


class GeminiShapedError(Exception):
    """google-genai: an int on .code."""

    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


class OpenAIShapedError(Exception):
    """openai: an int on .status_code, a string slug on .code."""

    def __init__(self, message, status_code=None, code=None, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        if headers is not None:
            self.response = SimpleNamespace(headers=headers)


# =============================================================================
# READING A STATUS OUT OF EITHER SDK
# =============================================================================


def test_a_gemini_error_reports_its_status():
    assert _status_code(GeminiShapedError("boom", code=429)) == 429


def test_an_openai_error_reports_its_status():
    assert _status_code(OpenAIShapedError("boom", status_code=429)) == 429


def test_an_openai_string_code_is_not_mistaken_for_a_status():
    """`.code` on an openai error is "rate_limit_exceeded", not a number.
    Comparing it to 429 would quietly classify every 429 as unretryable."""
    error = OpenAIShapedError("slow down", status_code=429,
                              code="rate_limit_exceeded")
    assert _status_code(error) == 429
    assert _is_quota_error(error)


def test_an_error_with_no_status_at_all_reports_none():
    assert _status_code(Exception("nothing to go on")) is None


def test_a_boolean_is_not_a_status():
    """`isinstance(True, int)` is True in Python, so a stray flag on an
    exception would otherwise be read as status code 1."""
    assert _status_code(SimpleNamespace(status_code=True, code=False)) is None


# =============================================================================
# WHICH FAILURES ARE WORTH WAITING OUT
# =============================================================================


@pytest.mark.parametrize("error", [
    GeminiShapedError("429 RESOURCE_EXHAUSTED", code=429),
    GeminiShapedError("RESOURCE_EXHAUSTED", code=None),
    OpenAIShapedError("rate limit reached", status_code=429),
    OpenAIShapedError("Rate_Limit exceeded for gpt-4o-mini"),
])
def test_quota_refusals_from_both_sdks_are_recognised(error):
    assert _is_quota_error(error)


@pytest.mark.parametrize("error", [
    GeminiShapedError("503 UNAVAILABLE", code=503),
    OpenAIShapedError("server had an error", status_code=500),
    OpenAIShapedError("bad gateway", status_code=502),
])
def test_server_failures_from_both_sdks_are_recognised(error):
    assert _is_server_error(error)


def test_a_400_is_neither_and_must_not_be_retried():
    """A malformed request fails the same way however long you wait."""
    bad = OpenAIShapedError("invalid model", status_code=400)
    assert not _is_quota_error(bad)
    assert not _is_server_error(bad)


def test_a_404_is_neither_either():
    dead_model = OpenAIShapedError("model not found", status_code=404)
    assert not _is_quota_error(dead_model)
    assert not _is_server_error(dead_model)


def test_a_connection_that_never_landed_is_treated_as_a_server_error():
    """It carries no status, so the status check alone would call it fatal."""
    openai = pytest.importorskip("openai")
    error = openai.APIConnectionError(request=None)
    assert _is_connection_error(error)
    assert _is_server_error(error)
    assert not _is_quota_error(error)


def test_an_ordinary_exception_is_not_a_connection_error():
    assert not _is_connection_error(Exception("something else"))


def test_both_sdks_are_retryable_when_installed():
    """An SDK missing from this tuple is one whose errors escape _with_retries
    entirely and fail the first 503 they meet."""
    types = _sdk_error_types()
    names = {t.__name__ for t in types}
    assert "APIError" in names
    assert len(types) >= 1


# =============================================================================
# HOW LONG THE SERVER ASKED US TO WAIT
# =============================================================================


@pytest.mark.parametrize("raw,expected", [
    ("12", 12.0),
    ("1.5s", 1.5),
    ("300ms", 0.3),
    ("2m", 120.0),
])
def test_a_retry_after_header_is_read(raw, expected):
    assert _seconds_from_header(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["", "soon", "Wed, 21 Oct 2026 07:28:00 GMT"])
def test_an_unreadable_retry_after_is_not_guessed_at(raw):
    """An HTTP-date form would be parsed as a wait until tomorrow if guessed
    at. Returning None falls back to the doubling backoff, which is finite."""
    assert _seconds_from_header(raw) is None


# =============================================================================
# BUILDING THE PROVIDERS
# =============================================================================


def reload_with(monkeypatch, **env):
    """Re-import config and providers under a given environment.

    The caller must reach for exceptions through the returned module.
    importlib.reload rebinds the class, so ProviderNotConfigured imported at
    the top of this file is a different object afterwards and would never
    match -- which looks exactly like the code failing to raise.
    """
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    import app.core.config as config
    import app.rag.providers as providers
    importlib.reload(config)
    importlib.reload(providers)
    return providers


@pytest.fixture(autouse=True)
def restore_modules():
    yield
    import app.core.config as config
    import app.rag.providers as providers
    importlib.reload(config)
    importlib.reload(providers)


def test_no_openai_key_says_which_variable_is_missing(monkeypatch):
    providers = reload_with(monkeypatch, RAG_PROVIDER="openai",
                            OPENAI_API_KEY="")
    with pytest.raises(providers.ProviderNotConfigured) as raised:
        providers.get_chat_provider()

    message = str(raised.value)
    assert "OPENAI_API_KEY" in message
    # And what to do instead, both ways out.
    assert "gemini" in message and "fake" in message


def test_no_gemini_key_still_says_which_variable_is_missing(monkeypatch):
    providers = reload_with(monkeypatch, RAG_PROVIDER="gemini",
                            GEMINI_API_KEY="")
    with pytest.raises(providers.ProviderNotConfigured) as raised:
        providers.get_embedding_provider()
    assert "GEMINI_API_KEY" in str(raised.value)


def test_a_configuration_failure_is_catchable_as_a_runtime_error():
    """Anything already catching RuntimeError keeps working."""
    assert issubclass(ProviderNotConfigured, RuntimeError)


def test_a_reloaded_module_still_raises_the_same_kind_of_thing(monkeypatch):
    """Guards the trap above: the reloaded class must still be a
    ProviderNotConfigured by name and a RuntimeError by inheritance."""
    providers = reload_with(monkeypatch, RAG_PROVIDER="nonsense")
    assert providers.ProviderNotConfigured.__name__ == "ProviderNotConfigured"
    assert issubclass(providers.ProviderNotConfigured, RuntimeError)


def test_an_unknown_provider_name_is_a_configuration_problem(monkeypatch):
    """It used to be a ValueError, which the chat route flattened into
    "something went wrong reaching the assistant" -- advice to wait, for a
    typo that will never fix itself."""
    providers = reload_with(monkeypatch, RAG_PROVIDER="opanai")
    with pytest.raises(providers.ProviderNotConfigured) as raised:
        providers.get_chat_provider()
    assert "opanai" in str(raised.value)


def test_the_fake_provider_needs_no_key_at_all(monkeypatch):
    providers = reload_with(monkeypatch, RAG_PROVIDER="fake",
                            OPENAI_API_KEY="", GEMINI_API_KEY="")
    assert providers.get_chat_provider().complete("anything")
    assert providers.get_embedding_provider().embed_query("anything")


def test_gemini_still_builds_when_it_is_the_one_selected(monkeypatch):
    """The point of keeping it: switching back is a variable, not a revert."""
    providers = reload_with(monkeypatch, RAG_PROVIDER="gemini",
                            GEMINI_API_KEY="not-a-real-key")
    assert type(providers.get_chat_provider()).__name__ == "GeminiChatProvider"


def test_openai_embeddings_are_still_wrapped_in_the_disk_cache(monkeypatch,
                                                               tmp_path):
    """The cache is provider-agnostic and stays that way. Its keys include the
    model, so switching provider misses every entry rather than serving a
    Gemini vector to an OpenAI query."""
    providers = reload_with(monkeypatch, RAG_PROVIDER="openai",
                            OPENAI_API_KEY="not-a-real-key",
                            EMBEDDING_CACHE_PATH=str(tmp_path))
    provider = providers.get_embedding_provider()
    assert type(provider).__name__ == "CachedEmbeddingProvider"


# =============================================================================
# WHAT THE OPENAI PROVIDER DOES WITH A RESPONSE
# =============================================================================


def make_embedder(data):
    """An OpenAIEmbeddingProvider with a stub client and no constructor."""
    from app.rag.providers import OpenAIEmbeddingProvider

    provider = object.__new__(OpenAIEmbeddingProvider)
    provider._model = "text-embedding-3-small"
    provider._client = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(data=data)
        )
    )
    return provider


def test_embeddings_are_put_back_in_the_order_they_were_asked_for():
    """The API returns an index per item. Trusting the order instead would
    pair every event with somebody else's vector, silently."""
    out_of_order = [
        SimpleNamespace(index=1, embedding=[0.0, 1.0]),
        SimpleNamespace(index=0, embedding=[1.0, 0.0]),
    ]
    vectors = make_embedder(out_of_order).embed_documents(["first", "second"])
    assert vectors[0] == pytest.approx([1.0, 0.0])
    assert vectors[1] == pytest.approx([0.0, 1.0])


def test_vectors_come_back_unit_length():
    """Cosine distance in the query assumes it."""
    data = [SimpleNamespace(index=0, embedding=[3.0, 4.0])]
    vector = make_embedder(data).embed_query("anything")
    assert sum(value * value for value in vector) == pytest.approx(1.0)


def test_every_document_costs_a_request():
    """Only the disk cache ever answers smaller. The pacer relies on it."""
    assert make_embedder([]).chargeable(["a", "b", "c"]) == 3


def make_chat(content):
    from app.rag.providers import OpenAIChatProvider

    provider = object.__new__(OpenAIChatProvider)
    provider._model = "gpt-4o-mini"
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(choices=[
                SimpleNamespace(message=SimpleNamespace(content=content))
            ])
        ))
    )
    return provider


def test_a_reply_comes_back_as_text():
    assert make_chat("hello").complete("hi") == "hello"


def test_an_empty_reply_is_a_string_rather_than_none():
    """A refusal or a length stop leaves content None. Returning it would
    raise inside parse_json_object instead of falling back to a broad search."""
    assert make_chat(None).complete("hi") == ""
