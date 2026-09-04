"""
Tests for which model each job talks to.

Two things have to hold, on every provider. The default ids belong to the
provider that is selected -- a Gemini id means nothing to OpenAI and the other
way round. And chat and judge are different models, so the judge is never
marking its own writing, and on Gemini they draw on separate daily allowances
as well. These check that both are real and not merely configured.
"""

import importlib

import pytest


def reload_with(monkeypatch, **env):
    """Re-import config and providers under a given environment.

    Both read their settings at import, so the environment has to be in place
    before the module object exists.
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
    return config, providers


@pytest.fixture(autouse=True)
def restore_modules():
    """Leave the imported modules as the rest of the suite expects them."""
    yield
    import app.core.config as config
    import app.rag.providers as providers
    importlib.reload(config)
    importlib.reload(providers)


def test_the_judge_gets_its_own_model_when_one_is_set(monkeypatch):
    config, _ = reload_with(
        monkeypatch, CHAT_MODEL="model-a", JUDGE_MODEL="model-b"
    )
    assert config.CHAT_MODEL == "model-a"
    assert config.JUDGE_MODEL == "model-b"


def defaults_for(monkeypatch, provider):
    return reload_with(monkeypatch, RAG_PROVIDER=provider, CHAT_MODEL=None,
                       JUDGE_MODEL=None, EMBEDDING_MODEL=None)[0]


def test_the_openai_defaults_are_openai_ids(monkeypatch):
    config = defaults_for(monkeypatch, "openai")
    assert config.CHAT_MODEL == "gpt-4o-mini"
    assert config.JUDGE_MODEL == "gpt-4.1-mini"
    assert config.EMBEDDING_MODEL == "text-embedding-3-small"


def test_the_gemini_defaults_are_the_lite_models(monkeypatch):
    """The non-lite flash models allow 20 requests a day, which one eval run
    spends entirely. Both defaults have to be lite ids for the suite to be
    runnable more than once a day."""
    config = defaults_for(monkeypatch, "gemini")
    assert config.CHAT_MODEL == "gemini-3.1-flash-lite"
    assert config.JUDGE_MODEL == "gemini-3.5-flash-lite"
    assert config.EMBEDDING_MODEL == "gemini-embedding-001"


def test_openai_is_what_you_get_without_choosing(monkeypatch):
    config = reload_with(monkeypatch, RAG_PROVIDER=None)[0]
    assert config.RAG_PROVIDER == "openai"


def test_no_provider_borrows_another_providers_ids(monkeypatch):
    """The failure this stops is a Gemini id sent to OpenAI, which is a 404
    several layers down inside somebody's question rather than a setup error.

    The values are copied out rather than compared across two module handles:
    reloading returns the same module object both times, so holding onto it
    would compare the second provider against itself and pass regardless.
    """
    fields = ("CHAT_MODEL", "JUDGE_MODEL", "EMBEDDING_MODEL")
    openai = {f: getattr(defaults_for(monkeypatch, "openai"), f) for f in fields}
    gemini = {f: getattr(defaults_for(monkeypatch, "gemini"), f) for f in fields}

    assert set(openai.values()).isdisjoint(gemini.values())


def test_the_embedding_default_is_the_width_the_column_was_built_for(monkeypatch):
    """text-embedding-3-small is 1536 natively and migration 0004 wrote
    vector(1536). A default that drifted would not error -- it would fail on
    insert, one event at a time, after paying to embed them."""
    config = defaults_for(monkeypatch, "openai")
    assert config.EMBEDDING_DIMENSIONS == 1536


def test_an_unset_judge_model_keeps_its_own_default(monkeypatch):
    """It used to fall back to CHAT_MODEL. That put both jobs back in one
    quota bucket, and the judge back on its own writing, the moment the
    variable went missing -- silently, which is the worst way to lose it."""
    config, _ = reload_with(monkeypatch, RAG_PROVIDER="gemini",
                            CHAT_MODEL="model-a", JUDGE_MODEL=None)
    assert config.JUDGE_MODEL == "gemini-3.5-flash-lite"
    assert config.JUDGE_MODEL != config.CHAT_MODEL


def test_an_empty_setting_is_treated_as_unset(monkeypatch):
    """An unset CI variable arrives as an empty string, not as absent, so
    os.getenv's default never fires and the model id would be ""."""
    config, _ = reload_with(
        monkeypatch, RAG_PROVIDER="", CHAT_MODEL="", JUDGE_MODEL="",
        EMBEDDING_MODEL=""
    )
    assert config.RAG_PROVIDER == "openai"
    assert config.CHAT_MODEL == "gpt-4o-mini"
    assert config.JUDGE_MODEL == "gpt-4.1-mini"
    assert config.EMBEDDING_MODEL == "text-embedding-3-small"


def test_the_two_providers_are_built_with_different_models(monkeypatch):
    _, providers = reload_with(
        monkeypatch, RAG_PROVIDER="fake",
        CHAT_MODEL="model-a", JUDGE_MODEL="model-b",
    )
    assert providers.get_chat_provider()._model == "model-a"
    assert providers.get_judge_provider()._model == "model-b"


def test_sharing_one_model_is_warned_about(monkeypatch, caplog):
    """Same model means one quota bucket and a judge marking its own work.
    Nothing stops someone setting them equal by hand, but it should never
    happen quietly."""
    _, providers = reload_with(
        monkeypatch, RAG_PROVIDER="gemini", GEMINI_API_KEY="dummy",
        CHAT_MODEL="model-a", JUDGE_MODEL="model-a",
    )
    providers._ANNOUNCED = False
    with caplog.at_level("WARNING"):
        providers.announce_models()

    assert any("same model" in record.message for record in caplog.records)


def test_no_warning_once_they_differ(monkeypatch, caplog):
    _, providers = reload_with(
        monkeypatch, RAG_PROVIDER="gemini", GEMINI_API_KEY="dummy",
        CHAT_MODEL="model-a", JUDGE_MODEL="model-b",
    )
    providers._ANNOUNCED = False
    with caplog.at_level("WARNING"):
        providers.announce_models()

    assert not [r for r in caplog.records if "same model" in r.message]


# =============================================================================
# THE EMBEDDING CACHE IS WIRED IN BY A PATH, AND ONLY BY A PATH
# =============================================================================


class StubEmbedder:
    """Stands in for the Gemini provider, which cannot be built without the
    SDK and a key."""

    def embed_documents(self, texts):
        return [[0.0] for _ in texts]

    def embed_query(self, text):
        return [0.0]

    def chargeable(self, texts):
        return len(texts)


def is_cached(provider):
    from app.rag.embedding_cache import CachedEmbeddingProvider
    return isinstance(provider, CachedEmbeddingProvider)


def test_a_cache_path_wraps_the_embedding_provider(monkeypatch, tmp_path):
    _, providers = reload_with(
        monkeypatch, RAG_PROVIDER="gemini", GEMINI_API_KEY="dummy",
        EMBEDDING_CACHE_PATH=str(tmp_path),
    )
    monkeypatch.setitem(providers._EMBEDDING_PROVIDERS, "gemini", StubEmbedder)

    assert is_cached(providers.get_embedding_provider())


def test_no_path_means_no_cache(monkeypatch):
    """Which is how the app runs. It embeds a question once and never sees it
    again, so there would be nothing to reuse and a directory to grow."""
    _, providers = reload_with(
        monkeypatch, RAG_PROVIDER="gemini", GEMINI_API_KEY="dummy",
        EMBEDDING_CACHE_PATH=None,
    )
    monkeypatch.setitem(providers._EMBEDDING_PROVIDERS, "gemini", StubEmbedder)

    assert not is_cached(providers.get_embedding_provider())


def test_the_fake_provider_is_never_cached(monkeypatch, tmp_path):
    """Its vectors cost nothing, so there is nothing to save -- and caching
    them would leave stub vectors on disk for a real run to find."""
    _, providers = reload_with(
        monkeypatch, RAG_PROVIDER="fake", EMBEDDING_CACHE_PATH=str(tmp_path),
    )

    assert not is_cached(providers.get_embedding_provider())


def test_the_chat_providers_are_not_cached(monkeypatch, tmp_path):
    """Only embeddings are worth reusing. A cached answer would make the eval
    grade a reply the model did not just give."""
    _, providers = reload_with(
        monkeypatch, RAG_PROVIDER="fake", EMBEDDING_CACHE_PATH=str(tmp_path),
    )

    assert not is_cached(providers.get_chat_provider())
    assert not is_cached(providers.get_judge_provider())
