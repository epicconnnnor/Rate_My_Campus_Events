"""
Tests for which model each job talks to.

The eval needs about twenty generation calls and the free tier allows twenty
per model per day, so the judge running on its own model is what makes the
whole thing fit. These check that the split is real and not just configured.
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


def test_an_unset_judge_model_falls_back_to_the_chat_model(monkeypatch):
    config, _ = reload_with(monkeypatch, CHAT_MODEL="model-a", JUDGE_MODEL=None)
    assert config.JUDGE_MODEL == "model-a"


def test_an_empty_judge_model_falls_back_too(monkeypatch):
    """An unset CI variable arrives as an empty string, not as absent."""
    config, _ = reload_with(monkeypatch, CHAT_MODEL="model-a", JUDGE_MODEL="")
    assert config.JUDGE_MODEL == "model-a"


def test_the_two_providers_are_built_with_different_models(monkeypatch):
    _, providers = reload_with(
        monkeypatch, RAG_PROVIDER="fake",
        CHAT_MODEL="model-a", JUDGE_MODEL="model-b",
    )
    assert providers.get_chat_provider()._model == "model-a"
    assert providers.get_judge_provider()._model == "model-b"


def test_sharing_one_model_is_warned_about(monkeypatch, caplog):
    """Same model means one quota bucket and a judge marking its own work.
    It is allowed, so that nothing breaks before a second model is picked, but
    it should never happen quietly."""
    _, providers = reload_with(
        monkeypatch, RAG_PROVIDER="gemini", GEMINI_API_KEY="dummy",
        CHAT_MODEL="model-a", JUDGE_MODEL=None,
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
