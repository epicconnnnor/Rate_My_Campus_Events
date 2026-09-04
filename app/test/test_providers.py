"""
Tests for which model each job talks to.

Two things have to hold. The models are the Flash-Lite ids, which are the ones
with 500 free requests a day rather than 20. And chat and judge are different
models, so they draw on separate daily allowances and the judge is not marking
its own writing. These check that both are real and not merely configured.
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


def test_the_defaults_are_the_lite_models(monkeypatch):
    """The non-lite flash models allow 20 requests a day, which one eval run
    spends entirely. Both defaults have to be lite ids for the suite to be
    runnable more than once a day."""
    config, _ = reload_with(
        monkeypatch, CHAT_MODEL=None, JUDGE_MODEL=None, EMBEDDING_MODEL=None
    )
    assert config.CHAT_MODEL == "gemini-3.5-flash-lite"
    assert config.JUDGE_MODEL == "gemini-3.1-flash-lite"
    assert config.EMBEDDING_MODEL == "gemini-embedding-001"


def test_an_unset_judge_model_keeps_its_own_default(monkeypatch):
    """It used to fall back to CHAT_MODEL. That put both jobs back in one
    quota bucket, and the judge back on its own writing, the moment the
    variable went missing -- silently, which is the worst way to lose it."""
    config, _ = reload_with(monkeypatch, CHAT_MODEL="model-a", JUDGE_MODEL=None)
    assert config.JUDGE_MODEL == "gemini-3.1-flash-lite"
    assert config.JUDGE_MODEL != config.CHAT_MODEL


def test_an_empty_setting_is_treated_as_unset(monkeypatch):
    """An unset CI variable arrives as an empty string, not as absent, so
    os.getenv's default never fires and the model id would be ""."""
    config, _ = reload_with(
        monkeypatch, CHAT_MODEL="", JUDGE_MODEL="", EMBEDDING_MODEL=""
    )
    assert config.CHAT_MODEL == "gemini-3.5-flash-lite"
    assert config.JUDGE_MODEL == "gemini-3.1-flash-lite"
    assert config.EMBEDDING_MODEL == "gemini-embedding-001"


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
