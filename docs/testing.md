# Testing guide

## Offline tests

Install dependencies in the project virtual environment, then run:

```powershell
pytest app/test -v --ignore=app/test/test_hallucinations.py
```

These tests cover routing, provider selection, retrieval behavior, OAuth helpers, event display, embedding caching, and rate pacing. They should run without a database or model API key.

## Live-model evaluation

The hallucination evaluation uses PostgreSQL with pgvector, frozen event data, and a real model API key. It is intentionally kept out of the regular test command because it consumes provider quota.

```powershell
pytest app/test/test_hallucinations.py -v
```

For a local evaluation, first configure a database and selected model provider, run migrations, and ensure the fixture events can be loaded. GitHub Actions runs this evaluation only when manually dispatched.

## Test data

`app/test/fixtures/events.json` contains frozen UMass event data. `golden_questions.json` contains the questions used to judge retrieval and generated answers. Keep changes to those files deliberate: they define the expected evaluation set.

## Before opening a pull request

Run the offline suite and confirm that documentation commands still match the current configuration. If you change models, migrations, ingestion, or retrieval behavior, run the relevant focused tests as well.
