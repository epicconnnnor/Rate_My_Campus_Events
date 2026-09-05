# Installation guide

This guide runs RateMyCampusEvents on Windows with PowerShell. The same environment variables work on macOS and Linux with their shell syntax.

## Prerequisites

- Python 3.13
- Docker Desktop
- An OpenAI API key or a Gemini API key for chatbot search

## 1. Clone the repository

```powershell
git clone https://github.com/epicconnnnor/Rate_My_Campus_Events.git
cd Rate_My_Campus_Events
```

## 2. Start PostgreSQL with pgvector

The database image must include pgvector. The standard PostgreSQL image does not include the `vector` extension required by the migrations.

```powershell
docker run -d --name dev_pg -e POSTGRES_USER=app -e POSTGRES_PASSWORD=app -e POSTGRES_DB=db -p 5432:5432 pgvector/pgvector:pg16
```

If a container named `dev_pg` already exists, use `docker start dev_pg`.

## 3. Create a virtual environment

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 4. Configure the application

Set configuration in the terminal before running commands. The application reads environment variables directly; `.env.example` is a reference list and is not loaded automatically.

```powershell
$env:DATABASE_USER = "app"
$env:DATABASE_PASS = "app"
$env:DATABASE_HOST = "localhost"
$env:DATABASE_PORT = "5432"
$env:DATABASE_NAME = "db"
$env:SECRET_KEY = python -c "import secrets; print(secrets.token_hex(32))"
$env:OPENAI_API_KEY = "your-openai-api-key"
```

To use Gemini, set the provider and its key instead:

```powershell
$env:RAG_PROVIDER = "gemini"
$env:GEMINI_API_KEY = "your-gemini-api-key"
```

See [`.env.example`](../.env.example) for OAuth credentials and optional model settings.

## 5. Create the database schema and import events

```powershell
alembic upgrade head
python -m app.ingest.localist
python -m app.rag.backfill
```

The first command creates the schema. The second imports the UMass Amherst calendar, and the third creates the vectors used by semantic search.

For a stable local demo without fetching the live calendar, use the bundled fixture data:

```powershell
$env:DEMO_DATE = "2026-09-02"
python -m app.demo_seed --embed --provider fake
```

The fake provider is useful for exercising the search flow locally, but does not provide semantic search quality.

## 6. Start the application

```powershell
uvicorn app.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000).

## Optional Google and GitHub sign-in

Create OAuth applications with Google and GitHub. For local development, register these callback URLs:

```text
http://localhost:8000/auth/google/callback
http://localhost:8000/auth/github/callback
```

Then set the matching `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GITHUB_CLIENT_ID`, and `GITHUB_CLIENT_SECRET` variables. A provider appears in the interface only when both of its values are configured.

## Common problems

| Problem | What to check |
| --- | --- |
| `extension "vector" is not available` | Start the database from `pgvector/pgvector:pg16`, then rerun `alembic upgrade head`. |
| `SECRET_KEY is not set` | Set `SECRET_KEY` in the terminal that starts Uvicorn. |
| Chatbot says it is not configured | Set the selected provider’s API key, then run `python -m app.rag.backfill`. |
| Chatbot cannot find recently imported events | Run `python -m app.rag.backfill` after ingestion. |
