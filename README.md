# RateMyCampusEvents

RateMyCampusEvents helps UMass Amherst students find things to do on campus. Browse the calendar by date and category, or ask the event chatbot a question in plain language and get answers grounded in the official UMass Amherst events calendar.

![RateMyCampusEvents event page](assets/event-detail.png)

## What you can do

- Browse upcoming events in a clear, chronological layout.
- Filter events by keyword, date, category, and free admission.
- Ask questions such as “What is happening this weekend?” or “Find free events next week.”
- Open event details with dates, locations, organizers, and source links.
- Create an account with email and password, Google, or GitHub.

## Built with

- **FastAPI** and **Jinja2** for the application and server-rendered pages
- **HTMX** for responsive interactions without a separate frontend application
- **PostgreSQL** and **pgvector** for event data and semantic search
- **SQLModel** and **Alembic** for database access and migrations
- **OpenAI** or **Google Gemini** for event search and answer generation
- **JWT** and OAuth for authentication

## Run locally

### 1. Prerequisites

- Python 3.13
- Docker Desktop
- An OpenAI API key, or a Gemini API key

### 2. Start PostgreSQL with pgvector

Run this from a terminal:

```powershell
docker run -d --name dev_pg -e POSTGRES_USER=app -e POSTGRES_PASSWORD=app -e POSTGRES_DB=db -p 5432:5432 pgvector/pgvector:pg16
```

### 3. Create a virtual environment and install dependencies

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 4. Configure the application

Set the environment variables in your terminal or in your deployment provider. See [`.env.example`](.env.example) for the full list.

For a local PostgreSQL container, use `localhost` as the database host:

```powershell
$env:DATABASE_HOST = "localhost"
$env:SECRET_KEY = "replace-with-a-long-random-value"
$env:OPENAI_API_KEY = "your-openai-api-key"
```

To use Gemini instead, set `RAG_PROVIDER` to `gemini` and provide `GEMINI_API_KEY`.

### 5. Set up the database and events

```powershell
alembic upgrade head
python -m app.ingest.localist
python -m app.rag.backfill
```

The ingestion command imports the official UMass Amherst calendar. The backfill command creates embeddings used by chatbot search.

### 6. Start the site

```powershell
uvicorn app.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

## Google sign-in setup

1. Create a web OAuth client in [Google Cloud Console](https://console.cloud.google.com/).
2. Add this authorized redirect URI for local development:

   ```text
   http://localhost:8000/auth/google/callback
   ```

3. Set these environment variables:

   ```powershell
   $env:GOOGLE_CLIENT_ID = "your-client-id"
   $env:GOOGLE_CLIENT_SECRET = "your-client-secret"
   ```

For production, replace `localhost:8000` with your live domain in Google Cloud and in the authorized redirect URI.

## Deployment notes

- Use a managed PostgreSQL database with the `vector` extension enabled.
- Run `alembic upgrade head` during deployment.
- Configure `DATABASE_*`, `SECRET_KEY`, your selected provider API key, and OAuth credentials as deployment secrets.
- Leave `DEMO_DATE` unset for live calendar results.
- Use `/healthz` as the application health-check endpoint.

## Tests

Run the offline test suite with:

```powershell
pytest app/test -v --ignore=app/test/test_hallucinations.py
```

The hallucination evaluation runs against a live model and is kept separate from the regular test suite.
