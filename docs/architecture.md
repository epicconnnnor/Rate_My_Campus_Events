# Technical architecture

## Request flow

```text
Browser
  -> FastAPI routes
    -> Jinja templates and HTMX fragments
    -> database helpers
      -> PostgreSQL + pgvector

UMass Localist calendar
  -> app/ingest/localist.py
  -> events table
  -> app/rag/backfill.py
  -> event_embeddings table
  -> chatbot retrieval and response
```

## Application folders

| Path | Responsibility |
| --- | --- |
| `app/api/` | Page, form, HTMX, authentication, and chatbot routes. |
| `app/core/` | Configuration, authentication security, and OAuth integration. |
| `app/db/` | Database engine and query helpers. |
| `app/models/` | SQLModel database models and authentication data models. |
| `app/ingest/` | UMass Localist import pipeline. |
| `app/rag/` | Embeddings, retrieval, model providers, and answer generation. |
| `app/static/` | Stylesheet served by the application. |
| `app/test/` | Unit tests, live-model evaluation, and frozen event fixtures. |
| `templates/` | Jinja pages and reusable HTMX fragments. |
| `migrations/` | Versioned Alembic schema changes. |

## Data model

`users` own user-created events, comments, reactions, and chat usage. `events` contains both manually created events and records imported from Localist. Each event can have one `event_embeddings` record, which stores the text used for retrieval and its pgvector embedding.

`reactions` records one positive or negative reaction per user and event. `comments` stores discussion attached to an event. `chat_usage` tracks how many questions a user asks each day.

The schema is managed only through Alembic. When a SQLModel model changes, add a migration rather than creating tables at application startup.

## Event synchronization and search

The scheduled workflow imports the official event feed into PostgreSQL. Event changes are detected using their external identifier and feed fields. The embedding backfill then updates only events whose retrieval text or source timestamp changed.

At question time, the chatbot applies any date, category, attendance, or format filters, runs vector retrieval against the event embeddings, and supplies only those retrieved events to the response model. This keeps responses tied to calendar records.

## Authentication

The app supports email and password login plus optional Google and GitHub OAuth. JWTs authenticate application requests, while the session middleware preserves OAuth state during the provider redirect.

## Configuration

All configuration comes from environment variables. See [`.env.example`](../.env.example) for the complete set and [the installation guide](installation.md) for local setup.
