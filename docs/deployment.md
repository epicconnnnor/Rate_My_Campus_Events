# Deployment guide

## Required services

- PostgreSQL with the pgvector extension enabled
- An OpenAI or Gemini API key for chatbot search
- A publicly reachable application URL for OAuth callbacks and health checks

## Required configuration

Set `DATABASE_USER`, `DATABASE_PASS`, `DATABASE_HOST`, `DATABASE_PORT`, `DATABASE_NAME`, and a long random `SECRET_KEY` as deployment secrets. Set the API key for the selected `RAG_PROVIDER` and configure OAuth credentials only for providers you plan to offer.

Never commit production values to `.env.example` or another tracked file.

## Release steps

1. Install the application dependencies.
2. Run `alembic upgrade head` before serving the new application version.
3. Configure the process to run `uvicorn app.main:app` behind HTTPS.
4. Use `/healthz` for the application health check.
5. Run ingestion and embedding backfill after deployment so the chatbot can search current events.

## Scheduled synchronization

The repository includes a daily GitHub Actions workflow that imports UMass Localist events. Configure its database secrets and `APP_URL` before enabling it. Add the selected model provider key and run `python -m app.rag.backfill` after import; new or changed events need embeddings before semantic search can retrieve them.

## OAuth callbacks

Register these URLs with each provider, replacing the host with the public application domain:

```text
https://your-domain.example/auth/google/callback
https://your-domain.example/auth/github/callback
```
