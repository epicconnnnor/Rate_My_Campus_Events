#!/bin/sh
set -eu

# The schema changes with the application, so make a rollout atomic from the
# operator's point of view: migrate first, then accept traffic.
alembic upgrade head
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --proxy-headers
