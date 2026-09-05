#!/bin/sh
set -eu

# The schema changes with the application, so make a rollout atomic from the
# operator's point of view: migrate first, then accept traffic.
alembic upgrade head
# Caddy terminates TLS in the sibling proxy container. Trust its forwarded
# protocol header so OAuth callback URLs are generated as https:// URLs.
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips='*'
