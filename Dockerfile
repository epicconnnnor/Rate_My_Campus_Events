FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY alembic.ini ./
COPY migrations ./migrations
COPY app ./app
COPY templates ./templates
COPY deploy/oracle/start.sh /usr/local/bin/start-app
RUN chmod +x /usr/local/bin/start-app && useradd --create-home appuser && chown -R appuser:appuser /app

USER appuser
EXPOSE 8000

CMD ["/usr/local/bin/start-app"]
