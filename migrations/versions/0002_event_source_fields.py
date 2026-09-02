"""Add Localist source/sync columns to events, allow imported (authorless) events

`date_time` (the free-text string user-created events use) is deliberately left
alone -- `starts_at`/`ends_at` are for imported events only.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-02

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Every existing row predates ingest, so 'user' is the correct backfill. The
    # server default stays on so a row inserted without an explicit source is
    # treated as user-created rather than failing the NOT NULL.
    op.add_column(
        "events",
        sa.Column("source", sa.Text(), nullable=False, server_default="user"),
    )
    op.add_column("events", sa.Column("external_id", sa.Text(), nullable=True))
    op.add_column(
        "events",
        sa.Column("starts_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "events",
        sa.Column("ends_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column("events", sa.Column("status", sa.Text(), nullable=True))
    op.add_column(
        "events",
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "events",
        sa.Column("external_updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column("events", sa.Column("is_free", sa.Boolean(), nullable=True))
    op.add_column("events", sa.Column("experience", sa.Text(), nullable=True))
    op.add_column(
        "events",
        sa.Column("keywords", postgresql.ARRAY(sa.Text()), nullable=True),
    )
    op.add_column("events", sa.Column("localist_url", sa.Text(), nullable=True))

    # Unique so ingest can upsert on it; indexed so the upsert lookup is cheap.
    op.create_index(
        "ix_events_external_id", "events", ["external_id"], unique=True
    )

    # Imported events have no author.
    op.alter_column("events", "created_by", existing_type=sa.Integer(), nullable=True)


def downgrade() -> None:
    # Restoring NOT NULL means imported events cannot survive: they have no
    # author to fall back on. Drop them (and anything hanging off them) first.
    op.execute(
        "DELETE FROM comments WHERE event_id IN "
        "(SELECT event_id FROM events WHERE created_by IS NULL)"
    )
    op.execute(
        "DELETE FROM reactions WHERE event_id IN "
        "(SELECT event_id FROM events WHERE created_by IS NULL)"
    )
    op.execute("DELETE FROM events WHERE created_by IS NULL")

    op.alter_column("events", "created_by", existing_type=sa.Integer(), nullable=False)

    op.drop_index("ix_events_external_id", table_name="events")

    for column in (
        "localist_url",
        "keywords",
        "experience",
        "is_free",
        "external_updated_at",
        "last_seen_at",
        "status",
        "ends_at",
        "starts_at",
        "external_id",
        "source",
    ):
        op.drop_column("events", column)
