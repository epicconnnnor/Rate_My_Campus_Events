"""Baseline: the four tables SQLModel.create_all() has been building so far

This revision exists so a fresh database and the existing dev database end up in
the same place. On a database that already has these tables (created by the old
`SQLModel.metadata.create_all()` startup hook) each CREATE is skipped, so
`alembic upgrade head` is safe to run against dev without touching any data.

Revision ID: 0001
Revises:
Create Date: 2026-09-02

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import context, op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _existing_tables() -> set:
    """Tables already present. Empty in --sql mode, where there is no connection."""
    if context.is_offline_mode():
        return set()
    return set(sa.inspect(op.get_bind()).get_table_names())


def upgrade() -> None:
    existing = _existing_tables()

    if "users" not in existing:
        op.create_table(
            "users",
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("email", sa.String(), nullable=False),
            sa.Column("password_hash", sa.String(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("user_id"),
        )
        op.create_index("ix_users_name", "users", ["name"], unique=False)
        op.create_index("ix_users_email", "users", ["email"], unique=True)

    if "events" not in existing:
        op.create_table(
            "events",
            sa.Column("event_id", sa.Integer(), nullable=False),
            sa.Column("title", sa.String(), nullable=False),
            sa.Column("description", sa.String(), nullable=True),
            sa.Column("date_time", sa.String(), nullable=True),
            sa.Column("location", sa.String(), nullable=True),
            sa.Column("created_by", sa.Integer(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["created_by"], ["users.user_id"]),
            sa.PrimaryKeyConstraint("event_id"),
        )
        op.create_index("ix_events_title", "events", ["title"], unique=False)

    if "reactions" not in existing:
        op.create_table(
            "reactions",
            sa.Column("reaction_id", sa.Integer(), nullable=False),
            sa.Column("event_id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("value", sa.Integer(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["event_id"], ["events.event_id"]),
            sa.ForeignKeyConstraint(["user_id"], ["users.user_id"]),
            sa.PrimaryKeyConstraint("reaction_id"),
        )

    if "comments" not in existing:
        op.create_table(
            "comments",
            sa.Column("comment_id", sa.Integer(), nullable=False),
            sa.Column("event_id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("text", sa.String(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["event_id"], ["events.event_id"]),
            sa.ForeignKeyConstraint(["user_id"], ["users.user_id"]),
            sa.PrimaryKeyConstraint("comment_id"),
        )


def downgrade() -> None:
    op.drop_table("comments")
    op.drop_table("reactions")
    op.drop_index("ix_events_title", table_name="events")
    op.drop_table("events")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_index("ix_users_name", table_name="users")
    op.drop_table("users")
