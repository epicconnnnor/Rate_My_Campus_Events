"""Add chat_usage, one row per user per day

Backs the daily cap on /chat. A counter rather than a log: the cap only needs
to know how many, and there is no reason to keep what anyone asked.

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-02

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "chat_usage",
        sa.Column("usage_id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("usage_date", sa.Date(), nullable=False),
        sa.Column(
            "request_count", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"]),
        sa.PrimaryKeyConstraint("usage_id"),
        # The key the counter upserts on.
        sa.UniqueConstraint("user_id", "usage_date", name="uq_chat_usage_day"),
    )


def downgrade() -> None:
    op.drop_table("chat_usage")
