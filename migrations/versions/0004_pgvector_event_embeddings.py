"""Enable pgvector and add event_embeddings

The vector width is written out literally here rather than read from config: it
is part of the column type, so changing it is this migration plus a full
re-index. app.core.config.EMBEDDING_DIMENSIONS has to agree with it.

No ANN index. At a few hundred events an exact scan is faster than ivfflat or
hnsw would be, and pgvector cannot index past 2000 dimensions anyway.

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-02

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

EMBEDDING_DIMENSIONS = 1536


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "event_embeddings",
        sa.Column("embedding_id", sa.Integer(), nullable=False),
        sa.Column("event_id", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIMENSIONS), nullable=False),
        sa.Column("source_updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("embedded_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["event_id"], ["events.event_id"]),
        sa.PrimaryKeyConstraint("embedding_id"),
        # One document per event, and the key the indexer upserts on.
        sa.UniqueConstraint("event_id", name="uq_event_embeddings_event_id"),
    )


def downgrade() -> None:
    op.drop_table("event_embeddings")
    # The extension is left installed: other things may be using it, and
    # dropping it would cascade into any remaining vector columns.
