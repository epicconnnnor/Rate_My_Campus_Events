"""Moderate community event submissions.

Revision ID: 0007
Revises: 0006
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: Union[str, None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "events",
        sa.Column("publication_status", sa.Text(), nullable=False, server_default="published"),
    )
    op.create_index("ix_events_publication_status", "events", ["publication_status"])


def downgrade() -> None:
    op.drop_index("ix_events_publication_status", table_name="events")
    op.drop_column("events", "publication_status")
