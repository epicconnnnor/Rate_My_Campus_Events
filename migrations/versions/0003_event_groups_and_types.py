"""Add groups and event_types to events

Both feed the embedded document. Localist publishes no departments for UMass;
groups[].name is the organizer ('UMass Athletics', 'Isenberg School of
Management') and filters.event_types is the kind of thing it is
('Lecture/Talk/Reading').

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-02

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "events",
        sa.Column("groups", postgresql.ARRAY(sa.Text()), nullable=True),
    )
    op.add_column(
        "events",
        sa.Column("event_types", postgresql.ARRAY(sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("events", "event_types")
    op.drop_column("events", "groups")
