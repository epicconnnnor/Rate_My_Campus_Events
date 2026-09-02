"""Allow users with no password

An account created through Google or GitHub never has one. Existing accounts
keep theirs, and either kind can still sign in the way it was made.

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-02

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import context, op

revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "users", "password_hash", existing_type=sa.String(), nullable=True
    )


def downgrade() -> None:
    # A provider-only account has no password to put back. Deleting those users
    # would take their events and comments with them, so refuse and let a
    # person decide instead of guessing.
    if not context.is_offline_mode():
        passwordless = op.get_bind().execute(
            sa.text("SELECT count(*) FROM users WHERE password_hash IS NULL")
        ).scalar()
        if passwordless:
            raise RuntimeError(
                f"{passwordless} user(s) signed up through a provider and have "
                "no password. Give them one, or delete them and whatever they "
                "created, before downgrading past 0006."
            )

    op.alter_column(
        "users", "password_hash", existing_type=sa.String(), nullable=False
    )
