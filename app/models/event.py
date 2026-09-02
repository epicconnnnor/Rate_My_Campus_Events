"""
SQLModel models for RateMyCampusEvents database tables

The schema is owned by Alembic (see migrations/). Changing a model here means
writing a migration to match -- nothing creates tables from these definitions
at runtime any more.
"""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import Boolean, Column, DateTime, Text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlmodel import Field, SQLModel


class User(SQLModel, table=True):
    __tablename__ = "users"

    user_id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    email: str = Field(unique=True, index=True)
    password_hash: str
    created_at: datetime = Field(default_factory=datetime.now)


class Event(SQLModel, table=True):
    __tablename__ = "events"

    event_id: Optional[int] = Field(default=None, primary_key=True)
    title: str = Field(index=True)
    description: Optional[str] = None
    # Free-text date for user-created events. Imported events use starts_at/ends_at.
    date_time: Optional[str] = None
    location: Optional[str] = None
    # NULL for imported events, which have no author.
    created_by: Optional[int] = Field(
        default=None, foreign_key="users.user_id", nullable=True
    )
    created_at: datetime = Field(default_factory=datetime.now)

    # -------------------------------------------------------------------------
    # Source / sync metadata (populated by the Localist ingest)
    # -------------------------------------------------------------------------

    # 'localist' or 'user'
    source: str = Field(
        default="user",
        sa_column=Column(Text, nullable=False, server_default="user"),
    )
    # Localist event id; the key ingest upserts on.
    external_id: Optional[str] = Field(
        default=None, sa_column=Column(Text, unique=True, index=True, nullable=True)
    )
    starts_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    ends_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    # Localist status, e.g. 'live'
    status: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    # Last sync that returned this event. An event that stops appearing goes
    # stale, it is never deleted.
    last_seen_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    # Localist's own updated_at; drives re-embedding later on.
    external_updated_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    is_free: Optional[bool] = Field(
        default=None, sa_column=Column(Boolean, nullable=True)
    )
    # 'inperson' / 'virtual' / 'hybrid'
    experience: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    keywords: Optional[List[str]] = Field(
        default=None, sa_column=Column(ARRAY(Text), nullable=True)
    )
    localist_url: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )


class Reaction(SQLModel, table=True):
    __tablename__ = "reactions"

    reaction_id: Optional[int] = Field(default=None, primary_key=True)
    event_id: int = Field(foreign_key="events.event_id")
    user_id: int = Field(foreign_key="users.user_id")
    value: int = Field(ge=-1, le=1)
    created_at: datetime = Field(default_factory=datetime.now)


class Comment(SQLModel, table=True):
    __tablename__ = "comments"

    comment_id: Optional[int] = Field(default=None, primary_key=True)
    event_id: int = Field(foreign_key="events.event_id")
    user_id: int = Field(foreign_key="users.user_id")
    text: str
    created_at: datetime = Field(default_factory=datetime.now)
