"""
SQLModel models for RateMyCampusEvents database tables

The schema is owned by Alembic (see migrations/). Changing a model here means
writing a migration to match -- nothing creates tables from these definitions
at runtime any more.
"""

from datetime import date, datetime, timezone
from typing import List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, Column, Date, DateTime, ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlmodel import Field, SQLModel

from app.core.config import EMBEDDING_DIMENSIONS


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
    # Organizers, e.g. 'UMass Athletics'. Part of the embedded document.
    groups: Optional[List[str]] = Field(
        default=None, sa_column=Column(ARRAY(Text), nullable=True)
    )
    # Localist event_types, e.g. 'Lecture/Talk/Reading'. Part of the embedded
    # document, and the axis a question like "not a lecture" turns on.
    event_types: Optional[List[str]] = Field(
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


class EventEmbedding(SQLModel, table=True):
    """One embedded document per event, for semantic search.

    `content` is stored alongside the vector so a re-index can tell whether the
    document actually changed before spending an API call on it.
    """

    __tablename__ = "event_embeddings"

    embedding_id: Optional[int] = Field(default=None, primary_key=True)
    event_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("events.event_id"),
            unique=True,
            nullable=False,
        )
    )
    content: str = Field(sa_column=Column(Text, nullable=False))
    embedding: List[float] = Field(
        sa_column=Column(Vector(EMBEDDING_DIMENSIONS), nullable=False)
    )
    # The event's external_updated_at as of the last embed, so a sync that
    # touched the event can be told apart from one that did not.
    source_updated_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    embedded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class ChatUsage(SQLModel, table=True):
    """How many chat questions a user has asked on a given day.

    A counter, not a log -- the cap is the only thing that needs to know, and
    what people asked is nobody's business.
    """

    __tablename__ = "chat_usage"

    usage_id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.user_id")
    usage_date: date = Field(sa_column=Column(Date, nullable=False))
    request_count: int = Field(
        default=0, sa_column=Column(Integer, nullable=False, server_default="0")
    )
