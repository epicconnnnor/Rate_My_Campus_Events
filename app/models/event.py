"""
SQLModel models for RateMyCampusEvents database tables
"""

from datetime import datetime
from typing import Optional

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
    date_time: Optional[str] = None
    location: Optional[str] = None
    created_by: int = Field(foreign_key="users.user_id")
    created_at: datetime = Field(default_factory=datetime.now)


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
