"""
Database Operations with SQLModel and PostgreSQL

This module provides CRUD operations for the RateMyCampusEvents application.
Uses SQLModel for ORM and PostgreSQL for persistent storage.
"""

from datetime import datetime
from typing import Dict, List, Optional

from app.core.config import DATABASE_URL
from app.models.event import Comment, Event, Reaction, User
from sqlmodel import Session, SQLModel, create_engine, select

# =============================================================================
# DATABASE ENGINE AND SESSION
# =============================================================================

# Create database engine
engine = create_engine(DATABASE_URL, echo=True)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_session():
    return Session(engine)


# =============================================================================
# USER CRUD
# =============================================================================

def create_user(user_data: Dict) -> Dict:
    with get_session() as session:
        user = User(
            name=user_data["name"],
            email=user_data["email"],
            password_hash=user_data["password_hash"]
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        return user.model_dump()


def get_user_by_id(user_id: int) -> Optional[Dict]:
    with get_session() as session:
        user = session.get(User, user_id)
        return user.model_dump() if user else None


def get_user_by_email(email: str) -> Optional[Dict]:
    with get_session() as session:
        statement = select(User).where(User.email == email)
        user = session.exec(statement).first()
        return user.model_dump() if user else None


def list_users() -> List[Dict]:
    with get_session() as session:
        statement = select(User)
        users = session.exec(statement).all()
        return [user.model_dump() for user in users]


def update_user(user_id: int, updates: Dict) -> Optional[Dict]:
    with get_session() as session:
        user = session.get(User, user_id)
        if not user:
            return None
        for key, value in updates.items():
            setattr(user, key, value)
        session.add(user)
        session.commit()
        session.refresh(user)
        return user.model_dump()


def delete_user(user_id: int) -> bool:
    with get_session() as session:
        user = session.get(User, user_id)
        if not user:
            return False
        session.delete(user)
        session.commit()
        return True


# =============================================================================
# EVENT CRUD
# =============================================================================

def create_event(event_data: Dict) -> Dict:
    with get_session() as session:
        event = Event(
            title=event_data["title"],
            description=event_data.get("description"),
            date_time=event_data.get("date_time"),
            location=event_data.get("location"),
            created_by=event_data["created_by"]
        )
        session.add(event)
        session.commit()
        session.refresh(event)
        return event.model_dump()


def get_event(event_id: int) -> Optional[Dict]:
    with get_session() as session:
        event = session.get(Event, event_id)
        return event.model_dump() if event else None


def list_events() -> List[Dict]:
    with get_session() as session:
        statement = select(Event).order_by(Event.created_at.desc())
        events = session.exec(statement).all()
        return [event.model_dump() for event in events]


def update_event(event_id: int, updates: Dict) -> Optional[Dict]:
    with get_session() as session:
        event = session.get(Event, event_id)
        if not event:
            return None
        for key, value in updates.items():
            setattr(event, key, value)
        session.add(event)
        session.commit()
        session.refresh(event)
        return event.model_dump()


def delete_event(event_id: int) -> bool:
    with get_session() as session:
        event = session.get(Event, event_id)
        if not event:
            return False
        session.delete(event)
        session.commit()
        return True


# =============================================================================
# REACTIONS
# =============================================================================

def upsert_reaction(event_id: int, user_id: int, value: int) -> Dict:
    """Create or update a reaction"""
    with get_session() as session:
        # Check for existing reaction
        statement = select(Reaction).where(
            Reaction.event_id == event_id,
            Reaction.user_id == user_id
        )
        existing = session.exec(statement).first()

        if existing:
            # Update existing reaction
            existing.value = value
            session.add(existing)
            session.commit()
            session.refresh(existing)
            return existing.model_dump()
        else:
            # Create new reaction
            reaction = Reaction(
                event_id=event_id,
                user_id=user_id,
                value=value
            )
            session.add(reaction)
            session.commit()
            session.refresh(reaction)
            return reaction.model_dump()


def get_reactions_for_event(event_id: int) -> List[Dict]:
    with get_session() as session:
        statement = select(Reaction).where(Reaction.event_id == event_id)
        reactions = session.exec(statement).all()
        return [reaction.model_dump() for reaction in reactions]


# =============================================================================
# COMMENTS
# =============================================================================

def create_comment(event_id: int, user_id: int, text: str) -> Dict:
    with get_session() as session:
        comment = Comment(
            event_id=event_id,
            user_id=user_id,
            text=text
        )
        session.add(comment)
        session.commit()
        session.refresh(comment)
        return comment.model_dump()


def get_comments_for_event(event_id: int) -> List[Dict]:
    with get_session() as session:
        statement = select(Comment).where(
            Comment.event_id == event_id
        ).order_by(Comment.created_at.desc())
        comments = session.exec(statement).all()
        return [comment.model_dump() for comment in comments]


def delete_comment(comment_id: int) -> bool:
    with get_session() as session:
        comment = session.get(Comment, comment_id)
        if not comment:
            return False
        session.delete(comment)
        session.commit()
        return True


def get_comment(comment_id: int) -> Optional[Dict]:
    with get_session() as session:
        comment = session.get(Comment, comment_id)
        return comment.model_dump() if comment else None
