"""
ForgeMind API Dependencies.

FastAPI dependency injection functions that provide
database sessions and service instances to route handlers.
"""

from typing import Generator

from sqlalchemy.orm import Session

from app.infrastructure.database import SessionLocal


def get_db() -> Generator[Session, None, None]:
    """
    Provide a transactional database session.

    Yields a session and ensures it is closed after the request completes.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
