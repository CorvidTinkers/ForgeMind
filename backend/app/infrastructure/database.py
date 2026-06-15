"""
ForgeMind Database Infrastructure.

SQLAlchemy 2.0 synchronous engine and session management.
Uses synchronous driver (psycopg2) for simplicity in Phase 1.
Celery workers are inherently synchronous, so this avoids
async complexity where it provides no benefit.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.domain.models import Base

settings = get_settings()

engine = create_engine(
    settings.database_url,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=settings.app_debug,
)

SessionLocal = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)


def init_db() -> None:
    """Create all tables. Called on application startup."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    """Yield a database session. Used as a FastAPI dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
