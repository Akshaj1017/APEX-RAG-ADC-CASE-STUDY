"""Database engine and session management (SQLite via SQLModel).

SQLModel = SQLAlchemy + Pydantic, so the same classes define both our database
tables and their validation. We expose:
  - init_db():     create tables at startup
  - get_session(): a FastAPI dependency that yields one session per request
"""
from typing import Iterator
from sqlmodel import SQLModel, Session, create_engine
from app.config import settings

# check_same_thread=False is the standard, safe setting for SQLite running under
# a web server: FastAPI may touch the connection from different worker threads.
engine = create_engine(
    f"sqlite:///{settings.sqlite_path}",
    echo=False,
    connect_args={"check_same_thread": False},
)


def init_db() -> None:
    """Create any tables that don't yet exist. Idempotent; called once on startup."""
    SQLModel.metadata.create_all(engine)


def get_session() -> Iterator[Session]:
    """Yield a session and guarantee it is closed afterwards.

    Used as a FastAPI dependency (Depends(get_session)) so every request gets its
    own session and connections are never leaked.
    """
    with Session(engine) as session:
        yield session
