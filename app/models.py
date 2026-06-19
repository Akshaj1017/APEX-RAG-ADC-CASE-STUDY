"""Relational tables — the bookkeeping, not the vectors.

  Conversation 1--* Message   (chat history)
  Document                    (registry of every ingested file + its policy metadata)

SQLModel means each class is BOTH the database table and a Pydantic model, so a
table row can be returned from the API without a second set of classes (DRY).
"""
from datetime import datetime, timezone
from typing import Optional
from sqlmodel import SQLModel, Field


def _utcnow() -> datetime:
    """Timezone-aware UTC now (datetime.utcnow() is deprecated in 3.12)."""
    return datetime.now(timezone.utc)


class Conversation(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    conversation_id: str = Field(index=True, unique=True)  # external id from the client
    created_at: datetime = Field(default_factory=_utcnow)


class Message(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    conversation_id: str = Field(index=True)               # indexed for fast history lookups
    role: str                                              # "user" | "assistant"
    content: str
    created_at: datetime = Field(default_factory=_utcnow)


class Document(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str = Field(index=True)
    # Policy metadata extracted at ingestion. Nullable on purpose: interim memos
    # carry no version or effective date.
    policy_id: Optional[str] = None
    version: Optional[str] = None
    effective_date: Optional[str] = None          # ISO string, e.g. "2025-01-01"
    status: Optional[str] = None                  # "active" | "interim" (self-declared)
    superseded: bool = False                      # True once a newer version is ingested
    n_chunks: int = 0
    uploaded_at: datetime = Field(default_factory=_utcnow)
