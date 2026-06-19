"""API request/response payloads (Pydantic).

Kept separate from the database models on purpose: the public API contract
should not be coupled to the storage layout, so either can change without
breaking the other. These also drive FastAPI's auto-generated /docs, so each
field carries a description and example.
"""
from pydantic import BaseModel, Field


# ---------- /upload ----------
class UploadResponse(BaseModel):
    conversation_id: str
    filename: str
    n_chunks: int = Field(description="Number of chunks indexed from this file.")
    policy_id: str | None = None
    version: str | None = None
    effective_date: str | None = None
    message: str = Field(description="Human-readable confirmation.")


# ---------- /chat ----------
class ChatRequest(BaseModel):
    conversation_id: str = Field(min_length=1, examples=["demo-1"])
    message: str = Field(min_length=1, examples=["What is the hotel limit for Asia?"])


class Citation(BaseModel):
    marker: str = Field(description="In-text marker, e.g. '[1]'.", examples=["[1]"])
    filename: str
    policy_id: str | None = None
    version: str | None = None
    page: int | None = None


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    citations: list[Citation]
    grounded: bool = Field(
        description="False when no relevant policy text was found; the guardrail "
                    "then returns a refusal instead of a guessed answer."
    )
