"""Orchestrates /chat -- the fast lane.

Steps (the chat-lifecycle):
  1. load recent conversation history
  2. retrieve conflict-aware context for the question
  3. GUARDRAIL: if nothing relevant was found, refuse instead of guessing
  4. build the prompt (system + numbered context + history + question)
  5. call the LLM
  6. attach structured citations for the [n] markers the answer used
  7. persist both turns
"""
import re
from typing import Any
from sqlmodel import Session, select
from app.config import settings
from app.db import engine
from app.models import Conversation, Message
from app.services import llm, retrieval

_WORD = re.compile(r"[a-z0-9]+")

_REFUSAL = ("I couldn't find anything about that in the available policy "
            "documents, so I can't answer from them.")

_SYSTEM_INSTRUCTIONS = (
    "You are Apex Global's internal policy assistant. Answer the user's question "
    "using ONLY the numbered policy excerpts below. Cite the excerpt(s) you used "
    "with their marker, e.g. [1]. If the excerpts do not contain the answer, say "
    "you don't know -- never invent policy details.\n\nContext:\n"
)


def answer(conversation_id: str, message: str) -> dict[str, Any]:
    history = _load_history(conversation_id)
    chunks = retrieval.retrieve(message)

    # Guardrail: refuse before calling the LLM if the context isn't relevant.
    if not _is_grounded(message, chunks):
        _persist(conversation_id, message, _REFUSAL)
        return {"conversation_id": conversation_id, "answer": _REFUSAL,
                "citations": [], "grounded": False}

    messages = [
        {"role": "system", "content": _build_system_prompt(chunks)},
        *history,
        {"role": "user", "content": message},
    ]
    reply = llm.chat_completion(messages)

    citations = _citations_for(reply, chunks)
    _persist(conversation_id, message, reply)
    return {"conversation_id": conversation_id, "answer": reply,
            "citations": citations, "grounded": True}


# ---------- prompt ----------
def _build_system_prompt(chunks: list[dict[str, Any]]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        m = chunk["metadata"]
        ver = f", v{m['version']}" if m.get("version") else ""
        page = f", p.{m['page']}" if m.get("page") else ""
        lines.append(f"[{i}] ({m.get('filename', '?')}{ver}{page}) {chunk['text']}")
    return _SYSTEM_INSTRUCTIONS + "\n".join(lines)


# ---------- guardrail ----------
_STOP = {"what", "is", "the", "a", "an", "for", "of", "to", "in", "on", "do", "does",
         "i", "my", "can", "are", "how", "when", "where", "which", "and", "or", "be",
         "with", "at", "this", "that"}


def _is_grounded(question: str, chunks: list[dict[str, Any]]) -> bool:
    """Cheap, provider-independent relevance gate: at least one meaningful word
    from the question must appear in the retrieved context. Vector search always
    returns *something*, so without this an off-topic question would still get an
    answer attempt."""
    if not chunks:
        return False
    q_words = {w for w in _WORD.findall(question.lower()) if w not in _STOP and len(w) > 2}
    context = " ".join(c["text"].lower() for c in chunks)
    return any(w in context for w in q_words)


# ---------- citations ----------
def _citations_for(reply: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    used = set(re.findall(r"\[(\d+)\]", reply))
    cites = []
    for i, chunk in enumerate(chunks, start=1):
        if used and str(i) not in used:
            continue                       # the answer didn't reference this excerpt
        m = chunk["metadata"]
        cites.append({"marker": f"[{i}]", "filename": m.get("filename"),
                      "policy_id": m.get("policy_id") or None,
                      "version": m.get("version") or None, "page": m.get("page")})
    return cites


# ---------- persistence ----------
def _load_history(conversation_id: str) -> list[dict[str, str]]:
    with Session(engine) as session:
        rows = session.exec(
            select(Message).where(Message.conversation_id == conversation_id)
            .order_by(Message.id.desc()).limit(settings.history_turns)
        ).all()
    return [{"role": r.role, "content": r.content} for r in reversed(rows)]


def _persist(conversation_id: str, user_msg: str, assistant_msg: str) -> None:
    with Session(engine) as session:
        exists = session.exec(select(Conversation).where(
            Conversation.conversation_id == conversation_id)).first()
        if not exists:
            session.add(Conversation(conversation_id=conversation_id))
        session.add(Message(conversation_id=conversation_id, role="user", content=user_msg))
        session.add(Message(conversation_id=conversation_id, role="assistant", content=assistant_msg))
        session.commit()
