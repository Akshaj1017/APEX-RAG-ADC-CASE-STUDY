"""Orchestrates /upload: the slow lane.

  parse PDF -> extract metadata -> chunk -> embed -> index in Chroma
            -> register in SQLite -> mark older versions of the same policy superseded

The supersession step is the engine behind the contradiction fix: when the 2025
travel policy is ingested, the 2022 one (same policy_id, lower version) is flagged
so retrieval can prefer the current one.
"""
from io import BytesIO
from pypdf import PdfReader
from sqlmodel import Session, select
from app.config import settings
from app.db import engine
from app.models import Conversation, Document
from app.services.metadata import extract_metadata
from app.services.chunking import chunk_pages
from app.services import embeddings, vectorstore


def ingest_pdf(conversation_id: str, filename: str, pdf_bytes: bytes) -> dict:
    pages = _parse_pdf(pdf_bytes)
    full_text = "\n".join(pages)
    meta = extract_metadata(full_text, filename)
    chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise ValueError(f"No extractable text in {filename}")

    # embed and index (idempotent: clear any previous chunks for this file first)
    vectorstore.delete_by_filename(filename)
    texts = [c["text"] for c in chunks]
    vecs = embeddings.embed_texts(texts)
    ids = [f"{filename}::{i}" for i in range(len(chunks))]
    metadatas = [{
        "filename": filename,
        "page": c["page"],
        "policy_id": meta["policy_id"] or "",
        "version": meta["version"] or "",
        "effective_date": meta["effective_date"] or "",
        "status": meta["status"] or "active",
    } for c in chunks]
    vectorstore.add_chunks(ids, texts, vecs, metadatas)

    with Session(engine) as s:
        # ensure the conversation exists
        if not s.exec(select(Conversation).where(
                Conversation.conversation_id == conversation_id)).first():
            s.add(Conversation(conversation_id=conversation_id))

        # register this document (replace any prior row for the same filename)
        for old in s.exec(select(Document).where(Document.filename == filename)).all():
            s.delete(old)
        doc = Document(filename=filename, policy_id=meta["policy_id"], version=meta["version"],
                       effective_date=meta["effective_date"], status=meta["status"],
                       n_chunks=len(chunks))
        s.add(doc)

        # recompute supersession for this policy group (order-independent)
        s.flush()  # make the new doc visible to the query below
        superseded = _resolve_supersession(s, meta["policy_id"])
        s.commit()

    return {
        "filename": filename, "n_chunks": len(chunks),
        "policy_id": meta["policy_id"], "version": meta["version"],
        "effective_date": meta["effective_date"], "status": meta["status"],
        "superseded_now": superseded,
    }


def _resolve_supersession(s: Session, pid: str | None) -> list:
    """Among all docs sharing a policy_id, the latest effective_date is current;
    every other dated version is marked superseded. Order-independent. Returns
    filenames newly retired by this ingestion."""
    if not pid:
        return []
    docs = s.exec(select(Document).where(Document.policy_id == pid)).all()
    dated = [d.effective_date for d in docs if d.effective_date]
    if len(docs) < 2 or not dated:
        return []
    latest = max(dated)
    newly_retired = []
    for d in docs:
        should = bool(d.effective_date) and d.effective_date < latest
        if should and not d.superseded:
            newly_retired.append(d.filename)
        d.superseded = should
        s.add(d)
    return newly_retired


# ---------- PDF parsing (kept here so /upload ingestion is one self-contained unit) ----------
def _parse_pdf(pdf_bytes: bytes) -> list[str]:
    """Return a list of page texts -- page by page, so each chunk can carry a
    page number for citations."""
    reader = PdfReader(BytesIO(pdf_bytes))
    return [_clean(page.extract_text() or "") for page in reader.pages]


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln)
