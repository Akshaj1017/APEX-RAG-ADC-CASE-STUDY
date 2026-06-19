"""ChromaDB wrapper -- the vector store interface.

Stores, for each chunk: the text, OUR embedding, and its metadata (filename,
page, policy_id, version, effective_date, status) together. We pass embeddings
in ourselves rather than letting Chroma embed, so the model choice stays in
embeddings.py and the two providers (real/fake) flow through unchanged.

Chroma is a pragmatic prototype choice: it persists to disk and supports
metadata out of the box. At enterprise scale this module is the seam we'd swap
for a managed vector DB -- callers only use add_chunks / query / all_chunks.
"""
from typing import Any
import chromadb
from app.config import settings

# One persistent client/collection per process. cosine distance matches our
# L2-normalised vectors (and Mistral's), so "nearest" means "most similar".
_client = chromadb.PersistentClient(path=settings.chroma_path)
_collection = _client.get_or_create_collection(
    name="policies", metadata={"hnsw:space": "cosine"}
)


def add_chunks(ids: list[str], texts: list[str],
               embeddings: list[list[float]], metadatas: list[dict[str, Any]]) -> None:
    """Index a batch of chunks (text + vector + metadata) under stable ids."""
    _collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)


def query(embedding: list[float], top_k: int) -> list[dict[str, Any]]:
    """Return the top_k nearest chunks as {text, metadata, distance}."""
    res = _collection.query(
        query_embeddings=[embedding], n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    if not res["ids"] or not res["ids"][0]:
        return []
    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def all_chunks() -> list[dict[str, Any]]:
    """Every indexed chunk. Used by the keyword (BM25) half of hybrid search."""
    res = _collection.get(include=["documents", "metadatas"])
    return [
        {"id": cid, "text": doc, "metadata": meta}
        for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"])
    ]


def delete_by_filename(filename: str) -> None:
    """Remove all chunks for a file (keeps re-ingestion idempotent)."""
    _collection.delete(where={"filename": filename})


def count() -> int:
    return _collection.count()
