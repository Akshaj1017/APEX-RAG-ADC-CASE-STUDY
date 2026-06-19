"""Hybrid, conflict-aware retrieval -- the heart of answer quality.

Two ideas combined:

1. HYBRID SEARCH. Semantic (vector) search understands meaning; keyword (BM25)
   search nails exact tokens like "USD 250" or "FIN-T&E-001" that embeddings
   blur. We run both and fuse them with Reciprocal Rank Fusion (RRF), which
   merges by RANK -- so we never have to reconcile two incompatible score scales
   (cosine distance vs BM25 score).

2. CONFLICT RESOLUTION. Raw search happily returns a superseded policy beside the
   current one. We fix that with metadata:
     - chunks from a SUPERSEDED document (older version of a policy) are dropped
     - chunks from an INTERIM document (temporary memo) are downweighted
   so the current, permanent policy wins.
"""
import re
from typing import Any
from rank_bm25 import BM25Okapi
from sqlmodel import Session, select
from app.config import settings
from app.db import engine
from app.models import Document
from app.services import embeddings, vectorstore

_RRF_K = 60              # standard RRF constant; dampens the weight of low ranks
_INTERIM_PENALTY = 0.5   # interim docs keep half their fused score
_CANDIDATES = 10         # how many to pull from each retriever before fusing


def retrieve(query_text: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """Return up to top_k chunks as {text, metadata, score}, best first."""
    top_k = top_k or settings.top_k
    corpus = vectorstore.all_chunks()
    if not corpus:
        return []

    vector_ranked = _vector_search(query_text)
    keyword_ranked = _keyword_search(query_text, corpus)

    fused = _reciprocal_rank_fusion([vector_ranked, keyword_ranked])
    resolved = _apply_conflict_policy(fused)
    resolved.sort(
        key=lambda c: (c["score"], c["metadata"].get("effective_date", "")),
        reverse=True,
    )
    return resolved[:top_k]


# ---------- the two retrievers ----------
def _vector_search(query_text: str) -> list[dict[str, Any]]:
    vec = embeddings.embed_query(query_text)
    return vectorstore.query(vec, _CANDIDATES)        # already nearest-first


def _keyword_search(query_text: str, corpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bm25 = BM25Okapi([_tokenize(c["text"]) for c in corpus])
    scores = bm25.get_scores(_tokenize(query_text))
    ranked = sorted(zip(corpus, scores), key=lambda pair: pair[1], reverse=True)
    return [chunk for chunk, score in ranked[:_CANDIDATES] if score > 0]


# ---------- fusion ----------
def _reciprocal_rank_fusion(result_lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}
    for results in result_lists:
        for rank, chunk in enumerate(results):
            entry = fused.setdefault(
                chunk["text"],
                {"text": chunk["text"], "metadata": chunk["metadata"], "score": 0.0},
            )
            entry["score"] += 1.0 / (_RRF_K + rank)
    return list(fused.values())


# ---------- conflict resolution ----------
def _apply_conflict_policy(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    superseded = _superseded_filenames()
    kept: list[dict[str, Any]] = []
    for chunk in chunks:
        meta = chunk["metadata"]
        if meta.get("filename") in superseded:
            continue                                   # drop retired policy versions
        if meta.get("status") == "interim":
            chunk["score"] *= _INTERIM_PENALTY         # downweight temporary memos
        kept.append(chunk)
    return kept


def _superseded_filenames() -> set[str]:
    with Session(engine) as session:
        rows = session.exec(select(Document).where(Document.superseded)).all()
    return {doc.filename for doc in rows}


_WORD = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())
