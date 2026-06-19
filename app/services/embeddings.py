"""Embedding provider behind a thin interface.

Two implementations, chosen by config:
  - "mistral": the real mistral-embed model (used on your machine / in prod)
  - "fake":    a deterministic local embedder (used offline, in tests, in CI)

WHY a fake at all: the embedder is the one component that needs a network and an
API key. Hiding it behind an interface with a local double means the whole
pipeline (and the eval suite) runs with zero cost and zero keys, and swapping
providers in production is a one-line change.

The fake is a hashing embedder: it maps tokens to fixed dimensions by a STABLE
hash and L2-normalises. Stable hashing (hashlib, not Python's built-in hash())
matters because the index is persisted -- a query embedded in a later process
must land in the same vector space as the chunks embedded earlier.
"""
import hashlib
import math
import re
from app.config import settings


# ---------- public interface ----------
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of documents. Order is preserved."""
    return _mistral_embed(texts) if _use_mistral() else [_fake(t) for t in texts]


def embed_query(text: str) -> list[float]:
    """Embed a single query. Kept separate so a real provider could apply a
    query-specific instruction or model without changing callers."""
    return embed_texts([text])[0]


def _use_mistral() -> bool:
    provider = settings.embed_provider
    if provider == "fake":
        return False
    if provider == "mistral":
        return True
    return bool(settings.mistral_api_key)  # "auto": real model only if a key exists


# ---------- real provider ----------
def _mistral_embed(texts: list[str]) -> list[list[float]]:
    from mistralai import Mistral  # lazy import: offline use needs no install
    client = Mistral(api_key=settings.mistral_api_key)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), 32):           # batch to respect rate limits
        batch = texts[start:start + 32]
        resp = client.embeddings.create(model=settings.embed_model, inputs=batch)
        vectors.extend(item.embedding for item in resp.data)
    return vectors


# ---------- local deterministic fake ----------
_TOKEN = re.compile(r"[a-z0-9]+")


def _fake(text: str) -> list[float]:
    dim = settings.embed_dim
    vec = [0.0] * dim
    for token in _TOKEN.findall(text.lower()):
        vec[_stable_hash(token) % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0   # avoid divide-by-zero on empty text
    return [v / norm for v in vec]


def _stable_hash(token: str) -> int:
    """Process-independent hash (Python's built-in hash() is randomised per run)."""
    return int(hashlib.md5(token.encode()).hexdigest(), 16)
