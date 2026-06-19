"""FastAPI entrypoint -- wires the application together.

Run locally:   uvicorn app.main:app --reload
Interactive docs at http://localhost:8000/docs
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.db import init_db
from app.routers import chat, upload


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()        # ensure tables exist before serving traffic
    yield
    # no shutdown work needed for the prototype


app = FastAPI(
    title="Apex Global Policy Assistant",
    description="A RAG API over Apex Global's policy documents. "
                "Two endpoints: /upload (ingest a PDF) and /chat (ask a question).",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(upload.router, tags=["ingestion"])
app.include_router(chat.router, tags=["chat"])


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}
