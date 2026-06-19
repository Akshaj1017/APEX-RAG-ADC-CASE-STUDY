"""POST /upload -- ingest a PDF into the index.

Thin by design: validate the input, hand off to the ingestion service, shape the
response. No business logic lives here.
"""
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from app.payloads import UploadResponse
from app.services import ingestion

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload(
    conversation_id: str = Form(...),
    file: UploadFile = File(...),
) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        # Ingestion is blocking (parse + embed). Run it in a threadpool so it
        # doesn't stall the async event loop -- a small echo of Part 2's
        # "decouple slow ingestion from fast chat" principle.
        result = await run_in_threadpool(
            ingestion.ingest_pdf, conversation_id, file.filename, pdf_bytes
        )
    except ValueError as exc:                       # e.g. no extractable text
        raise HTTPException(status_code=422, detail=str(exc))

    note = ""
    if result["superseded_now"]:
        note = f" Superseded older version(s): {', '.join(result['superseded_now'])}."
    return UploadResponse(
        conversation_id=conversation_id,
        filename=result["filename"],
        n_chunks=result["n_chunks"],
        policy_id=result["policy_id"],
        version=result["version"],
        effective_date=result["effective_date"],
        message=f"Indexed {result['n_chunks']} chunks.{note}",
    )
