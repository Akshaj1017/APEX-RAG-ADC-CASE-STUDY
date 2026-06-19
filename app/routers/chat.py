"""POST /chat -- answer a question from the indexed policies.

Thin: validate the request (Pydantic does it), delegate to the chat service,
return the structured answer. No logic here.
"""
from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from app.payloads import ChatRequest, ChatResponse
from app.services import chat

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    # The chat service is synchronous (DB reads + LLM call). Offload it so the
    # event loop stays free for other requests.
    result = await run_in_threadpool(chat.answer, request.conversation_id, request.message)
    return ChatResponse(**result)
