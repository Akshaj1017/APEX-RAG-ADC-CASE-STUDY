"""Chat-model provider behind a thin interface.

Same pattern as embeddings.py: one public function, two implementations chosen
by config. The fake lets the chat endpoint (history, citations, guardrail) be
tested end-to-end offline, with no key.
"""
from app.config import settings

# Low temperature: this is factual policy Q&A, not creative writing. We want the
# model to stick to the retrieved context, so we minimise randomness.
_TEMPERATURE = 0.1


def chat_completion(messages: list[dict[str, str]]) -> str:
    """Take OpenAI-style [{role, content}, ...] messages and return the reply text."""
    return _mistral_chat(messages) if _use_mistral() else _fake_chat(messages)


def _use_mistral() -> bool:
    provider = settings.llm_provider
    if provider == "fake":
        return False
    if provider == "mistral":
        return True
    return bool(settings.mistral_api_key)  # "auto"


# ---------- real provider ----------
def _mistral_chat(messages: list[dict[str, str]]) -> str:
    from mistralai import Mistral  # lazy import: offline use needs no install
    client = Mistral(api_key=settings.mistral_api_key)
    resp = client.chat.complete(
        model=settings.chat_model,
        messages=messages,
        temperature=_TEMPERATURE,
    )
    return resp.choices[0].message.content or ""


# ---------- offline fake ----------
def _fake_chat(messages: list[dict[str, str]]) -> str:
    """Deterministic stand-in. Echoes the first cited context line so the chat
    plumbing -- citations, history, guardrail -- is fully exercised offline."""
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    first_cited = next((ln.strip() for ln in system.splitlines()
                        if ln.strip().startswith("[1]")), "")
    if not first_cited:
        return "I couldn't find anything relevant in the policies."
    return f"(offline answer) {first_cited} [1]"
