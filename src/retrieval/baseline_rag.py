"""
Deliberately dumb comparison baseline: one hybrid_search call, straight
to an LLM answer, no sufficiency checking, no retrieval loop. This is the
naive contrast the eval harness measures the full agent loop's uplift
against.

Calls OpenRouter directly rather than going through src/agent/llm_client
.py's call_llm() — that wrapper is currently a temporary Gemini stand-in
for the agent team (see its own docstring), not yet OpenRouter-backed.
Keeping this file self-contained means it doesn't inherit that
in-progress swap or accidentally require GEMINI_API_KEY.
"""

import logging
import os

import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from src.retrieval.hybrid_search import hybrid_search

load_dotenv()
logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Verify this id is still live at https://openrouter.ai/models — OpenRouter's
# free-tier lineup changes over time. Override via OPENROUTER_MODEL in .env
# if it's been retired or renamed.
DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

RETRIEVAL_K = 8


@retry(
    stop=stop_after_attempt(5),  # 4 retries: waits of 1s, 2s, 4s, 8s
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def _call_openrouter(prompt: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set in .env")

    model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
    response = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}]},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def _build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[{chunk['chunk_id']}] ({chunk['source_type']}, {chunk['reliability']}) {chunk['text']}"
        for chunk in chunks
    )
    return (
        "Answer the question using only the context below. If the context "
        "doesn't contain the answer, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


def baseline_rag(question: str) -> str:
    """
    Naive one-shot RAG: single hybrid_search call, context stuffed into
    one LLM prompt, no sufficiency checking or retrieval loop.

    Args:
        question: The user's question.

    Returns:
        The LLM's answer text. A fallback message if retrieval found
        nothing, or if the LLM call fails after retries.
    """
    chunks = hybrid_search(question, k=RETRIEVAL_K)
    if not chunks:
        return "I couldn't find any relevant information to answer that."

    prompt = _build_prompt(question, chunks)
    try:
        return _call_openrouter(prompt)
    except Exception:
        logger.exception("Baseline RAG LLM call failed for question: %r", question)
        return "Something went wrong generating an answer."
