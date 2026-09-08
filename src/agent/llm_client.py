"""
Shared LLM call wrapper — every component that needs a text LLM call
goes through here, not through a provider SDK directly. Both Person B
(sufficiency, conflict) and Person C (coverage, semantic validation,
final synthesis) use this one function.

Transport: OpenRouter (OpenAI-compatible chat/completions), per the
Technical Spec's stack — one provider, model flexibility, free-tier
quota across the team's accounts. The model is chosen PER CALL via the
`model` argument, defaulting to the cheap/fast tier so existing callers
that pass only a prompt keep working unchanged.

Retry/backoff (spec §2.13) lives here, in one place, rather than
duplicated across every file that needs an LLM call.

The HTTP client is built lazily on first real call, so importing this
module never requires a key or network — pytest and clean checkouts
stay side-effect-free.
"""
import os
import time

from agent.models import DEFAULT_FAST_MODEL

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

_MAX_RETRIES = 4
_BACKOFF_SECONDS = [1, 2, 4, 8]  # exponential backoff, per spec §2.13


def call_llm(prompt: str, model: str = DEFAULT_FAST_MODEL) -> str:
    """
    Send a single-turn text prompt to an OpenRouter model and return the
    raw text response.

    `model` defaults to the fast tier (see agent.models). Callers that
    need a different tier pass it explicitly, e.g.:
        call_llm(prompt, model=SYNTHESIS_MODEL)

    Retries with exponential backoff on failure. Raises RuntimeError if
    every attempt fails, since a silent empty-string return would
    corrupt whatever parses this response downstream (e.g.
    sufficiency.py's JSON parsing).
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }

    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            import requests  # imported lazily so module import needs no deps/key
            resp = requests.post(
                _OPENROUTER_URL,
                headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"] or ""
        except Exception as e:  # narrow once OpenRouter's error types are known
            last_error = e
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_BACKOFF_SECONDS[attempt])
    raise RuntimeError(f"LLM call failed after {_MAX_RETRIES} attempts") from last_error
