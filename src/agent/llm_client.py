"""
Shared LLM call wrapper — every component that needs a text LLM call
goes through here, not through a provider SDK directly. Both Person B
(sufficiency, conflict) and Person C (coverage, semantic validation,
final synthesis) use this one function.

Transport: Google Gemini, via the `google-genai` package (the current,
supported SDK — the older `google-generativeai` package is fully
deprecated by Google and receives no further updates, so it was not
used here). The model is chosen PER CALL via the `model` argument,
defaulting to the cheap/fast tier so existing callers that pass only a
prompt keep working unchanged. Swapped in from OpenRouter, which had
become unusable under free-tier rate limits; the public signature and
return type are unchanged, so callers needed zero edits.

Retry/backoff (spec §2.13) lives here, in one place, rather than
duplicated across every file that needs an LLM call.

The client is built lazily on first real call, so importing this
module never requires a key or network — pytest and clean checkouts
stay side-effect-free.
"""
import os
import re
import time

from agent.models import DEFAULT_FAST_MODEL

_MAX_RETRIES = 4
_BACKOFF_SECONDS = [1, 2, 4, 8]  # exponential backoff, per spec §2.13

# Gemini routinely wraps structured output in a markdown code fence (```json
# ... ```) even when the prompt asks for "ONLY a JSON object, no other text" --
# OpenRouter's prior model did not do this, so every caller here parses the
# return value with json.loads() directly and none of them strip fences
# themselves (confirmed empirically: the full test suite's real-Gemini tests
# failed on this exact shape the first time this swap was verified). Stripping
# it here, once, is what keeps call_llm's contract -- "the raw text response"
# -- actually true for every caller, without editing sufficiency.py,
# conflict.py, composer.py, or coverage_validation.py to each work around a
# transport-specific quirk individually.
_CODE_FENCE = re.compile(r"^```[a-zA-Z0-9_+-]*\s*\n(.*)\n```\s*$", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    match = _CODE_FENCE.match(text.strip())
    return match.group(1).strip() if match else text


def call_llm(prompt: str, model: str = DEFAULT_FAST_MODEL) -> str:
    """
    Send a single-turn text prompt to a Gemini model and return the
    raw text response.

    `model` defaults to the fast tier (see agent.models). Callers that
    need a different tier pass it explicitly, e.g.:
        call_llm(prompt, model=SYNTHESIS_MODEL)

    Retries with exponential backoff on failure. Raises RuntimeError if
    every attempt fails, since a silent empty-string return would
    corrupt whatever parses this response downstream (e.g.
    sufficiency.py's JSON parsing).
    """
    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            from google import genai  # imported lazily so module import needs no deps/key
            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            resp = client.models.generate_content(model=model, contents=prompt)
            return _strip_code_fence(resp.text or "")
        except Exception as e:  # narrow once Gemini's error types are known
            last_error = e
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_BACKOFF_SECONDS[attempt])
    raise RuntimeError(f"LLM call failed after {_MAX_RETRIES} attempts") from last_error
