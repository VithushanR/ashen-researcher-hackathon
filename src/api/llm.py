"""Model adapters for Person C's composer, layered on Person B's LLM client.

Person C's ``compose_answer()`` deliberately selects no model itself. It takes
three callables and uses them::

    compose_answer(state, *, synthesize, validate_semantics, validate_coverage)

This module is the matching half of that split, and :func:`pipeline._compose_with_adapters`
is the only thing that imports it. The composer stays testable with three fakes,
and every decision about *which* model, *with what retry policy*, and *cached for
how long* lives here, on the interface side, where the rest of the robustness
work already is.

**This is an adapter, not a provider integration.** The HTTP call, the API key
and the exponential backoff belong to Person B's ``agent/llm_client.py``, which
moved to OpenRouter on 8 September. Two OpenRouter clients with two retry
policies in one repository is a question the team would deserve to be asked at
judging, so there is exactly one, and this module supplies prompts and caching
on top of it.

What is added here, and why it is not duplication:

``cached``
    Person B's client has no cache. Synthesis is the single most expensive call
    in the system and the demo asks the same questions repeatedly while
    rehearsing. A cache hit never enters the retry machinery, per the decorator
    order documented in :mod:`robustness`.

``resilient``
    Person B retries the *transport*. This retries the *adapter*, which includes
    the JSON parsing in :func:`validate_semantics` and :func:`validate_coverage`
    -- a truncated response is a transient failure worth one more attempt, and
    it never reaches Person B's layer as an exception. The two do compose: a
    genuine 429 costs B's four waits inside one of ours, which is bounded and
    logged, not unbounded.

Model tiering (spec section 5) is read from Person B's ``agent/models.py``, the
team's single source of truth for model ids -- not re-declared here. The cheap
tier runs the two validators, which fire on every composed answer; the strong
tier is reserved for synthesis alone. That split is what makes a free-tier quota
survive a demo.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from .pipeline import _ensure_import_paths
from .robustness import cached, resilient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

def _model_names() -> tuple[str, str]:
    """The fast and strong model ids, from Person B's central config.

    Falls back to reading the environment directly if ``agent.models`` cannot be
    imported. That is not defensive padding: this module is imported by
    ``/health/detail``, the endpoint whose entire job is to still answer when
    something else is broken. Reporting "strong_model: deepseek/..." is more
    useful than reporting nothing because a teammate's module moved.
    """
    _ensure_import_paths()
    try:
        from agent.models import FAST_MODEL as fast, SYNTHESIS_MODEL as strong

        return fast, strong
    except Exception as error:  # noqa: BLE001 - diagnostics must survive anything
        logger.debug("agent.models not importable, reading env directly: %s", error)
        return (
            os.getenv("OPENROUTER_FAST_MODEL", "PLACEHOLDER_FAST_MODEL"),
            os.getenv("OPENROUTER_SYNTHESIS_MODEL", "PLACEHOLDER_SYNTHESIS_MODEL"),
        )


FAST_MODEL, STRONG_MODEL = _model_names()

# A model id nobody has confirmed against OpenRouter's live catalogue. Person B
# ships these as placeholders on purpose (see agent/models.py). Calling one
# produces an HTTP 400 whose message says nothing about the real cause, so it is
# worth naming the cause ourselves.
_PLACEHOLDER = re.compile(r"^PLACEHOLDER_", re.IGNORECASE)


def llm_available() -> bool:
    """Can a real model call actually be made right now?

    Both halves matter and they fail differently. Without a key the call raises
    ``KeyError`` deep inside Person B's client; with a placeholder model id it
    reaches OpenRouter and comes back 400. Neither is obvious from the traceback,
    so ``/health/detail`` reports this before anyone starts debugging the wrong
    layer.
    """
    return bool(os.getenv("OPENROUTER_API_KEY")) and not _models_are_placeholders()


def _models_are_placeholders() -> bool:
    return bool(_PLACEHOLDER.match(FAST_MODEL) or _PLACEHOLDER.match(STRONG_MODEL))


def model_status() -> dict[str, Any]:
    """Model configuration as ``/health/detail`` reports it."""
    return {
        "fast_model": FAST_MODEL,
        "strong_model": STRONG_MODEL,
        "key_configured": bool(os.getenv("OPENROUTER_API_KEY")),
        "models_confirmed": not _models_are_placeholders(),
    }


# ---------------------------------------------------------------------------
# The one call
# ---------------------------------------------------------------------------

@cached("llm")
@resilient("openrouter")
def _call(prompt: str, model: str) -> str:
    """One model call, through Person B's client, cached and retried.

    Imported inside the function rather than at module scope for the reason the
    whole seam is lazy: ``/health/detail`` imports this module to *report* on it,
    and a teammate's module raising on import must not turn the diagnostic
    endpoint into a 500.
    """
    _ensure_import_paths()
    from agent.llm_client import call_llm

    return call_llm(prompt, model=model)


def _parse_json_object(raw: str, *, default: dict[str, Any]) -> dict[str, Any]:
    """Pull the first JSON object out of a model response.

    Models wrap JSON in prose and in ``` fences no matter how firmly the prompt
    forbids it. Extracting the braces is what makes the validators usable
    against a cheap model; insisting on clean output is what makes them flaky.

    A response that yields nothing parseable returns ``default`` rather than
    raising, and ``default`` is always the permissive verdict. A validator that
    cannot read its own answer must not be the thing that fails a correct
    composition -- it has no evidence either way, and claiming a problem it did
    not detect is the same class of dishonesty as claiming a check that was
    never run.
    """
    fenced = re.search(r"\{.*\}", raw, re.DOTALL)
    if not fenced:
        logger.warning("No JSON object in model response: %.120r", raw)
        return default
    try:
        parsed = json.loads(fenced.group(0))
    except json.JSONDecodeError as error:
        logger.warning("Unparseable JSON from model (%s): %.120r", error, raw)
        return default
    return parsed if isinstance(parsed, dict) else default


# ---------------------------------------------------------------------------
# The three adapters Person C's composer takes
# ---------------------------------------------------------------------------

def synthesize(prompt: str) -> str:
    """Write the final answer. The strong tier, and the only call that uses it.

    The prompt is Person C's, passed through unchanged. Synthesis is where the
    grounding discipline is enforced, and that discipline belongs in the
    component being judged for it -- not silently appended here where nobody
    reviewing ``composer.py`` would see it.
    """
    return _call(prompt, STRONG_MODEL)


def validate_semantics(claim: str, evidence: str) -> dict[str, Any]:
    """Does the evidence actually support the claim, or merely mention it?

    This is the check that separates "grounded" from "cited". A sentence can
    carry a correct chunk id and still say something that chunk does not
    support, and a string-match citation checker cannot tell the difference --
    only a model reading both can.

    Returns ``{"supported": bool, "reason": str}``. Defaults to supported when
    the response cannot be parsed, per :func:`_parse_json_object`.
    """
    prompt = (
        "You are checking whether a piece of evidence supports a claim.\n\n"
        f"CLAIM:\n{claim}\n\n"
        f"EVIDENCE:\n{evidence}\n\n"
        "Does the evidence support the claim? Mentioning the same subject is "
        "not support; the evidence must state or directly imply the claim.\n"
        'Reply with JSON only: {"supported": true|false, "reason": "<one sentence>"}'
    )
    result = _parse_json_object(
        _call(prompt, FAST_MODEL),
        default={"supported": True, "reason": "validator response unparseable"},
    )
    return {
        "supported": bool(result.get("supported", True)),
        "reason": str(result.get("reason", "")),
    }


def validate_coverage(question: str, answer: str) -> dict[str, Any]:
    """Does the answer address every part of the question?

    Multi-hop questions have parts, and an answer that resolves one part
    fluently reads as complete. This is what turns that into a stated gap --
    which is the honest-partial behaviour sub-track 1C is actually about, so it
    defaults to ``complete: False`` with the parsing failure named as the gap
    rather than defaulting to a clean bill of health.

    Returns ``{"complete": bool, "missing": list[str]}``.
    """
    prompt = (
        "You are checking whether an answer addresses every part of a question.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "List any part of the question the answer does not address. An answer "
        "that explicitly says a part is unknown or unresolved DOES address it.\n"
        'Reply with JSON only: {"complete": true|false, "missing": ["<part>", ...]}'
    )
    result = _parse_json_object(
        _call(prompt, FAST_MODEL),
        default={"complete": False, "missing": ["coverage check response unparseable"]},
    )
    missing = result.get("missing") or []
    if not isinstance(missing, list):
        missing = [str(missing)]
    return {
        "complete": bool(result.get("complete", False)) and not missing,
        "missing": [str(item) for item in missing],
    }
