"""Thin JSON adapters for Person C; transport and validation stay shared."""

import json
import os

from agent import llm_client
from agent.models import FAST_MODEL, SYNTHESIS_MODEL

# api/main.py's /health/detail reports this as the "strong" tier alongside
# FAST_MODEL. SYNTHESIS_MODEL is that tier in agent/models.py's naming --
# aliased here rather than duplicated so the two never drift.
STRONG_MODEL = SYNTHESIS_MODEL


def llm_available() -> bool:
    """Whether the Gemini key llm_client.call_llm() needs is actually set."""
    return bool(os.environ.get("GEMINI_API_KEY"))


def synthesize(prompt: str) -> object:
    """Decode synthesis output; the composer selects and validates its schema."""
    return json.loads(llm_client.call_llm(prompt, model=SYNTHESIS_MODEL))


def validate_coverage(prompt: str) -> object:
    """Decode coverage output without replacing the existing verdict checks."""
    return json.loads(llm_client.call_llm(prompt, model=FAST_MODEL))


def validate_semantics(prompt: str) -> object:
    """Decode semantic output without replacing the existing verdict checks."""
    return json.loads(llm_client.call_llm(prompt, model=FAST_MODEL))
