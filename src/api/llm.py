"""Thin JSON adapters for Person C; transport and validation stay shared."""

import json

from agent import llm_client
from agent.models import FAST_MODEL, SYNTHESIS_MODEL


def synthesize(prompt: str) -> object:
    """Decode synthesis output; the composer selects and validates its schema."""
    return json.loads(llm_client.call_llm(prompt, model=SYNTHESIS_MODEL))


def validate_coverage(prompt: str) -> object:
    """Decode coverage output without replacing the existing verdict checks."""
    return json.loads(llm_client.call_llm(prompt, model=FAST_MODEL))


def validate_semantics(prompt: str) -> object:
    """Decode semantic output without replacing the existing verdict checks."""
    return json.loads(llm_client.call_llm(prompt, model=FAST_MODEL))
