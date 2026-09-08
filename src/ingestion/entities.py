"""
Heuristic entity extraction for chunks that have no other source of
entities (i.e. everything except images, which get entities from
filename/reference matching in parse_codex_tables.py).

No NER model is installed in this project, so this is a regex-based
capitalized-multi-word-phrase heuristic: a run of two or more consecutive
Title-Case words — optionally joined by lowercase connector words like
"the"/"of" (e.g. "Sabelle Mournvale the Twice-Crowned") — is treated as
an entity mention. This is deliberately conservative: false negatives
(a real name never matched) are expected and acceptable for a hackathon
prototype, whereas false positives are the real risk, so trailing
connector words are trimmed off, single-word runs are dropped (a lone
sentence-initial "The" doesn't count), and overly long runs are dropped
entirely rather than truncated awkwardly.
"""

import re

_CONNECTORS = {"the", "of", "and", "de", "von", "la", "le", "van", "der"}
_TITLE_WORD = r"[A-Z][a-zA-Z'’-]*"
_CONNECTOR_ALTERNATION = "|".join(sorted(_CONNECTORS))
_PHRASE_RE = re.compile(
    rf"\b{_TITLE_WORD}(?:\s+(?:{_TITLE_WORD}|{_CONNECTOR_ALTERNATION}))*\b"
)

MAX_ENTITY_WORDS = 6


def extract_entities(text: str) -> list[str]:
    """
    Extract candidate entity names from text via the capitalized-phrase
    heuristic described in the module docstring.

    Args:
        text: Chunk text to scan.

    Returns:
        Deduplicated list of candidate entity name strings, in first
        -seen order. Empty list for empty input or no matches.
    """
    if not text:
        return []

    candidates: dict[str, None] = {}

    for match in _PHRASE_RE.finditer(text):
        words = match.group(0).split()

        while words and words[-1].lower() in _CONNECTORS:
            words.pop()

        if len(words) < 2 or len(words) > MAX_ENTITY_WORDS:
            continue

        candidates.setdefault(" ".join(words), None)

    return list(candidates.keys())
