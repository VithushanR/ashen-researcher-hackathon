"""
FAKE STAND-IN for Person A's real hybrid_search().

Person B imports this from Day 1. Swap the import in loop.py for the real
`retrieval.hybrid_search` the moment Person A hands it off — the function
signature and return shape (Contract 2) are identical, so nothing else in
the loop should need to change.

Location note: this lives in the repo-root fixtures/ folder (sibling of
src/), per the Fake Data Fixtures plan, so it can be shared across the
team and reused by pytest later.
"""
import json
import random
from pathlib import Path

_FIXTURE_PATH = Path(__file__).parent / "fake_chunks.json"
_ALL_CHUNKS = json.loads(_FIXTURE_PATH.read_text())


def hybrid_search(query: str, k: int = 8) -> list[dict]:
    """
    Fake version of Contract 2. Returns up to k chunk dicts (Contract 1
    format), best match first.

    Does NOT do real search — returns chunks whose text loosely overlaps
    with the query, or a random sample if nothing matches, so the agent
    loop always has something to reason about while Person A's real index
    isn't ready yet.
    """
    query_words = set(query.lower().split())
    scored = []
    for chunk in _ALL_CHUNKS:
        text_words = set(chunk["text"].lower().split())
        overlap = len(query_words & text_words)
        scored.append((overlap, chunk))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    if scored[0][0] == 0:
        return random.sample(_ALL_CHUNKS, min(k, len(_ALL_CHUNKS)))

    return [chunk for _, chunk in scored[:k]]
