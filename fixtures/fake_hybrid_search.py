"""
FAKE STAND-IN for Person A's real hybrid_search().

Person B: import and use this from Day 1. Swap the import for the real
`retrieval.hybrid_search` the moment Person A hands it off — the function
signature and return shape are identical, so nothing else in your loop
code should need to change.
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

    This fake version does NOT do real search — it just returns chunks
    whose text loosely overlaps with the query, or a random sample if
    nothing matches, so the agent loop always has *something* to reason
    about while Person A's real index isn't ready yet.
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


if __name__ == "__main__":
    results = hybrid_search("Isolde Mournvale faction", k=3)
    for r in results:
        print(r["chunk_id"], "-", r["text"][:70])