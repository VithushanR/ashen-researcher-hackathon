"""
Manual verification script — NOT part of the pipeline.

Builds the vector + BM25 indexes from data/chunks.jsonl (skips rebuild
if already up to date, per each index's own fingerprint check), then
runs a handful of real queries through hybrid_search() and prints the
top results so you can eyeball whether retrieval is actually working
before handing hybrid_search() off to Person B and C.

Usage:
    python scripts/test_retrieval.py
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from src.retrieval.bm25_index import build_bm25_index
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.vector_index import build_vector_index

CHUNKS_PATH = Path("data/chunks.jsonl")

TEST_QUERIES = [
    "Isolde Fellgard",
    "Morwenna Ashgrove garrison strength",
    "Crookgate Keep founded",
]


def load_chunks() -> list[dict]:
    chunks = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def main() -> None:
    print(f"Loading chunks from {CHUNKS_PATH}...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks.\n")

    print("Building/verifying vector index (this embeds via Voyage if not already built)...")
    build_vector_index(chunks)
    print("Vector index ready.\n")

    print("Building/verifying BM25 index...")
    build_bm25_index(chunks)
    print("BM25 index ready.\n")

    for query in TEST_QUERIES:
        print("=" * 70)
        print(f"QUERY: {query!r}")
        print("=" * 70)

        results = hybrid_search(query, k=5)

        if not results:
            print("  (no results returned)")
            continue

        for rank, chunk in enumerate(results, start=1):
            preview = chunk["text"][:150].replace("\n", " ")
            print(f"\n  [{rank}] chunk_id: {chunk['chunk_id']}")
            print(f"      filename: {chunk['filename']}")
            print(f"      reliability: {chunk['reliability']}  content_type: {chunk['content_type']}")
            print(f"      text: {preview}...")

        print()


if __name__ == "__main__":
    main()