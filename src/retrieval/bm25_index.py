"""
Builds and persists a BM25 keyword index over the ingested chunks.

BM25Okapi holds no notion of chunk identity itself — it just scores
tokenized documents by position — so the index is always persisted
alongside a parallel chunk_ids list (same order as the tokenized corpus
it was built from) to map scores back to chunks. Rebuilding is skipped
when the persisted pickle already matches the current chunk set, using
the same chunk_id+text fingerprint approach as vector_index.py.
"""

import hashlib
import logging
import os
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

PERSIST_PATH = Path(os.getenv("BM25_INDEX_PATH", "data/bm25_index.pkl"))

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-token tokenizer shared by indexing and
    querying so both sides split text identically."""
    return _TOKEN_RE.findall(text.lower())


def _fingerprint(chunks: list[dict]) -> str:
    hasher = hashlib.sha256()
    for chunk in sorted(chunks, key=lambda c: c["chunk_id"]):
        hasher.update(chunk["chunk_id"].encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(chunk["text"].encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def _load_persisted() -> dict | None:
    if not PERSIST_PATH.exists():
        return None
    try:
        with PERSIST_PATH.open("rb") as f:
            return pickle.load(f)
    except Exception:
        logger.exception("Failed to load persisted BM25 index: %s", PERSIST_PATH)
        return None


def build_bm25_index(chunks: list[dict]) -> tuple[BM25Okapi, list[str]]:
    """
    Build (or load, if unchanged) a BM25 index over the given chunks.

    Args:
        chunks: Final locked-contract chunk dicts.

    Returns:
        (bm25_index, chunk_ids) — chunk_ids[i] is the chunk_id that
        bm25_index's i-th indexed document corresponds to.
    """
    fingerprint = _fingerprint(chunks)

    cached = _load_persisted()
    if cached is not None and cached.get("fingerprint") == fingerprint:
        logger.info("BM25 index already up to date (%d chunks) — skipping rebuild.", len(chunks))
        return cached["bm25"], cached["chunk_ids"]

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PERSIST_PATH.open("wb") as f:
        pickle.dump(
            {"bm25": bm25, "chunk_ids": chunk_ids, "fingerprint": fingerprint}, f
        )

    logger.info("BM25 index built: %d chunks indexed.", len(chunks))
    return bm25, chunk_ids


def get_bm25_index() -> tuple[BM25Okapi, list[str]] | None:
    """Load the persisted BM25 index without rebuilding. Returns None if
    no index has been built yet."""
    cached = _load_persisted()
    if cached is None:
        return None
    return cached["bm25"], cached["chunk_ids"]
