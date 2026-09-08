"""
Builds and persists a Chroma vector index over the ingested chunks,
embedded with Voyage's voyage-4-large model.

Re-embedding is skipped when the persisted index is already up to date:
a manifest file alongside the Chroma store records a hash of every
chunk's (chunk_id, text), and build_vector_index compares against it
before doing any embedding work.

Each chunk's full locked-contract dict is stored verbatim as a JSON
string in Chroma metadata ("chunk_json") rather than mapped field-by
-field, since Chroma metadata values must be str/int/float/bool — this
sidesteps lossy handling of chunk fields that can be None (page,
document_id, section) or a list (entities), and lets hybrid_search.py
reconstruct the exact original chunk on read.
"""

import hashlib
import json
import logging
import os
from pathlib import Path

import chromadb
import voyageai
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

load_dotenv()

PERSIST_DIR = os.getenv("VECTOR_INDEX_DIR", "data/chroma")
COLLECTION_NAME = "ashen_corpus"
EMBED_MODEL = "voyage-4-large"
EMBED_BATCH_SIZE = 128

_MANIFEST_FILENAME = "manifest.json"


def _fingerprint(chunks: list[dict]) -> str:
    """Stable hash of chunk_id+text pairs, used to detect whether the
    persisted index already reflects this exact chunk set."""
    hasher = hashlib.sha256()
    for chunk in sorted(chunks, key=lambda c: c["chunk_id"]):
        hasher.update(chunk["chunk_id"].encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(chunk["text"].encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def _manifest_path() -> Path:
    return Path(PERSIST_DIR) / _MANIFEST_FILENAME


def _read_manifest_fingerprint() -> str | None:
    path = _manifest_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("fingerprint")
    except Exception:
        logger.exception("Failed to read vector index manifest: %s", path)
        return None


def _write_manifest_fingerprint(fingerprint: str) -> None:
    _manifest_path().write_text(
        json.dumps({"fingerprint": fingerprint}), encoding="utf-8"
    )


def _get_client() -> chromadb.ClientAPI:
    Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=PERSIST_DIR)


def get_collection() -> chromadb.Collection:
    """Open the persisted collection without rebuilding it. Used by
    hybrid_search.py, which only ever queries."""
    return _get_client().get_or_create_collection(name=COLLECTION_NAME)


@retry(
    stop=stop_after_attempt(5),  # 4 retries: waits of 1s, 2s, 4s, 8s
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def _embed_batch(voyage_client: voyageai.Client, texts: list[str]) -> list[list[float]]:
    result = voyage_client.embed(texts, model=EMBED_MODEL, input_type="document")
    return result.embeddings


def _chunk_metadata(chunk: dict) -> dict:
    return {
        "chunk_json": json.dumps(chunk, ensure_ascii=False),
        "source_type": chunk["source_type"],
        "reliability": chunk["reliability"],
        "content_type": chunk["content_type"],
    }


def build_vector_index(chunks: list[dict]) -> chromadb.Collection:
    """
    Embed every chunk with Voyage and persist them into a Chroma
    collection on disk. Skips embedding entirely if the persisted index
    already matches this exact chunk set.

    Args:
        chunks: Final locked-contract chunk dicts.

    Returns:
        The persisted Chroma collection.
    """
    client = _get_client()
    fingerprint = _fingerprint(chunks)

    if _read_manifest_fingerprint() == fingerprint:
        logger.info("Vector index already up to date (%d chunks) — skipping re-embed.", len(chunks))
        return client.get_or_create_collection(name=COLLECTION_NAME)

    voyage_api_key = os.getenv("VOYAGE_API_KEY")
    if not voyage_api_key:
        raise RuntimeError("VOYAGE_API_KEY is not set in .env")
    voyage_client = voyageai.Client(api_key=voyage_api_key)

    # Rebuild from scratch rather than diffing — simpler and correct for
    # a corpus that's re-ingested wholesale each time.
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION_NAME)

    any_batch_failed = False
    for batch_start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[batch_start : batch_start + EMBED_BATCH_SIZE]
        texts = [chunk["text"] for chunk in batch]
        try:
            embeddings = _embed_batch(voyage_client, texts)
        except Exception:
            logger.exception(
                "Failed to embed batch starting at index %d — skipping this batch.",
                batch_start,
            )
            any_batch_failed = True
            continue

        collection.add(
            ids=[chunk["chunk_id"] for chunk in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[_chunk_metadata(chunk) for chunk in batch],
        )

    if any_batch_failed:
        # Leave the manifest absent/stale so the next run sees a
        # fingerprint mismatch and retries the whole build — a manifest
        # written here would mark a partially-embedded collection
        # (e.g. 17/3217 chunks under rate limiting) as "up to date"
        # forever, since the fingerprint is keyed on the input chunk set,
        # not on what actually made it into the collection.
        logger.warning(
            "Vector index build incomplete — at least one batch failed to embed. "
            "Manifest not written; the next build_vector_index() call will retry."
        )
    else:
        _write_manifest_fingerprint(fingerprint)
        logger.info("Vector index built: %d chunks embedded.", len(chunks))

    return collection
