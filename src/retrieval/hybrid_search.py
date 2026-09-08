"""
hybrid_search(query, k) — the retrieval contract two teammates depend on.

Pipeline: vector search (Chroma/Voyage embeddings) and BM25 keyword
search each contribute a candidate pool; the pools are merged by simple
order-preserving union (no weighted/RRF fusion — the actual ranking
authority is the Voyage rerank-2.5 call that follows, so the merge step
only needs to gather a reasonably complete candidate set, not pre-rank
it); Voyage rerank-2.5 then reranks the merged pool and the top k go out
in the exact locked chunk contract shape, best match first.

Both indexes must already be built (vector_index.build_vector_index,
bm25_index.build_bm25_index) before this is called — hybrid_search only
opens and queries the persisted indexes, it never rebuilds them.
"""

import json
import logging
import os

import voyageai
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from src.retrieval.bm25_index import get_bm25_index, tokenize
from src.retrieval.vector_index import EMBED_MODEL, get_collection

load_dotenv()
logger = logging.getLogger(__name__)

RERANK_MODEL = "rerank-2.5"
VECTOR_CANDIDATE_POOL = 25
BM25_CANDIDATE_POOL = 25

_voyage_client: voyageai.Client | None = None


def _get_voyage_client() -> voyageai.Client:
    global _voyage_client
    if _voyage_client is None:
        api_key = os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError("VOYAGE_API_KEY is not set in .env")
        _voyage_client = voyageai.Client(api_key=api_key)
    return _voyage_client


_RETRY_KWARGS = dict(
    stop=stop_after_attempt(5),  # 4 retries: waits of 1s, 2s, 4s, 8s
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)


@retry(**_RETRY_KWARGS)
def _embed_query(client: voyageai.Client, query: str) -> list[float]:
    result = client.embed([query], model=EMBED_MODEL, input_type="query")
    return result.embeddings[0]


@retry(**_RETRY_KWARGS)
def _rerank(client: voyageai.Client, query: str, documents: list[str]):
    return client.rerank(query, documents, model=RERANK_MODEL)


def _vector_candidate_ids(collection, voyage_client: voyageai.Client, query: str) -> list[str]:
    try:
        query_embedding = _embed_query(voyage_client, query)
    except Exception:
        logger.exception("Query embedding failed for %r — skipping vector search.", query)
        return []

    try:
        results = collection.query(
            query_embeddings=[query_embedding], n_results=VECTOR_CANDIDATE_POOL
        )
    except Exception:
        logger.exception("Vector index query failed for %r.", query)
        return []

    return results["ids"][0] if results.get("ids") else []


def _bm25_candidate_ids(query: str) -> list[str]:
    index = get_bm25_index()
    if index is None:
        logger.warning("No BM25 index found on disk — skipping keyword search.")
        return []
    bm25, chunk_ids = index

    try:
        scores = bm25.get_scores(tokenize(query))
    except Exception:
        logger.exception("BM25 scoring failed for %r.", query)
        return []

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [chunk_ids[i] for i in ranked_indices[:BM25_CANDIDATE_POOL] if scores[i] > 0]


def _fetch_chunks(collection, chunk_ids: list[str]) -> dict[str, dict]:
    """Reconstruct full locked-contract chunk dicts from the vector
    index's stored chunk_json metadata, keyed by chunk_id."""
    if not chunk_ids:
        return {}

    result = collection.get(ids=chunk_ids)
    chunks: dict[str, dict] = {}
    for chunk_id, metadata in zip(result["ids"], result["metadatas"]):
        try:
            chunks[chunk_id] = json.loads(metadata["chunk_json"])
        except Exception:
            logger.exception("Failed to reconstruct chunk %s from index metadata.", chunk_id)
    return chunks


def hybrid_search(query: str, k: int = 8) -> list[dict]:
    """
    Run hybrid vector + keyword search, rerank, and return the top k
    chunks in the locked contract format, best match first.

    Args:
        query: The search query.
        k: Number of chunks to return.

    Returns:
        Up to k chunk dicts, best match first. Empty list if nothing was
        retrievable (e.g. both indexes empty or unreachable).
    """
    collection = get_collection()
    voyage_client = _get_voyage_client()

    vector_ids = _vector_candidate_ids(collection, voyage_client, query)
    bm25_ids = _bm25_candidate_ids(query)

    # Order-preserving union: vector hits first, then any BM25-only hits.
    candidate_ids = list(dict.fromkeys(vector_ids + bm25_ids))
    if not candidate_ids:
        return []

    id_to_chunk = _fetch_chunks(collection, candidate_ids)
    candidates = [id_to_chunk[cid] for cid in candidate_ids if cid in id_to_chunk]
    if not candidates:
        return []

    try:
        rerank_result = _rerank(voyage_client, query, [c["text"] for c in candidates])
        reranked = [candidates[item.index] for item in rerank_result.results]
    except Exception:
        logger.exception(
            "Rerank failed for %r — falling back to unreranked merged order.", query
        )
        reranked = candidates

    return reranked[:k]
