"""Diagnostic: how many chunks actually made it into each index?"""

from src.retrieval.vector_index import get_collection
from src.retrieval.bm25_index import get_bm25_index

collection = get_collection()
print(f"Vector index chunk count: {collection.count()}")

bm25_result = get_bm25_index()
if bm25_result is None:
    print("BM25 index: not found on disk")
else:
    bm25, chunk_ids = bm25_result
    print(f"BM25 index chunk count: {len(chunk_ids)}")