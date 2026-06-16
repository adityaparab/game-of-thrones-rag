"""Hybrid retrieval: dense (Qdrant) + sparse (BM25), fused with Reciprocal Rank
Fusion, then reranked with a Cohere cross-encoder. Returns the top-N chunks."""
import os
import pickle
import sys
from typing import List, Dict, Optional, Tuple

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from config import settings, get_qdrant_client, ensure_qdrant_accessible
from chunking import tokenize
import models

_client: Optional[QdrantClient] = None
_bm25 = None
_chunks: Optional[List[Dict]] = None


def _load_bm25_from_qdrant(client: QdrantClient) -> Tuple[BM25Okapi, List[Dict]]:
    """Rebuild BM25 + chunk list from Qdrant payloads (used when no local pickle exists)."""
    if not client.collection_exists(settings.collection):
        sys.exit(
            f"BM25 index not found at {settings.bm25_path!r} and Qdrant collection "
            f"{settings.collection!r} does not exist. Run build_index.py first."
        )

    points_by_id: Dict[int, Dict] = {}
    offset = None
    while True:
        batch, offset = client.scroll(
            settings.collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in batch:
            if point.payload:
                points_by_id[int(point.id)] = point.payload
        if offset is None:
            break

    if not points_by_id:
        sys.exit(
            f"BM25 index not found at {settings.bm25_path!r} and Qdrant collection "
            f"{settings.collection!r} is empty. Run build_index.py first."
        )

    max_id = max(points_by_id)
    if len(points_by_id) != max_id + 1:
        sys.exit(
            f"Qdrant collection {settings.collection!r} has non-contiguous point IDs; "
            "cannot rebuild BM25 index. Re-run build_index.py."
        )

    chunks = [points_by_id[i] for i in range(max_id + 1)]
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    print(f"Loaded BM25 index from Qdrant ({len(chunks)} chunks)", file=sys.stderr)
    return bm25, chunks


def _load():
    global _client, _bm25, _chunks
    if _client is None:
        ensure_qdrant_accessible()
        _client = get_qdrant_client()
    if _bm25 is None:
        if os.path.isfile(settings.bm25_path):
            with open(settings.bm25_path, "rb") as f:
                data = pickle.load(f)
            _bm25, _chunks = data["bm25"], data["chunks"]
        else:
            assert _client is not None
            _bm25, _chunks = _load_bm25_from_qdrant(_client)


def dense_search(query: str, k: int) -> List[int]:
    assert _client is not None, "call _load() first"
    qvec = models.embed_query(query)
    res = _client.query_points(settings.collection, query=qvec, limit=k, with_payload=False)
    return [int(p.id) for p in res.points]


def sparse_search(query: str, k: int) -> List[int]:
    assert _bm25 is not None, "call _load() first"
    scores = _bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return ranked[:k]


def rrf(rankings: List[List[int]], k: int) -> List[int]:
    """Reciprocal Rank Fusion: combine multiple ranked ID lists into one."""
    fused: Dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused, key=lambda d: fused[d], reverse=True)


def hybrid_retrieve(query: str, top_n: Optional[int] = None) -> List[Dict]:
    """Full retrieval pipeline -> list of chunk dicts (best first), each with a
    'score' (rerank relevance) and 'rank' field added."""
    _load()
    assert _chunks is not None
    top_n = top_n or settings.rerank_top_n

    dense_ids = dense_search(query, settings.dense_top_k)
    sparse_ids = sparse_search(query, settings.sparse_top_k)
    candidate_ids = rrf([dense_ids, sparse_ids], settings.rrf_k)[: max(settings.dense_top_k, 40)]

    candidates = [_chunks[i] for i in candidate_ids]
    reranked = models.rerank(query, [c["text"] for c in candidates], top_n)

    out = []
    for rank, (idx, score) in enumerate(reranked, 1):
        c = dict(candidates[idx])
        c["score"], c["rank"] = score, rank
        out.append(c)
    return out
