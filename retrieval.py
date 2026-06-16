"""Hybrid retrieval: dense (Qdrant) + sparse (BM25), fused with Reciprocal Rank
Fusion, then reranked with a Cohere cross-encoder. Returns the top-N chunks."""
import pickle
from typing import List, Dict, Optional

from qdrant_client import QdrantClient

from config import settings
from chunking import tokenize
import models

_client: Optional[QdrantClient] = None
_bm25 = None
_chunks: Optional[List[Dict]] = None


def _load():
    global _client, _bm25, _chunks
    if _client is None:
        _client = (QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
                   if settings.qdrant_url else QdrantClient(path=settings.qdrant_path))
    if _bm25 is None:
        with open(settings.bm25_path, "rb") as f:
            data = pickle.load(f)
        _bm25, _chunks = data["bm25"], data["chunks"]


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
