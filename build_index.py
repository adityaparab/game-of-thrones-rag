"""Build the indexes from the Markdown corpus.

  Dense  : OpenAI embeddings -> local embedded Qdrant collection
  Sparse : BM25 over the same chunks -> pickled to disk

Point IDs are shared between both indexes (0..N-1) so dense and sparse results
can be fused by ID. Run once after ingestion (and again when the corpus changes).

  python build_index.py            # uses MD_DIR from config (default out_md/)
"""
import pickle
import sys
from typing import Callable, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from rank_bm25 import BM25Okapi

from config import settings, get_qdrant_client, ensure_qdrant_accessible, QdrantConnectionError
from chunking import iter_corpus, tokenize
import models


def get_qdrant() -> QdrantClient:
    return get_qdrant_client()


def build(embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None):
    try:
        ensure_qdrant_accessible()
    except QdrantConnectionError as e:
        sys.exit(str(e))
    embed_fn = embed_fn or models.embed_texts

    print("Loading + chunking corpus ...")
    chunks = list(iter_corpus())
    if not chunks:
        sys.exit(f"No chunks found in {settings.md_dir!r}. Set MD_DIR or move your .md files there.")
    texts = [c["text"] for c in chunks]
    print(f"  {len(chunks)} chunks")

    # ---- Dense: embed + upsert to Qdrant ----
    print(f"Embedding with {settings.embed_model} ...")
    vectors = embed_fn(texts)
    dim = len(vectors[0])

    client = get_qdrant()
    if client.collection_exists(settings.collection):
        client.delete_collection(settings.collection)
    client.create_collection(
        settings.collection,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    )

    print("Upserting vectors into Qdrant ...")
    B = 256
    for i in range(0, len(chunks), B):
        pts = [
            qm.PointStruct(id=i + j, vector=vectors[i + j], payload=chunks[i + j])
            for j in range(len(chunks[i:i + B]))
        ]
        client.upsert(settings.collection, points=pts)

    # ---- Sparse: BM25 over the same chunk texts ----
    print("Building BM25 index ...")
    bm25 = BM25Okapi([tokenize(t) for t in texts])
    with open(settings.bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    print(f"Done. {len(chunks)} chunks indexed "
          f"(Qdrant: {settings.qdrant_path or settings.qdrant_url}, BM25: {settings.bm25_path})")


if __name__ == "__main__":
    build()
