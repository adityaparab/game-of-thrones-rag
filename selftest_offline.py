"""Offline smoke test — validates chunking + Qdrant + BM25 + RRF + retrieval
WITHOUT any API keys, by stubbing the embedding/rerank model calls with a
deterministic hashing embedder. Run: python selftest_offline.py

This proves the plumbing works; real quality needs the real models in models.py.
"""
import hashlib
import os
import tempfile

import models  # noqa  (patched below)
from config import settings


# ---- deterministic fake embedder: hash tokens into a small fixed-dim vector ----
def fake_embed(texts, batch=128):
    dim = 64
    out = []
    for t in texts:
        v = [0.0] * dim
        for tok in t.lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % dim] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        out.append([x / norm for x in v])
    return out


def main():
    # point storage at temp dirs and shrink dim
    tmp = tempfile.mkdtemp()
    settings.qdrant_path = os.path.join(tmp, "qdrant")
    settings.bm25_path = os.path.join(tmp, "bm25.pkl")
    settings.md_dir = "sample_md"
    settings.embed_dim = 64

    # write a tiny corpus
    os.makedirs("sample_md", exist_ok=True)
    open("sample_md/Jon_Snow.md", "w").write(
        '---\ntitle: "Jon Snow"\nurl: https://x/Jon_Snow\nhouse: "House Stark"\n---\n\n'
        "# Jon Snow\n\nJon Snow is a bastard of House Stark.\n\n"
        "## Biography\n\nJon joins the Night's Watch and is elected Lord Commander.\n")
    open("sample_md/Daenerys.md", "w").write(
        '---\ntitle: "Daenerys Targaryen"\nurl: https://x/Daenerys\nhouse: "House Targaryen"\n---\n\n'
        "# Daenerys Targaryen\n\nDaenerys is the Mother of Dragons of House Targaryen.\n\n"
        "## Dragons\n\nHer dragons are Drogon, Rhaegal and Viserion.\n")

    # stub the network-bound model calls
    models.embed_texts = fake_embed
    models.embed_query = lambda t: fake_embed([t])[0]
    # rerank falls back to identity automatically (no Cohere key)

    import build_index
    build_index.build(embed_fn=fake_embed)

    import retrieval
    res = retrieval.hybrid_retrieve("Who are the dragons of the Mother of Dragons?", top_n=3)
    print("\nTop results:")
    for c in res:
        print(f"  {c['rank']}. {c['title']} — {c['section']}")

    assert any("Targaryen" in c["title"] for c in res), "expected Daenerys chunk in results"
    assert res[0]["url"].startswith("https://"), "payload/url missing"
    print("\nOK — chunking, Qdrant upsert/search, BM25, RRF and payloads all wired correctly.")


if __name__ == "__main__":
    main()
