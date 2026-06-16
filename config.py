"""Central configuration. All values overridable via environment / .env."""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# Which LLM powers chat/generation: "openai" or "deepseek". DeepSeek speaks the
# OpenAI-compatible API, so chat just points at a different base URL + key.
# (Embeddings have no DeepSeek equivalent and always use OpenAI — see below.)
_provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
_default_chat = "deepseek-chat" if _provider == "deepseek" else "gpt-4o"
_default_small = "deepseek-chat" if _provider == "deepseek" else "gpt-4o-mini"


@dataclass
class Settings:
    # ---- LLM provider for chat/generation ("openai" | "deepseek") ----
    llm_provider: str = _provider

    # ---- API keys ----
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")        # chat (if openai) + always embeddings
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")    # chat (if deepseek)
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    cohere_api_key: str = os.getenv("COHERE_API_KEY", "")  # optional; rerank disabled if blank

    # ---- Models (names move fast in 2026 — set these to whatever you run) ----
    embed_model: str = os.getenv("EMBED_MODEL", "text-embedding-3-large")
    embed_dim: int = int(os.getenv("EMBED_DIM", "3072"))          # 3072 for 3-large, 1536 for 3-small
    chat_model: str = os.getenv("CHAT_MODEL", _default_chat)      # final grounded answer (quality tier)
    small_model: str = os.getenv("SMALL_MODEL", _default_small)   # rewrite/decompose/grade (cheap tier)
    rerank_model: str = os.getenv("RERANK_MODEL", "rerank-v3.5")  # Cohere cross-encoder

    # ---- Storage ----
    qdrant_path: str = os.getenv("QDRANT_PATH", "./qdrant_data")          # local embedded mode
    qdrant_url: str = os.getenv("QDRANT_URL", "")                         # set this to use a server instead
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")                 # auth for a managed/remote Qdrant
    collection: str = os.getenv("QDRANT_COLLECTION", "got_wiki")
    bm25_path: str = os.getenv("BM25_PATH", "./bm25_index.pkl")
    md_dir: str = os.getenv("MD_DIR", "out_md")

    # ---- Retrieval params ----
    dense_top_k: int = int(os.getenv("DENSE_TOP_K", "30"))
    sparse_top_k: int = int(os.getenv("SPARSE_TOP_K", "30"))
    rerank_top_n: int = int(os.getenv("RERANK_TOP_N", "6"))
    rrf_k: int = 60

    # ---- Chunking ----
    chunk_max_chars: int = int(os.getenv("CHUNK_MAX_CHARS", "1800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # ---- Agent ----
    max_agent_loops: int = int(os.getenv("MAX_AGENT_LOOPS", "2"))


settings = Settings()
