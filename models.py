"""LLM communication through LangChain. Centralised so a provider swap is one file.

  embed_texts / embed_query  -> OpenAI embeddings   (langchain-openai)
  rerank                     -> Cohere rerank        (langchain-cohere; graceful no-op if no key)
  chat / chat_stream         -> OpenAI *or* DeepSeek chat, picked by LLM_PROVIDER

DeepSeek exposes an OpenAI-compatible API, so chat reuses ``ChatOpenAI`` pointed at
DeepSeek's base URL with its own key. Embeddings have no DeepSeek equivalent and
always use OpenAI (so OPENAI_API_KEY is still required to build/query the index).

Public function signatures are unchanged, so retrieval/rag/agent/build_index keep
working as before — only the implementation now goes through LangChain.
"""
import time
from typing import List, Tuple, Optional, Iterator, Any, Dict

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import settings

# --------------------------------------------------------------------------- #
# Lazily-built singletons (so importing this module never requires API keys)   #
# --------------------------------------------------------------------------- #
_embeddings: Optional[OpenAIEmbeddings] = None
_chat_cache: Dict[Tuple[str, float], ChatOpenAI] = {}
_reranker: Any = None
_rerank_ready = False


def _get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    assert settings.openai_api_key, "OPENAI_API_KEY not set"
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model=settings.embed_model,
            api_key=settings.openai_api_key,  # type: ignore[arg-type]
            max_retries=8,
        )
    return _embeddings


def _get_chat(model: str, temperature: float) -> ChatOpenAI:
    key = (model, temperature)
    if key not in _chat_cache:
        if settings.llm_provider == "deepseek":
            assert settings.deepseek_api_key, "DEEPSEEK_API_KEY not set (LLM_PROVIDER=deepseek)"
            _chat_cache[key] = ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=settings.deepseek_api_key,  # type: ignore[arg-type]
                base_url=settings.deepseek_base_url,
            )
        else:
            assert settings.openai_api_key, "OPENAI_API_KEY not set"
            _chat_cache[key] = ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=settings.openai_api_key,  # type: ignore[arg-type]
            )
    return _chat_cache[key]


def _get_reranker():
    """Build the Cohere reranker once. Returns None if no key is configured."""
    global _reranker, _rerank_ready
    if not _rerank_ready:
        _rerank_ready = True
        if settings.cohere_api_key:
            try:
                from langchain_cohere import CohereRerank

                _reranker = CohereRerank(
                    model=settings.rerank_model,
                    cohere_api_key=settings.cohere_api_key,  # type: ignore[arg-type]
                )
            except Exception:
                _reranker = None
    return _reranker


# --------------------------------------------------------------------------- #
# Embeddings                                                                   #
# --------------------------------------------------------------------------- #
def embed_texts(
    texts: List[str],
    batch: int = 128,
    pause: float = 0.4,
    max_attempts: int = 8,
) -> List[List[float]]:
    """Embed texts in batches, throttled and resilient to TPM rate limits.

    ``pause`` is a small sleep between successful batches to keep the rolling
    tokens-per-minute usage under the account limit. On a 429 we honour the
    server's retry hint (or back off exponentially) and retry the same batch.
    """
    from openai import RateLimitError

    emb = _get_embeddings()
    vectors: List[List[float]] = []
    total = len(texts)
    for i in range(0, total, batch):
        chunk = texts[i:i + batch]
        for attempt in range(max_attempts):
            try:
                vectors.extend(emb.embed_documents(chunk))
                break
            except RateLimitError as exc:
                if attempt == max_attempts - 1:
                    raise
                wait = _retry_after_seconds(exc, default=2.0 * (2 ** attempt))
                print(
                    f"  rate limited at {i + len(chunk)}/{total}; "
                    f"retrying in {wait:.1f}s (attempt {attempt + 1}/{max_attempts})"
                )
                time.sleep(wait)
        else:  # pragma: no cover - loop always breaks or raises
            raise RuntimeError("embedding retries exhausted")
        if pause:
            time.sleep(pause)
    return vectors


def _retry_after_seconds(exc: Exception, default: float) -> float:
    """Pull a retry delay from an OpenAI error, falling back to ``default``."""
    try:
        headers = getattr(getattr(exc, "response", None), "headers", None) or {}
        for key in ("retry-after-ms",):
            if key in headers:
                return max(0.2, float(headers[key]) / 1000.0)
        if "retry-after" in headers:
            return max(0.2, float(headers["retry-after"]))
    except (TypeError, ValueError):
        pass
    return default


def embed_query(text: str) -> List[float]:
    return _get_embeddings().embed_query(text)


# --------------------------------------------------------------------------- #
# Reranking (Cohere cross-encoder via LangChain)                              #
# --------------------------------------------------------------------------- #
def rerank(query: str, documents: List[str], top_n: int) -> List[Tuple[int, float]]:
    """Return [(original_index, relevance_score), ...] best-first.
    Falls back to identity order if Cohere is not configured."""
    reranker = _get_reranker()
    if not reranker or not documents:
        return [(i, 1.0) for i in range(min(top_n, len(documents)))]
    reranker.top_n = min(top_n, len(documents))
    results = reranker.rerank(documents, query)
    return [(r["index"], float(r["relevance_score"])) for r in results]


# --------------------------------------------------------------------------- #
# Chat                                                                         #
# --------------------------------------------------------------------------- #
def chat(messages, model: Optional[str] = None, temperature: float = 0.1, **kw) -> str:
    llm = _get_chat(model or settings.chat_model, temperature)
    result: BaseMessage = llm.invoke(messages, **kw)
    content = result.content
    return content if isinstance(content, str) else str(content)


def chat_stream(messages, model: Optional[str] = None, temperature: float = 0.1, **kw) -> Iterator[str]:
    llm = _get_chat(model or settings.chat_model, temperature)
    for chunk in llm.stream(messages, **kw):
        content = chunk.content
        if content:
            yield content if isinstance(content, str) else str(content)
