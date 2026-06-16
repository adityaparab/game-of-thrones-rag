# Game of Thrones — Agentic RAG

Production-shaped RAG over your extracted Markdown wiki pages.
**Hybrid retrieval (dense + BM25) → Reciprocal Rank Fusion → Cohere rerank → grounded answer with citations**, wrapped in an **agentic loop** (query rewriting, multi-hop decomposition, sufficiency-graded re-retrieval).

LLM communication goes through **LangChain**; the agentic loop is a **LangGraph** state machine. Dependencies are managed with **uv**, and the app ships as a **Docker** image (FastAPI + Streamlit) deployable to **Railway**, with **Qdrant** running as its own container locally.

## Architecture

```
(manual) dump.xml.bz2 ──ingest_mwparser.py──► out_md/*.md      [separate, manual step]

out_md/*.md ──chunking.py──► chunks (text + metadata)
                                │
                build_index.py ─┼─► Qdrant (dense vectors)   ◄─ OpenAI embeddings (LangChain)
                                └─► BM25 index (sparse, pickled)

query ─► agent.py  LangGraph: rewrite → retrieve → grade → (re-retrieve loop) → finalize
            └─► retrieval.py  dense + BM25 → RRF → Cohere rerank (LangChain) → top-6
                    └─► rag.py  grounded prompt → LLM (LangChain) → answer + [n] citations
                            └─► cli.py · api.py · app_streamlit.py
```

## Setup

```bash
uv sync                       # runtime deps (LangChain, LangGraph, Qdrant, …)
uv sync --group dev           # + pyright (type checking)
cp .env.example .env          # add OPENAI_API_KEY (and COHERE_API_KEY for rerank)
```

Set `CHAT_MODEL` / `EMBED_MODEL` in `.env` to whatever models you run — names are
not hard-coded anywhere.

### Choosing the LLM provider (OpenAI or DeepSeek)

The chat/generation LLM is selected with one variable in `.env`:

```bash
LLM_PROVIDER=openai     # or: deepseek
```

- `openai` → uses `OPENAI_API_KEY` (default chat models `gpt-4o` / `gpt-4o-mini`).
- `deepseek` → uses `DEEPSEEK_API_KEY` + `DEEPSEEK_BASE_URL` (default model `deepseek-chat`).
  DeepSeek speaks the OpenAI-compatible API, so it plugs straight into the same code path.

Leave `CHAT_MODEL` / `SMALL_MODEL` commented out to get sensible per-provider defaults,
or set them to override.

> **Note:** embeddings have no DeepSeek equivalent, so `OPENAI_API_KEY` is **always**
> required to build and query the vector index — even when `LLM_PROVIDER=deepseek`.

## Step 1 — Ingest (separate, manual, run only when the corpus changes)

Ingestion converts a MediaWiki XML dump into structured Markdown. It is **not** part
of the app, the Docker stack, or `build_index.py`, and its dependencies live in an
optional extra so they never ship with the deployed service.

```bash
uv sync --extra ingest
uv run --extra ingest python ingest_mwparser.py gameofthrones_pages_current.xml.bz2 out_md/
```

## Step 2 — Build the index

Start a local Qdrant (its own container), then embed every chunk into it and build
the BM25 index:

```bash
docker compose up -d qdrant            # local Qdrant from Dockerfile.qdrant
QDRANT_URL=http://localhost:6333 python build_index.py
```

> Validate the wiring with **no API keys** first: `python selftest_offline.py`

## Step 3 — Run — three interfaces, same core

```bash
python cli.py                  # terminal chat (agentic). add --simple for single-shot
uvicorn api:app --port 8000    # POST /chat  (streaming SSE; mode: agentic|simple)
streamlit run app_streamlit.py # browser chat with clickable sources
```

## Docker (local)

The whole stack — Qdrant + FastAPI + Streamlit — runs from `docker-compose.yml`:

```bash
docker compose up --build              # qdrant (6333) + api (8000) + streamlit (8501)
docker compose up qdrant               # just the vector DB
docker compose run --rm api python build_index.py   # build the index inside the stack
```

- `Dockerfile` — the app image; serves FastAPI **or** Streamlit (`APP=api|streamlit`).
- `Dockerfile.qdrant` — a separate image for the local Qdrant database.

## Deploy (Railway)

`railway.toml` builds the app from `Dockerfile`. Railway injects `$PORT`, which the
entrypoint honours automatically. Qdrant is **not** deployed here — the app connects
to an external/managed Qdrant via `QDRANT_URL`.

1. Create a service from this repo (Dockerfile build).
2. Set variables: `OPENAI_API_KEY`, `QDRANT_URL` (+ `QDRANT_API_KEY`, `COHERE_API_KEY`).
3. For the Streamlit UI, add a second service from the same repo and set `APP=streamlit`
   (and clear the `/health` healthcheck path).

## Files

| File | Role |
|------|------|
| `config.py` | All settings / model names via env |
| `models.py` | LangChain wrappers: OpenAI/DeepSeek chat, OpenAI embeddings, Cohere rerank |
| `chunking.py` | Markdown → context-prefixed chunks + BM25 tokenizer |
| `build_index.py` | Embed → Qdrant; build BM25 |
| `retrieval.py` | Dense + sparse + RRF + rerank |
| `rag.py` | Grounded prompt, citations, streaming answer |
| `agent.py` | LangGraph state machine: rewrite / decompose / grade / re-retrieve |
| `cli.py` · `api.py` · `app_streamlit.py` | Interfaces |
| `ingest_mwparser.py` | **Manual** dump → Markdown ingestion (optional `ingest` extra) |
| `selftest_offline.py` | Pipeline smoke test, no keys needed |
| `Dockerfile` · `Dockerfile.qdrant` · `docker-compose.yml` · `railway.toml` | Containers / deploy |

## Dependencies

Managed entirely with **uv** (`pyproject.toml` + `uv.lock`):

- runtime: `uv sync`
- dev (pyright): `uv sync --group dev`
- ingestion only: `uv sync --extra ingest`

The first `uv sync` generates `uv.lock` (commit it for reproducible builds). Type-check with `uv run pyright`.

## Notes & next steps

- **Concurrency:** run Qdrant as a server (the local container, or a managed instance)
  and set `QDRANT_URL` so the API and Streamlit can share it.
- **Cost/latency knobs:** `RERANK_TOP_N`, `DENSE/SPARSE_TOP_K`, `CHUNK_MAX_CHARS`,
  model tiering (`SMALL_MODEL` does rewrite/grade), `MAX_AGENT_LOOPS`.
- **Evaluate it:** build a golden Q→source set and score with RAGAS / DeepEval; add
  LangSmith tracing (LangGraph emits traces out of the box when `LANGSMITH_*` is set).
