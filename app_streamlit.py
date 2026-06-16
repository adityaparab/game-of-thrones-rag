"""Browser chat UI.  streamlit run app_streamlit.py

Streams the answer and shows numbered, clickable sources under each reply.

On Railway, set RAG_API_URL to your FastAPI service URL so this UI stays
lightweight and does not rebuild the BM25 index on every container start.
"""
import json
import os
import urllib.error
import urllib.request
from typing import Iterator, List, Dict, Tuple

import streamlit as st

import agent
import rag
import retrieval


st.set_page_config(page_title="GoT Wiki Assistant", page_icon="🐉")
st.title("🐉 Game of Thrones Wiki Assistant")

mode = st.sidebar.radio("Mode", ["agentic", "simple"],
                        help="agentic = query rewriting + multi-hop re-retrieval")

RAG_API_URL = os.getenv("RAG_API_URL", "").strip().rstrip("/")


@st.cache_resource(show_spinner="Connecting to knowledge base...")
def _warm_local_backend() -> str:
    """Load Qdrant + BM25 once per process (can take a minute on first boot)."""
    retrieval._load()
    return "ready"


def _answer_local(question: str, chat_mode: str) -> Tuple[Iterator[str], List[Dict]]:
    _warm_local_backend()
    if chat_mode == "agentic":
        return agent.answer(question)
    return rag.answer(question)


def _answer_via_api(question: str, chat_mode: str) -> Tuple[Iterator[str], List[Dict]]:
    url = f"{RAG_API_URL}/chat"
    body = json.dumps({"question": question, "mode": chat_mode}).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    sources: List[Dict] = []

    def token_stream() -> Iterator[str]:
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    data = json.loads(payload)
                    if "token" in data:
                        yield data["token"]
                    elif "sources" in data:
                        sources.clear()
                        sources.extend(data["sources"])
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"API error {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Could not reach RAG API at {url}: {e.reason}") from e

    return token_stream(), sources


def _answer(question: str, chat_mode: str) -> Tuple[Iterator[str], List[Dict]]:
    if RAG_API_URL:
        return _answer_via_api(question, chat_mode)
    return _answer_local(question, chat_mode)


if "history" not in st.session_state:
    st.session_state.history = []

if RAG_API_URL:
    st.sidebar.caption(f"Backend: API ({RAG_API_URL})")
else:
    st.sidebar.caption("Backend: local (set RAG_API_URL on Railway for faster responses)")

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about Westeros..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        answer_text = ""
        sources: List[Dict] = []
        try:
            with st.spinner("Searching the wiki..."):
                stream, sources = _answer(prompt, mode)
            answer_text = st.write_stream(stream)
        except Exception as exc:
            st.error(f"Sorry, something went wrong: {exc}")
            if not RAG_API_URL and not os.getenv("QDRANT_URL", "").strip():
                st.info("Tip: set QDRANT_URL (and API keys) on this service, or point "
                        "RAG_API_URL at your FastAPI service instead.")
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    label = f"[{s['n']}] {s['title']} — {s['section']}"
                    st.markdown(f"- [{label}]({s['url']})" if s["url"] else f"- {label}")
    if answer_text:
        st.session_state.history.append({"role": "assistant", "content": answer_text})
