"""Core RAG: build a grounded prompt from retrieved chunks and generate an
answer with inline [n] citations. `answer()` streams tokens; `sources` are
returned alongside so any interface can render them."""
from typing import List, Dict, Iterator, Tuple, Optional

import models
from retrieval import hybrid_retrieve

SYSTEM = (
    "You are a Game of Thrones lore expert answering from a wiki knowledge base.\n"
    "Rules:\n"
    "- Answer ONLY using the numbered context passages provided.\n"
    "- Cite every claim with its source number in square brackets, e.g. [2].\n"
    "- If the context does not contain the answer, say you don't know — do not guess.\n"
    "- Be concise and specific. Prefer names, houses, and episodes from the context."
)


def build_context(chunks: List[Dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(f"[{i}] {c['title']} — {c['section']}\n{c['text']}")
    return "\n\n".join(blocks)


def _messages(question: str, chunks: List[Dict]):
    context = build_context(chunks)
    user = f"Context passages:\n\n{context}\n\nQuestion: {question}\n\nAnswer with citations:"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def sources_of(chunks: List[Dict]) -> List[Dict]:
    seen, out = set(), []
    for i, c in enumerate(chunks, 1):
        key = c["url"] or c["title"]
        out.append({"n": i, "title": c["title"], "section": c["section"], "url": c["url"]})
    return out


def answer(question: str, chunks: Optional[List[Dict]] = None) -> Tuple[Iterator[str], List[Dict]]:
    """Return (token_stream, sources). Pass `chunks` to reuse an agent's retrieval."""
    if chunks is None:
        chunks = hybrid_retrieve(question)
    stream = models.chat_stream(_messages(question, chunks))
    return stream, sources_of(chunks)


def answer_text(question: str, chunks: Optional[List[Dict]] = None) -> Tuple[str, List[Dict]]:
    """Non-streaming convenience wrapper."""
    if chunks is None:
        chunks = hybrid_retrieve(question)
    text = models.chat(_messages(question, chunks))
    return text, sources_of(chunks)
