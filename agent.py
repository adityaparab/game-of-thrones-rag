"""Agentic retrieval implemented as a LangGraph state machine.

Graph:
  rewrite  -> turn the raw question into clean search queries (+ decompose multi-hop)
  retrieve -> hybrid search each pending query, dedupe chunks into the running pool
  grade    -> is the evidence sufficient? if not, queue a follow-up query (loops up to
              MAX_AGENT_LOOPS via a conditional edge back into `retrieve`)
  finalize -> rerank the merged pool against the original question for a clean top-N

`generate` (the grounded, streamed answer) is handled by rag.answer once the graph
has produced the final chunk set. The small/cheap model does rewrite+grade; the
quality model writes the answer.
"""
import json
from typing import List, Dict, Tuple, Iterator, TypedDict

from langgraph.graph import StateGraph, START, END

import models
from config import settings
from retrieval import hybrid_retrieve
import rag


# --------------------------------------------------------------------------- #
# LLM helpers (cheap model: rewrite + grade)                                   #
# --------------------------------------------------------------------------- #
def _json_call(prompt: str, fallback: dict) -> dict:
    """Call the cheap model and parse a JSON object, tolerantly."""
    try:
        raw = models.chat(
            [{"role": "system", "content": "Respond with a single minified JSON object only."},
             {"role": "user", "content": prompt}],
            model=settings.small_model, temperature=0.0,
        )
        start, end = raw.find("{"), raw.rfind("}")
        return json.loads(raw[start:end + 1])
    except Exception:
        return fallback


def rewrite(question: str) -> List[str]:
    """Return a list of search queries (1 for simple, 2-4 for multi-hop questions)."""
    data = _json_call(
        f'Rewrite this Game of Thrones question into precise wiki search queries. '
        f'If it requires multiple facts (multi-hop), split into sub-queries.\n'
        f'Question: "{question}"\n'
        f'JSON schema: {{"queries": ["..."]}} (max 4).',
        fallback={"queries": [question]},
    )
    queries = [q for q in data.get("queries", []) if isinstance(q, str) and q.strip()]
    return queries[:4] or [question]


def grade(question: str, chunks: List[Dict]) -> Tuple[bool, str]:
    """Decide whether retrieved evidence is sufficient; if not, suggest a follow-up query."""
    snippets = "\n".join(f"- {c['title']} — {c['section']}: {c['text'][:200]}" for c in chunks[:8])
    data = _json_call(
        f'Question: "{question}"\nRetrieved evidence:\n{snippets}\n\n'
        f'Is this enough to fully answer the question? '
        f'JSON: {{"sufficient": true|false, "followup_query": "..."}}',
        fallback={"sufficient": True, "followup_query": ""},
    )
    return bool(data.get("sufficient", True)), (data.get("followup_query") or "").strip()


def _merge(existing: List[Dict], new: List[Dict]) -> List[Dict]:
    seen = {(c["title"], c["section"]) for c in existing}
    for c in new:
        if (c["title"], c["section"]) not in seen:
            existing.append(c)
            seen.add((c["title"], c["section"]))
    return existing


# --------------------------------------------------------------------------- #
# LangGraph state + nodes                                                      #
# --------------------------------------------------------------------------- #
class AgentState(TypedDict):
    question: str
    pending_queries: List[str]      # queries still to retrieve
    collected: List[Dict]           # running, deduped chunk pool
    loops: int
    chunks: List[Dict]              # final reranked top-N
    verbose: bool


def rewrite_node(state: AgentState) -> dict:
    queries = rewrite(state["question"])
    if state.get("verbose"):
        print(f"  [agent] sub-queries: {queries}")
    return {"pending_queries": queries, "collected": [], "loops": 0}


def retrieve_node(state: AgentState) -> dict:
    collected = state["collected"]
    for q in state["pending_queries"]:
        _merge(collected, hybrid_retrieve(q, top_n=settings.rerank_top_n))
    return {"collected": collected, "pending_queries": []}


def grade_node(state: AgentState) -> dict:
    if state["loops"] >= settings.max_agent_loops:
        return {"pending_queries": []}
    ok, followup = grade(state["question"], state["collected"])
    if ok or not followup:
        return {"pending_queries": []}
    if state.get("verbose"):
        print(f"  [agent] loop {state['loops'] + 1}: insufficient -> '{followup}'")
    return {"pending_queries": [followup], "loops": state["loops"] + 1}


def finalize_node(state: AgentState) -> dict:
    collected = state["collected"]
    reranked = models.rerank(state["question"], [c["text"] for c in collected], settings.rerank_top_n)
    return {"chunks": [collected[i] for i, _ in reranked]}


def _route_after_grade(state: AgentState) -> str:
    """Loop back to retrieve if the grader queued a follow-up, else finalize."""
    return "retrieve" if state["pending_queries"] else "finalize"


def _build_graph():
    g = StateGraph(AgentState)
    g.add_node("rewrite", rewrite_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade", grade_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "rewrite")
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", _route_after_grade,
                            {"retrieve": "retrieve", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()


_GRAPH = _build_graph()


# --------------------------------------------------------------------------- #
# Public API (unchanged signatures)                                           #
# --------------------------------------------------------------------------- #
def retrieve_agentic(question: str, verbose: bool = False) -> List[Dict]:
    """Run the agent graph and return the final ranked chunk set (no generation)."""
    final = _GRAPH.invoke({
        "question": question,
        "pending_queries": [],
        "collected": [],
        "loops": 0,
        "chunks": [],
        "verbose": verbose,
    })
    return final["chunks"]


def answer(question: str, verbose: bool = False) -> Tuple[Iterator[str], List[Dict]]:
    chunks = retrieve_agentic(question, verbose=verbose)
    return rag.answer(question, chunks=chunks)
