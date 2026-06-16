"""FastAPI service with a streaming /chat endpoint.

  uvicorn api:app --reload --port 8000

  curl -N -X POST localhost:8000/chat \
       -H 'content-type: application/json' \
       -d '{"question":"Who killed the Night King?","mode":"agentic"}'

Note: local embedded Qdrant is single-process. For real concurrent serving,
run Qdrant as a server (set QDRANT_URL) so API + Streamlit can share it.
"""
import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import rag
import agent

app = FastAPI(title="GoT Agentic RAG")


class Query(BaseModel):
    question: str
    mode: str = "agentic"   # "agentic" | "simple"


@app.post("/chat")
def chat(q: Query):
    stream, sources = (agent.answer(q.question) if q.mode == "agentic"
                       else rag.answer(q.question))

    def gen():
        # stream answer tokens as SSE, then a final 'sources' event
        for tok in stream:
            yield f"data: {json.dumps({'token': tok})}\n\n"
        yield f"data: {json.dumps({'sources': sources})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/health")
def health():
    return {"status": "ok"}
