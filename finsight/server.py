"""FinSight Agent — FastAPI service: exposes the agent as an HTTP API.

Run:
    uvicorn finsight.server:app --reload

Interactive API docs:
    http://localhost:8000/docs

Design notes:
- agent.invoke is synchronous and slow (tens of seconds to minutes); endpoints are
  defined with plain sync def, and FastAPI runs them in a thread pool automatically,
  keeping the event loop unblocked;
- each thread_id maps to an isolated agent session and sandbox workspace;
  if thread_id is omitted a new session is created and returned in the response;
- analysis artifacts (charts/reports) are downloaded via
  /chat/{thread_id}/artifacts/{name}.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from langchain.messages import HumanMessage
from pydantic import BaseModel, Field

from .agent import OUTPUT_ROOT, create_finsight_agent, new_thread_id

app = FastAPI(
    title="FinSight Agent API",
    description=(
        "Finance research assistant powered by LangChain Deep Agents: "
        "RAG knowledge Q&A with citations + sandboxed data analysis "
        "that produces charts and markdown reports."
    ),
    version="1.0.0",
)

# Session cache: thread_id -> agent (in-memory; lost on process restart;
# in production, swap in a persistent checkpointer + a Redis session table)
_sessions: dict[str, object] = {}


# ---------------------------------------------------------------------------
# Request / response models (pydantic validation, auto-generated OpenAPI docs)
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str = Field(
        ..., description="The user's question", examples=["How is the Sharpe ratio calculated?"]
    )
    thread_id: str | None = Field(
        None, description="Session ID; a new session is created when omitted"
    )


class Artifact(BaseModel):
    name: str = Field(description="File name, e.g. report.md")
    suffix: str = Field(description="File extension, e.g. .png")
    size: int = Field(description="File size in bytes")


class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    artifacts: list[Artifact] = Field(
        default_factory=list,
        description="Artifacts newly delivered this turn; download them via /chat/{thread_id}/artifacts/{name}",
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_agent(thread_id: str):
    """Return the agent for this session, creating it on first use."""
    if thread_id not in _sessions:
        agent, _backend = create_finsight_agent(thread_id)
        _sessions[thread_id] = agent
    return _sessions[thread_id]


def _artifact_paths(thread_id: str) -> set[str]:
    """All artifact paths delivered by this session so far."""
    run_root = OUTPUT_ROOT / thread_id
    if not run_root.exists():
        return set()
    return {str(p) for p in run_root.rglob("*") if p.is_file()}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    """Health check (for deployment-platform liveness probes)."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Ask FinSight a question; knowledge Q&A takes ~1 min, data-analysis tasks ~3-5 min."""
    thread_id = req.thread_id or new_thread_id()
    agent = _get_agent(thread_id)

    before = _artifact_paths(thread_id)
    result = agent.invoke(
        {"messages": [HumanMessage(content=req.message)]},
        {"configurable": {"thread_id": thread_id}},
    )
    new_paths = sorted(_artifact_paths(thread_id) - before)

    artifacts = [
        Artifact(name=p.name, suffix=p.suffix, size=p.stat().st_size)
        for p in (Path(s) for s in new_paths)
    ]
    return ChatResponse(
        thread_id=thread_id,
        answer=result["messages"][-1].text,
        artifacts=artifacts,
    )


@app.get("/chat/{thread_id}/artifacts/{name}")
def download_artifact(thread_id: str, name: str) -> FileResponse:
    """Download a session artifact. The lookup is confined to the session's output
    directory to prevent path traversal."""
    run_root = OUTPUT_ROOT / thread_id
    if run_root.exists():
        for p in run_root.rglob("*"):
            if p.is_file() and p.name == name:
                return FileResponse(p)
    raise HTTPException(status_code=404, detail="Artifact not found")
