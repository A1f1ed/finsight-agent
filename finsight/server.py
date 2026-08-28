"""FinSight Agent — FastAPI 服务：把 agent 暴露为 HTTP API。

运行:
    uvicorn finsight.server:app --reload

交互式 API 文档:
    http://localhost:8000/docs

设计要点：
- agent.invoke 是同步且耗时的（数十秒到数分钟），端点用同步 def 定义，
  FastAPI 会自动把它放到线程池执行，不会阻塞事件循环；
- 每个 thread_id 对应独立的 agent 会话与沙箱工作区；
  thread_id 不传则自动创建新会话并随响应返回；
- 分析产物（图表/报告）通过 /chat/{thread_id}/artifacts/{name} 下载。
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

# 会话缓存: thread_id -> agent（内存态，进程重启即失效；
# 生产环境可换持久化 checkpointer + Redis 会话表）
_sessions: dict[str, object] = {}


# ---------------------------------------------------------------------------
# 请求 / 响应模型（pydantic 校验，自动生成 OpenAPI 文档）
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str = Field(
        ..., description="用户问题", examples=["夏普比率怎么计算？"]
    )
    thread_id: str | None = Field(
        None, description="会话 ID；不传则创建新会话"
    )


class Artifact(BaseModel):
    name: str = Field(description="文件名，如 report.md")
    suffix: str = Field(description="扩展名，如 .png")
    size: int = Field(description="文件大小（字节）")


class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    artifacts: list[Artifact] = Field(
        default_factory=list,
        description="本轮新产出的交付物，可用 /chat/{thread_id}/artifacts/{name} 下载",
    )


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------


def _get_agent(thread_id: str):
    """获取（或首次创建）会话对应的 agent。"""
    if thread_id not in _sessions:
        agent, _backend = create_finsight_agent(thread_id)
        _sessions[thread_id] = agent
    return _sessions[thread_id]


def _artifact_paths(thread_id: str) -> set[str]:
    """当前会话已交付的全部产物路径。"""
    run_root = OUTPUT_ROOT / thread_id
    if not run_root.exists():
        return set()
    return {str(p) for p in run_root.rglob("*") if p.is_file()}


# ---------------------------------------------------------------------------
# 端点
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    """健康检查（部署平台探活用）。"""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """向 FinSight 提问；知识问答约 1 分钟，数据分析任务约 3-5 分钟。"""
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
    """下载会话产物。遍历限定在该会话的 output 目录内，防止路径穿越。"""
    run_root = OUTPUT_ROOT / thread_id
    if run_root.exists():
        for p in run_root.rglob("*"):
            if p.is_file() and p.name == name:
                return FileResponse(p)
    raise HTTPException(status_code=404, detail="Artifact not found")
