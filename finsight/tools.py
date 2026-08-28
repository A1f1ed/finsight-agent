"""自定义工具：知识库检索 + 报告交付。

- search_knowledge: 向量检索知识库，把命中片段写入 agent 文件系统 /retrieved/，
  供子代理读取分析（沿用 Deep Agents 的 "retrieve, offload, delegate" 模式）；
- publish_report: 把分析产物（图表 PNG / 报告 Markdown）从 agent 工作区导出到
  本地 output/ 目录，供前端界面展示。
"""

import uuid
from pathlib import Path

from langchain.tools import tool


def make_tools(vector_store, backend, output_dir: Path):
    """工厂函数：把向量库、后端和输出目录绑定进工具闭包。"""

    @tool(parse_docstring=True)
    def search_knowledge(query: str) -> str:
        """Search the finance knowledge base and save matching chunks to the agent filesystem.

        Args:
            query: Natural language search query.

        Returns:
            File paths where retrieved chunks were saved under /retrieved/.
        """
        retrieved_docs = vector_store.similarity_search(query, k=4)
        batch_id = uuid.uuid4().hex[:8]
        uploads: list[tuple[str, bytes]] = []
        saved_paths: list[str] = []

        for index, doc in enumerate(retrieved_docs, start=1):
            path = f"/retrieved/{batch_id}/chunk_{index}.md"
            content = (
                f"# Source: {doc.metadata.get('source', 'unknown')}\n\n"
                f"{doc.page_content}"
            )
            uploads.append((path, content.encode("utf-8")))
            saved_paths.append(path)

        backend.upload_files(uploads)
        return (
            f"Saved {len(saved_paths)} knowledge chunks:\n" + "\n".join(saved_paths)
        )

    @tool(parse_docstring=True)
    def publish_report(file_paths: list[str], summary: str) -> str:
        """Export analysis artifacts from the agent workspace to the user-visible output folder.

        Call this at the end of every data analysis task, after charts and the
        markdown report are generated.

        Args:
            file_paths: Backend paths of artifacts to export, e.g. ["/output/report.md", "/output/dashboard.png"].
            summary: One-sentence description of the delivered report.

        Returns:
            Local filesystem paths of the exported artifacts.
        """
        run_dir = output_dir / uuid.uuid4().hex[:8]
        run_dir.mkdir(parents=True, exist_ok=True)

        downloaded = backend.download_files(list(file_paths))
        local_paths: list[str] = []
        errors: list[str] = []
        for item in downloaded:
            if item.content is None:
                errors.append(f"{item.path}: {item.error or 'unknown error'}")
                continue
            # backend 路径形如 /output/report.md，只取文件名落到 run_dir
            local_path = run_dir / Path(item.path).name
            local_path.write_bytes(item.content)
            local_paths.append(str(local_path))

        if not local_paths:
            return "Failed to publish artifacts: " + "; ".join(errors)

        result = f"Published artifacts ({summary}):\n" + "\n".join(local_paths)
        if errors:
            result += "\nFailed files: " + "; ".join(errors)
        return result

    return [search_knowledge, publish_report]
