"""RAG 索引：加载本地金融知识库 -> 切分 -> embedding -> Chroma 持久化向量库。

知识库是 knowledge/ 目录下的 markdown 文件（技术指标、风险指标、港股常识）。

使用 Chroma 的嵌入式（embedded）模式：数据以文件形式持久化在 chroma_db/
目录（sqlite + HNSW 索引），无需安装或启动任何数据库服务。

行为：
- 首次启动：切分文档并调用 embedding API 建索引，写入 chroma_db/
- 之后启动：直接从磁盘打开集合，不再调用 embedding API（启动快、零成本）
- 知识库文件更新后：运行 python scripts/reindex_knowledge.py 重建索引
"""

import hashlib
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import PROJECT_ROOT, get_embeddings

KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "finsight_knowledge"


def load_knowledge_docs() -> list[Document]:
    """读取 knowledge/ 下所有 markdown 文件为 Document 列表。"""
    docs: list[Document] = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        docs.append(
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"source": f"knowledge/{path.name}"},
            )
        )
    return docs


def build_chunks() -> tuple[list[Document], list[str]]:
    """切分知识库文档，并生成确定性的块 ID。

    ID 由 来源文件 + 序号 + 内容哈希 决定：同样的内容重复写入不会产生重复
    向量，这是持久化向量库的基本要求（幂等写入）。
    """
    docs = load_knowledge_docs()
    # 文档切分大小：每块约 1000 字符，200 字符重叠，保证语义完整
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    ids = [
        hashlib.md5(
            f"{doc.metadata['source']}#{i}|{doc.page_content}".encode("utf-8")
        ).hexdigest()
        for i, doc in enumerate(splits)
    ]
    return splits, ids


def _collection_count(vector_store: Chroma) -> int:
    """查询集合中已有向量数。"""
    return len(vector_store.get(include=[]).get("ids", []))


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """打开持久化向量库：集合为空时自动建索引，否则直接复用磁盘数据。

    Chroma 嵌入式模式只需 persist_directory 指向本地目录，
    数据（sqlite 元数据 + HNSW 向量索引）自动落盘，进程重启不丢失。
    """
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
    count = _collection_count(vector_store)
    if count == 0:
        splits, ids = build_chunks()
        vector_store.add_documents(documents=splits, ids=ids)
        print(
            f"[rag] Indexed {len(splits)} chunks from {len(ids)} -> "
            f"persisted to {CHROMA_DIR.name}/"
        )
    else:
        print(
            f"[rag] Loaded Chroma collection '{COLLECTION_NAME}' "
            f"({count} chunks) from {CHROMA_DIR.name}/ — no re-embedding."
        )
    return vector_store


def reindex() -> int:
    """删除并重建知识库索引（修改了 knowledge/ 里的文档后运行）。"""
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
    store.delete_collection()

    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
    splits, ids = build_chunks()
    store.add_documents(documents=splits, ids=ids)
    print(f"[rag] Reindexed {len(splits)} chunks into '{COLLECTION_NAME}'.")
    return len(splits)
