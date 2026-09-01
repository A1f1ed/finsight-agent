"""RAG indexing: load the local finance knowledge base -> split -> embed -> persistent Chroma store.

The knowledge base is the markdown files under knowledge/ (technical indicators,
risk metrics, HK market basics).

Uses Chroma in embedded mode: data is persisted as files inside chroma_db/
(SQLite metadata + HNSW index) — no database service needs to be installed or started.

Behavior:
- First startup: splits the docs, calls the embedding API to build the index, writes chroma_db/
- Later startups: opens the collection straight from disk, no embedding API calls (fast, zero cost)
- After editing knowledge files: run python scripts/reindex_knowledge.py to rebuild
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
    """Read every markdown file under knowledge/ into a list of Documents."""
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
    """Split the knowledge docs and generate deterministic chunk IDs.

    The ID is derived from source file + index + content hash: re-ingesting the
    same content never creates duplicate vectors — idempotent writes, a basic
    requirement for a persistent vector store.
    """
    docs = load_knowledge_docs()
    # Chunking: ~1000 characters per chunk with 200-character overlap, keeping semantics intact
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
    """Number of vectors already in the collection."""
    return len(vector_store.get(include=[]).get("ids", []))


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """Open the persistent vector store: build the index if empty, otherwise reuse disk data.

    Chroma embedded mode only needs persist_directory pointing at a local folder;
    data (SQLite metadata + HNSW index) is written to disk automatically and
    survives process restarts.
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
    """Drop and rebuild the knowledge index (run after editing files under knowledge/)."""
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
