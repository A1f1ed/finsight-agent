"""Environment and model configuration: loads .env, builds the chat model and embeddings.

All secrets are injected via environment variables — never commit real keys to Git.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load the .env at the project root
load_dotenv(PROJECT_ROOT / ".env")

# Chat model (called through the Alibaba Cloud Bailian OpenAI-compatible endpoint)
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3.7-plus")
# Embedding model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3.7-text-embedding")


def get_chat_model():
    """Build the chat model, using OpenAI-compatible mode against Bailian."""
    return init_chat_model(
        model=CHAT_MODEL,
        model_provider="openai",
        base_url=os.getenv("DASHSCOPE_BASE_URL"),
        api_key=os.getenv("DASHSCOPE_API_KEY"),
    )


def get_embeddings() -> OpenAIEmbeddings:
    """Build the embedding model.

    Two compatibility pitfalls (hit in practice):
    1. The Bailian compatible endpoint only accepts string arrays, so the default
       tiktoken pre-tokenization must be disabled (check_embedding_ctx_length=False),
       otherwise token-id arrays are sent and rejected.
    2. The Bailian embedding API caps each batch at 20 texts, so OpenAIEmbeddings'
       chunk_size (the API batch size) must be set to 20.
    """
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=os.getenv("EMBEDDING_BASE_URL", os.getenv("DASHSCOPE_BASE_URL")),
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        check_embedding_ctx_length=False,
        chunk_size=20,
    )
