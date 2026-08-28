"""环境与模型配置：加载 .env，构建聊天模型与 embedding 模型。

所有密钥通过环境变量注入，不要把真实 key 提交到 Git。
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 优先加载项目根目录下的 .env
load_dotenv(PROJECT_ROOT / ".env")

# 聊天模型（通过阿里云百炼 OpenAI 兼容端点调用）
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3.7-plus")
# Embedding 模型
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3.7-text-embedding")


def get_chat_model():
    """构建聊天模型。使用 OpenAI 兼容模式接入阿里云百炼。"""
    return init_chat_model(
        model=CHAT_MODEL,
        model_provider="openai",
        base_url=os.getenv("DASHSCOPE_BASE_URL"),
        api_key=os.getenv("DASHSCOPE_API_KEY"),
    )


def get_embeddings() -> OpenAIEmbeddings:
    """构建 embedding 模型。

    注意两个兼容性要点（踩过坑）：
    1. 百炼兼容模式只接受字符串数组，必须关闭默认的 tiktoken 预分词
       （check_embedding_ctx_length=False），否则会发送 token id 数组报错；
    2. 百炼 embedding 接口限制单次最多 20 条文本，所以 OpenAIEmbeddings 的
       chunk_size（API 批量大小）必须设为 20。
    """
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=os.getenv("EMBEDDING_BASE_URL", os.getenv("DASHSCOPE_BASE_URL")),
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        check_embedding_ctx_length=False,
        chunk_size=20,
    )
