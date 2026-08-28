"""重建 Chroma 知识库索引（修改了 knowledge/ 里的文档后运行）。

用法:
    python scripts/reindex_knowledge.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finsight.rag import reindex

if __name__ == "__main__":
    n = reindex()
    print(f"Done. Collection now holds {n} chunks.")
