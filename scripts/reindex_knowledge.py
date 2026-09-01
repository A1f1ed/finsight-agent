"""Rebuild the Chroma knowledge index (run after editing docs under knowledge/).

Usage:
    python scripts/reindex_knowledge.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finsight.rag import reindex

if __name__ == "__main__":
    n = reindex()
    print(f"Done. Collection now holds {n} chunks.")
