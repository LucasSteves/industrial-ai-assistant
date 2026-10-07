"""Indexa (do zero) todos os PDFs da pasta /knowledge pelo terminal.

Uso:  python scripts/indexar.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import config  # noqa: E402
from rag.vector_store import KnowledgeBase  # noqa: E402


def main():
    print(f"Pasta: {config.KNOWLEDGE_DIR}")
    kb = KnowledgeBase()
    result = kb.index_folder(config.KNOWLEDGE_DIR, reset=True)
    if not result:
        print("Nenhum PDF encontrado.")
        return
    for name, n in result.items():
        print(f"  {name:<40} {n:>4} chunks")
    print(f"Total na base: {kb.count()} chunks")


if __name__ == "__main__":
    main()
