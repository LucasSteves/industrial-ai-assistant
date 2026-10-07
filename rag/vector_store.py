"""Etapas 3 e 4 do pipeline: Embeddings -> Vector DB (ChromaDB) + busca.

A busca é HÍBRIDA:
  * busca semântica normal (similaridade de cosseno entre embeddings);
  * busca literal para códigos técnicos citados na pergunta (E045, X200,
    NR-12, RC-75...). Modelos de embedding pequenos "enxergam" mal códigos,
    então um trecho que contém literalmente o código pedido ganha um bônus
    de relevância. Isso evita, por exemplo, responder o erro E044 quando o
    operador perguntou do E045.
"""
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from . import config
from .loader import list_pdfs, load_pdf, split_documents

# Código técnico = 1 a 4 letras + (hífen opcional) + 2 a 5 dígitos. Ex.: E045, X200, NR-12, RC-75, O9100
_CODE_RE = re.compile(r"(?<![\w-])([A-Za-z]{1,4})-?(\d{2,5})(?![\w-])")


@dataclass
class RetrievedChunk:
    text: str
    source_file: str
    title: str
    page: int
    score: float          # relevância final (0 a 1, maior = mais parecido)
    keyword_match: bool   # True se o trecho contém literalmente um código da pergunta
    chunk_id: str

    def to_dict(self) -> Dict:
        return asdict(self)


def _normalize_code(s: str) -> str:
    return re.sub(r"[\s\-]", "", s).upper()


def extract_codes(text: str) -> List[str]:
    """Extrai códigos técnicos da pergunta, já normalizados (sem hífen, maiúsculos)."""
    found = []
    for letters, digits in _CODE_RE.findall(text or ""):
        code = (letters + digits).upper()
        if code not in found:
            found.append(code)
    return found


def _code_variants(code: str) -> List[str]:
    """Formas como o código pode aparecer escrito no documento (busca literal é case-sensitive)."""
    m = re.match(r"([A-Z]+)(\d+)", code)
    if not m:
        return [code]
    letters, digits = m.groups()
    variants = {f"{letters}{digits}", f"{letters}-{digits}"}
    if len(digits) < 3:  # E45 -> E045
        variants |= {f"{letters}{digits.zfill(3)}", f"{letters}-{digits.zfill(3)}"}
    return sorted(variants)


def missing_codes(question: str, chunks: List[RetrievedChunk]) -> List[str]:
    """Códigos citados na pergunta que NÃO aparecem em nenhum trecho recuperado.

    Usado pelo controle de alucinação: se o operador pergunta do erro E999 e o
    E999 não existe na documentação, a resposta é bloqueada antes de chegar ao LLM.
    """
    corpus = _normalize_code(" ".join(c.text for c in chunks))
    missing = []
    for code in extract_codes(question):
        if not any(_normalize_code(v) in corpus for v in _code_variants(code)):
            missing.append(code)
    return missing


def get_embeddings() -> Embeddings:
    """Modelo de embeddings local (baixado do Hugging Face na primeira execução)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


class KnowledgeBase:
    """Base de conhecimento vetorial persistida em disco (pasta dados/chroma)."""

    def __init__(self, embeddings: Optional[Embeddings] = None,
                 persist_dir: Path = config.CHROMA_DIR,
                 collection_name: str = config.COLLECTION_NAME):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.embeddings = embeddings or get_embeddings()
        self.store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(persist_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------ info
    def count(self) -> int:
        return self.store._collection.count()

    def indexed_files(self) -> Dict[str, int]:
        """{arquivo: quantidade de chunks} de tudo que está no banco."""
        data = self.store.get(include=["metadatas"])
        counts: Dict[str, int] = {}
        for m in data.get("metadatas") or []:
            name = (m or {}).get("source_file", "?")
            counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items()))

    # ------------------------------------------------------------- ingestão
    def remove_file(self, file_name: str) -> None:
        self.store._collection.delete(where={"source_file": file_name})

    def index_file(self, path: Path) -> int:
        """Indexa (ou reindexa) um PDF. Os chunks antigos do mesmo arquivo são substituídos."""
        path = Path(path)
        chunks = split_documents(load_pdf(path))
        self.remove_file(path.name)
        if not chunks:
            return 0
        ids = [c.metadata["chunk_id"] for c in chunks]
        batch = 64
        for i in range(0, len(chunks), batch):
            self.store.add_documents(chunks[i:i + batch], ids=ids[i:i + batch])
        return len(chunks)

    def index_folder(self, folder: Path = config.KNOWLEDGE_DIR, reset: bool = False) -> Dict[str, int]:
        """Indexa todos os PDFs da pasta. reset=True apaga a base antes."""
        if reset:
            self.reset()
        result = {}
        for pdf in list_pdfs(folder):
            result[pdf.name] = self.index_file(pdf)
        return result

    def reset(self) -> None:
        self.store.reset_collection()

    # ---------------------------------------------------------------- busca
    def _query(self, query: str, k: int, where_document=None):
        kwargs = {"k": k}
        if where_document:
            kwargs["where_document"] = where_document
        # distância de cosseno -> relevância = 1 - distância
        return self.store.similarity_search_with_score(query, **kwargs)

    def search(self, query: str, k: int = config.TOP_K) -> List[RetrievedChunk]:
        total = self.count()
        if total == 0:
            return []
        k = min(k, total)

        codes = extract_codes(query)
        found: Dict[str, RetrievedChunk] = {}

        def add(doc, distance):
            m = doc.metadata
            cid = m.get("chunk_id") or f'{m.get("source_file")}|{m.get("page")}|{m.get("chunk")}'
            if cid in found:
                return
            semantic = max(0.0, min(1.0, 1.0 - float(distance)))
            # quantos códigos da pergunta aparecem literalmente neste trecho
            text_norm = _normalize_code(doc.page_content)
            matches = sum(
                1 for code in codes
                if any(_normalize_code(v) in text_norm for v in _code_variants(code))
            )
            score = min(1.0, semantic + config.KEYWORD_BOOST * matches)
            found[cid] = RetrievedChunk(
                text=doc.page_content, source_file=m.get("source_file", "?"),
                title=m.get("title", m.get("source_file", "?")), page=int(m.get("page", 0)),
                score=round(score, 4), keyword_match=matches > 0, chunk_id=cid,
            )

        # 1) busca semântica
        for doc, dist in self._query(query, k):
            add(doc, dist)

        # 2) busca literal pelos códigos técnicos da pergunta
        for code in codes:
            for variant in _code_variants(code):
                try:
                    hits = self._query(query, k, where_document={"$contains": variant})
                except Exception:
                    hits = []
                for doc, dist in hits:
                    add(doc, dist)

        ranked = sorted(found.values(), key=lambda c: c.score, reverse=True)
        return self._expand_to_page(ranked)[:k]

    # ------------------------------------------------- página completa
    def _expand_to_page(self, ranked: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Junta a cada trecho os demais trechos da mesma página.

        Uma página longa vira 2+ chunks; sem isto, um procedimento pode chegar
        cortado ao LLM (ex.: E045 na pág. 34 — o passo 3 fica no 2º chunk, que
        não contém o código nem se parece com a pergunta). Mantém um único
        resultado por página, com a maior pontuação.
        """
        merged: List[RetrievedChunk] = []
        seen = set()
        for c in ranked:
            key = (c.source_file, c.page)
            if key in seen:
                continue
            seen.add(key)
            data = self.store.get(
                where={"$and": [{"source_file": c.source_file}, {"page": c.page}]},
                include=["documents", "metadatas"],
            )
            parts = sorted(zip(data.get("metadatas") or [], data.get("documents") or []),
                           key=lambda p: int((p[0] or {}).get("chunk", 0)))
            if len(parts) > 1:
                text = parts[0][1]
                for _, nxt in parts[1:]:
                    text = _join_overlap(text, nxt)
                c = RetrievedChunk(**{**c.to_dict(), "text": text})
            merged.append(c)
        return merged


def _join_overlap(a: str, b: str, max_overlap: int = config.CHUNK_OVERLAP + 50) -> str:
    """Concatena dois chunks consecutivos removendo o trecho repetido pelo chunk_overlap."""
    for n in range(min(len(a), len(b), max_overlap), 0, -1):
        if a.endswith(b[:n]):
            return a + b[n:]
    return a + "\n" + b
