"""Etapas 1 e 2 do pipeline: PDF Loader -> Chunking.

Cada página do PDF vira um Document do LangChain com o número da página
(começando em 1, como aparece no rodapé do PDF). O chunking é feito por
página, então todo chunk sabe exatamente de qual arquivo e página veio:
é isso que permite a rastreabilidade das fontes nas respostas.
"""
import hashlib
import warnings
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .config import CHUNK_OVERLAP, CHUNK_SIZE

try:  # PDF Loader do LangChain (langchain-community)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from langchain_community.document_loaders import PyPDFLoader
except ImportError:  # fallback caso o pacote não esteja disponível
    PyPDFLoader = None


def _load_pages(path: Path) -> List[Document]:
    if PyPDFLoader is not None:
        return PyPDFLoader(str(path)).load()
    reader = PdfReader(str(path))
    return [Document(page_content=pg.extract_text() or "", metadata={"page": i})
            for i, pg in enumerate(reader.pages)]


def list_pdfs(folder: Path) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.glob("*.pdf") if p.is_file())


def _document_title(path: Path) -> str:
    """Título do documento: metadado do PDF ou, se não houver, o nome do arquivo."""
    try:
        meta = PdfReader(str(path)).metadata
        if meta and meta.title and meta.title.strip():
            return meta.title.strip()
    except Exception:
        pass
    return path.stem.replace("_", " ").strip()


def load_pdf(path: Path) -> List[Document]:
    """Lê um PDF e devolve um Document por página, com metadados padronizados."""
    path = Path(path)
    title = _document_title(path)
    pages = _load_pages(path)

    docs = []
    for d in pages:
        text = " ".join((d.page_content or "").split())  # normaliza espaços e quebras
        if not text:
            continue
        page_number = int(d.metadata.get("page", 0)) + 1  # PyPDFLoader conta a partir de 0
        docs.append(Document(
            page_content=text,
            metadata={"source_file": path.name, "title": title, "page": page_number},
        ))
    return docs


def split_documents(pages: List[Document]) -> List[Document]:
    """Divide as páginas em chunks e gera um ID determinístico para cada um.

    O ID determinístico (hash de arquivo + página + posição + conteúdo) faz com
    que reindexar o mesmo arquivo não duplique trechos no banco vetorial.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )
    chunks = []
    for page in pages:
        for idx, piece in enumerate(splitter.split_text(page.page_content)):
            meta = dict(page.metadata)
            meta["chunk"] = idx
            raw = f'{meta["source_file"]}|{meta["page"]}|{idx}|{piece}'
            meta["chunk_id"] = hashlib.sha1(raw.encode("utf-8")).hexdigest()
            chunks.append(Document(page_content=piece, metadata=meta))
    return chunks
