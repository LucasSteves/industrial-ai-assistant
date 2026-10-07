"""Testes offline do pipeline (não precisam de Ollama nem de internet).

Usam um embedding "de mentirinha" (bag-of-words com hash) e um LLM falso,
para validar: loader + páginas, busca híbrida, controle de alucinação,
citação de fontes e reescrita com histórico.

Rodar:  python -m pytest -q
"""
import hashlib
import math
import re
import unicodedata

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag import config
from rag.loader import list_pdfs, load_pdf, split_documents
from rag.rag_chain import IndustrialAssistant, extract_citations, is_not_found, strip_citations
from rag.vector_store import KnowledgeBase, extract_codes


class HashEmbeddings(Embeddings):
    """Embedding determinístico por palavras: suficiente para testar a lógica."""
    DIM = 512

    def _vec(self, text):
        t = "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")
        v = [0.0] * self.DIM
        for w in re.findall(r"\w{3,}", t):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.DIM] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class ExplodingLLM(FakeListChatModel):
    """LLM que falha se for chamado: prova que o bloqueio aconteceu antes da geração."""
    def _call(self, *a, **k):
        raise AssertionError("O LLM não deveria ter sido chamado")


@pytest.fixture(scope="module")
def kb(tmp_path_factory):
    base = KnowledgeBase(embeddings=HashEmbeddings(), persist_dir=tmp_path_factory.mktemp("chroma"),
                         collection_name="teste")
    base.index_folder(config.KNOWLEDGE_DIR, reset=True)
    return base


def test_knowledge_folder_has_the_5_pdfs():
    names = [p.name for p in list_pdfs(config.KNOWLEDGE_DIR)]
    assert names == ["01_manual_cnc.pdf", "02_procedimento_seguranca.pdf", "03_manual_compressor.pdf",
                     "04_politica_manutencao.pdf", "05_checklist_operacional.pdf"]


def test_loader_keeps_real_page_numbers():
    pages = load_pdf(config.KNOWLEDGE_DIR / "01_manual_cnc.pdf")
    e045 = [p for p in pages if "E045" in p.page_content]
    assert e045 and e045[0].metadata["page"] == 34
    chunks = split_documents(pages)
    assert all(len(c.page_content) <= config.CHUNK_SIZE for c in chunks)
    assert len({c.metadata["chunk_id"] for c in chunks}) == len(chunks)


def test_reindexing_does_not_duplicate(kb):
    before = kb.count()
    kb.index_file(config.KNOWLEDGE_DIR / "04_politica_manutencao.pdf")
    assert kb.count() == before


def test_extract_codes():
    assert extract_codes("A CNC modelo X200 deu erro e045 conforme NR-12") == ["X200", "E045", "NR12"]


def test_hybrid_search_finds_exact_error_code(kb):
    hits = kb.search("A máquina CNC modelo X200 apresentou erro E045. Qual procedimento devo realizar?", k=5)
    assert hits[0].source_file == "01_manual_cnc.pdf" and hits[0].page == 34
    assert "E045" in hits[0].text


def test_answer_with_sources(kb):
    llm = FakeListChatModel(responses=["De acordo com o Manual do Compressor RC-75, o óleo recomendado é o "
                                       "ISO VG 68 [Fonte 1]."])
    bot = IndustrialAssistant(kb, llm=llm, min_relevance=0.05)
    ans = bot.ask("Qual óleo deve ser utilizado no compressor?")
    assert ans.found
    assert "[Fonte" not in ans.text
    assert ans.sources and ans.sources[0].file == ans.chunks[0].source_file
    assert "📄 Fonte:" in ans.full_text()


def test_unknown_error_code_is_blocked_before_llm(kb):
    bot = IndustrialAssistant(kb, llm=ExplodingLLM(responses=["x"]))
    ans = bot.ask("A CNC X200 apresentou o erro E999, o que faço?")
    assert not ans.found
    assert ans.text == config.NOT_FOUND_MESSAGE
    assert ans.sources == []


def test_low_relevance_is_blocked_before_llm(kb):
    bot = IndustrialAssistant(kb, llm=ExplodingLLM(responses=["x"]), min_relevance=0.99)
    ans = bot.ask("Qual a receita de bolo de cenoura?")
    assert ans.text == config.NOT_FOUND_MESSAGE


def test_llm_not_found_is_normalized_to_exact_string(kb):
    llm = FakeListChatModel(responses=["Infelizmente não encontrei essa informação nos documentos."])
    bot = IndustrialAssistant(kb, llm=llm, min_relevance=0.0)
    ans = bot.ask("Qual a cor do compressor?")
    assert ans.text == config.NOT_FOUND_MESSAGE and not ans.sources


def test_history_rewrites_follow_up_question(kb):
    llm = FakeListChatModel(responses=[
        "Qual a capacidade de óleo do compressor RC-75?",                      # reescrita
        "De acordo com o Manual do Compressor RC-75, são 22 litros [Fonte 1].",  # resposta
    ])
    bot = IndustrialAssistant(kb, llm=llm, min_relevance=0.05)
    history = [{"role": "user", "content": "Qual óleo deve ser utilizado no compressor?"},
               {"role": "assistant", "content": "O manual recomenda óleo ISO VG 68."}]
    ans = bot.ask("E qual a quantidade?", history=history)
    assert ans.standalone_question == "Qual a capacidade de óleo do compressor RC-75?"
    assert ans.found


def test_helpers():
    assert is_not_found(config.NOT_FOUND_MESSAGE)
    assert not is_not_found("De acordo com o manual, o óleo é ISO VG 68.")
    assert extract_citations("a [Fonte 2] b [Fontes 1, 3]") == [2, 1, 3]
    assert strip_citations("Use ISO VG 68 [Fonte 1].") == "Use ISO VG 68."
    assert strip_citations("Use ISO VG 68 [Fonte 1].\n\nFonte: Manual do Compressor, Página 12.") == "Use ISO VG 68."
    assert strip_citations("Fonte de ar seco a 6 bar.") == "Fonte de ar seco a 6 bar."
    assert strip_citations("Conforme a página 34, [Fonte 1], faça:\n\n1. Desligar.\n\n"
                           "[Fonte 1] Documento: Manual CNC | Arquivo: 01_manual_cnc.pdf | Página 34") \
        == "Conforme a página 34, faça:\n\n1. Desligar."


def test_search_returns_whole_page(kb):
    # A pág. 34 do manual CNC tem 2 chunks; o passo 3 do E045 fica só no segundo.
    hits = kb.search("A máquina CNC modelo X200 apresentou erro E045. Qual procedimento devo realizar?")
    page34 = [c for c in hits if c.source_file == "01_manual_cnc.pdf" and c.page == 34]
    assert len(page34) == 1
    assert "Desligar o equipamento" in page34[0].text and "seção 8.1" in page34[0].text
    assert "SP-X200-07" in page34[0].text
