"""Motor RAG: recuperação + geração com histórico e controle de alucinação.

Fluxo de uma pergunta:
  1. Se houver histórico, o LLM reescreve a pergunta como pergunta independente
     ("e qual a quantidade?" -> "Qual a quantidade de óleo do compressor?").
  2. Busca híbrida no ChromaDB (semântica + códigos técnicos).
  3. Controle de alucinação (Nível 3), em camadas:
       a) base vazia                                   -> NOT_FOUND
       b) código citado (ex.: E999) não existe na base -> NOT_FOUND
       c) nenhum trecho acima da similaridade mínima   -> NOT_FOUND
       d) prompt obriga o LLM a responder só com o contexto
          ou devolver exatamente a frase NOT_FOUND
       e) pós-verificação: se o LLM sinalizou que não achou,
          a resposta é trocada pela string EXATA e sem fontes.
  4. Rastreabilidade: o LLM cita [Fonte N]; o sistema converte em
     "📄 Fonte: arquivo.pdf — Página N" usando os metadados reais dos chunks.
"""
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from . import config
from .vector_store import KnowledgeBase, RetrievedChunk, extract_codes, missing_codes

# ----------------------------------------------------------------- prompts
ANSWER_SYSTEM = """Você é o Industrial AI Assistant da Rocket Industrial Solutions, assistente de suporte técnico para operadores e técnicos de manutenção.

REGRAS OBRIGATÓRIAS:
1. Responda EXCLUSIVAMENTE com informações presentes no CONTEXTO abaixo. Não use conhecimento próprio, não deduza e não complete lacunas.
2. Se a resposta não estiver EXPLICITAMENTE no contexto, responda exatamente esta frase e nada mais:
{not_found}
3. Comece citando o documento, por exemplo: "De acordo com o <título do documento>, ...".
4. Procedimentos devem ser apresentados como lista numerada, na mesma ordem do documento.
5. Copie valores técnicos (códigos, pressões, viscosidades, quantidades, intervalos) exatamente como estão no contexto.
6. Ao final de cada informação, indique de qual trecho ela veio usando a marcação [Fonte N]. Não escreva linha de "Fonte:" no final; o sistema já lista as fontes.
7. Seja objetivo e responda em português do Brasil."""

ANSWER_HUMAN = """CONTEXTO:
{context}

PERGUNTA: {question}

RESPOSTA:"""

REWRITE_SYSTEM = """Você reescreve perguntas. Dado o histórico de uma conversa e a última pergunta do usuário, reescreva a última pergunta como uma pergunta completa e independente, substituindo referências como "ele", "essa máquina", "e o óleo?" pelo assunto correto do histórico.
NÃO responda a pergunta. Devolva SOMENTE a pergunta reescrita, em uma linha. Se a pergunta já for independente, devolva-a sem alterações."""

REWRITE_HUMAN = """HISTÓRICO:
{history}

ÚLTIMA PERGUNTA: {question}

PERGUNTA REESCRITA:"""


# ------------------------------------------------------------------ tipos
@dataclass
class Source:
    file: str
    title: str
    page: int

    def label(self) -> str:
        return f"📄 Fonte: {self.file} — Página {self.page}"


@dataclass
class Answer:
    text: str
    found: bool
    sources: List[Source] = field(default_factory=list)
    chunks: List[RetrievedChunk] = field(default_factory=list)
    standalone_question: str = ""
    reason: str = ""  # por que a resposta foi bloqueada (útil para depuração)

    def full_text(self) -> str:
        if not self.sources:
            return self.text
        return self.text + "\n\n" + "\n".join(s.label() for s in self.sources)


# --------------------------------------------------------------- helpers
def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def is_not_found(text: str) -> bool:
    """Detecta se o LLM sinalizou que não encontrou a informação."""
    t = _strip_accents((text or "").strip().lower())
    if not t:
        return True
    return "nao encontrei essa informacao" in t or t.startswith("nao encontrei")


_CITATION_RE = re.compile(r"\[\s*Fontes?\s*([\d\s,e]+)\]", re.IGNORECASE)


def extract_citations(text: str) -> List[int]:
    nums: List[int] = []
    for group in _CITATION_RE.findall(text or ""):
        for n in re.findall(r"\d+", group):
            if int(n) not in nums:
                nums.append(int(n))
    return nums


# Linha de fonte escrita pelo próprio LLM ("Fonte: Manual..., Página 34."): duplicaria a lista do sistema.
_SOURCE_LINE_RE = re.compile(r"^[ \t>*_-]*(📄\s*)?Fontes?\s*(consultadas?)?\s*:.*$\n?", re.IGNORECASE | re.MULTILINE)


# Cabeçalho do bloco de contexto (build_context) que o LLM às vezes copia para a resposta.
_CONTEXT_HEADER_RE = re.compile(r"Documento:[^\n|]*\|\s*Arquivo:[^\n|]*\|\s*P[aá]gina\s*\d+", re.IGNORECASE)


def strip_citations(text: str) -> str:
    text = _SOURCE_LINE_RE.sub("", text or "")
    text = _CONTEXT_HEADER_RE.sub("", text)
    text = _CITATION_RE.sub("", text)
    text = re.sub(r"[ \t]+([.,;:])", r"\1", text)
    text = re.sub(r",{2,}", ",", text)
    text = re.sub(r"\n[ \t]+\n", "\n\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def format_history(history: List[Dict], turns: int = config.HISTORY_TURNS) -> str:
    lines = []
    for msg in history[-turns * 2:]:
        who = "Usuário" if msg.get("role") == "user" else "Assistente"
        content = msg.get("content", "")
        if msg.get("role") == "assistant":
            content = content.split("\n📄")[0][:600]  # resposta sem a lista de fontes
        lines.append(f"{who}: {content}")
    return "\n".join(lines)


def build_context(chunks: List[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[Fonte {i}] Documento: {c.title} | Arquivo: {c.source_file} | Página {c.page}\n{c.text}")
    return "\n\n".join(blocks)


def get_llm(model: str = config.LLM_MODEL) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    return ChatOllama(model=model, base_url=config.OLLAMA_URL, temperature=0, num_ctx=config.LLM_NUM_CTX)


# ---------------------------------------------------------------- motor
class IndustrialAssistant:
    def __init__(self, kb: KnowledgeBase, llm: Optional[BaseChatModel] = None,
                 top_k: int = config.TOP_K, min_relevance: float = config.MIN_RELEVANCE):
        self.kb = kb
        self.llm = llm or get_llm()
        self.top_k = top_k
        self.min_relevance = min_relevance
        parser = StrOutputParser()
        self.answer_chain = ChatPromptTemplate.from_messages(
            [("system", ANSWER_SYSTEM), ("human", ANSWER_HUMAN)]
        ) | self.llm | parser
        self.rewrite_chain = ChatPromptTemplate.from_messages(
            [("system", REWRITE_SYSTEM), ("human", REWRITE_HUMAN)]
        ) | self.llm | parser

    # 1) pergunta independente --------------------------------------------
    def rewrite_question(self, question: str, history: List[Dict]) -> str:
        if not history:
            return question
        last_user = next((m["content"] for m in reversed(history) if m.get("role") == "user"), "")
        fallback = f"{last_user} {question}".strip()  # usado se a reescrita do LLM for inválida
        try:
            out = self.rewrite_chain.invoke({"history": format_history(history), "question": question})
        except Exception:
            return fallback
        out = out.strip().splitlines()[0].strip() if out.strip() else ""
        out = re.sub(r"^(pergunta( reescrita)?\s*:\s*)", "", out, flags=re.IGNORECASE).strip(' "\'*')
        # Proteções: modelo pequeno às vezes RESPONDE em vez de reescrever, ou perde o código citado.
        lost_codes = set(extract_codes(question)) - set(extract_codes(out))
        if (not out or not out.endswith("?") or len(out) > max(300, len(question) * 4)
                or is_not_found(out) or lost_codes):
            return fallback
        return out

    # 2..4) pergunta completa -----------------------------------------------
    def ask(self, question: str, history: Optional[List[Dict]] = None) -> Answer:
        history = history or []
        standalone = self.rewrite_question(question, history)

        def blocked(reason: str, chunks=None) -> Answer:
            return Answer(text=config.NOT_FOUND_MESSAGE, found=False, chunks=chunks or [],
                          standalone_question=standalone, reason=reason)

        chunks = self.kb.search(standalone, k=self.top_k)
        if not chunks:
            return blocked("Base vazia: nenhum documento indexado.")

        # 3b) código técnico inexistente na documentação
        missing = missing_codes(f"{question} {standalone}", chunks)
        if missing:
            return blocked(f"Código(s) não encontrado(s) na documentação: {', '.join(missing)}.", chunks)

        # 3c) similaridade mínima
        relevant = [c for c in chunks if c.score >= self.min_relevance or c.keyword_match]
        if not relevant:
            best = max(c.score for c in chunks)
            return blocked(f"Nenhum trecho atingiu a relevância mínima ({best:.2f} < {self.min_relevance:.2f}).", chunks)

        # 3d) geração restrita ao contexto
        raw = self.answer_chain.invoke({
            "context": build_context(relevant),
            "question": standalone,
            "not_found": config.NOT_FOUND_MESSAGE,
        })

        # 3e) pós-verificação
        if is_not_found(raw):
            return blocked("O modelo indicou que o contexto não contém a resposta.", relevant)

        # 4) rastreabilidade
        cited = [n for n in extract_citations(raw) if 1 <= n <= len(relevant)]
        used = [relevant[n - 1] for n in cited] if cited else relevant[:1]
        sources: List[Source] = []
        for c in used:
            if not any(s.file == c.source_file and s.page == c.page for s in sources):
                sources.append(Source(file=c.source_file, title=c.title, page=c.page))

        return Answer(text=strip_citations(raw), found=True, sources=sources,
                      chunks=relevant, standalone_question=standalone)
