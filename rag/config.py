"""Configuração central do Industrial AI Assistant.

Qualquer valor pode ser sobrescrito por variável de ambiente
(ex.: set LLM_MODEL=llama3.1 no Windows antes de iniciar).
"""
import os
from pathlib import Path

# Silencia telemetria do Chroma e avisos do tokenizer (precisa vir antes dos imports deles)
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent

# --- Pastas ---------------------------------------------------------------
KNOWLEDGE_DIR = Path(os.getenv("KNOWLEDGE_DIR", ROOT / "knowledge"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", ROOT / "dados" / "chroma"))
COLLECTION_NAME = "rocket_industrial_kb"

# --- LLM (Ollama local) ---------------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "4096"))

# --- Embeddings (multilíngue, roda local na CPU) --------------------------
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# --- Chunking e recuperação -----------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "5"))
# Similaridade mínima (cosseno, 0 a 1) para um trecho ser considerado relevante.
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.35"))
# Bônus de score para trechos que contêm literalmente um código técnico da pergunta (E045, X200, NR-12...)
KEYWORD_BOOST = 0.20

# --- Conversa -------------------------------------------------------------
HISTORY_TURNS = 4  # pares pergunta/resposta usados para entender perguntas de acompanhamento

# --- Nível 3: string EXATA exigida pelo enunciado --------------------------
NOT_FOUND_MESSAGE = "Não encontrei essa informação na documentação disponível."
