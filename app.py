"""Industrial AI Assistant — Rocket Industrial Solutions (interface Streamlit).

Executar:  streamlit run app.py
"""
import re
from pathlib import Path

import streamlit as st

from rag import config
from rag.loader import list_pdfs
from rag.ollama_utils import list_models, ollama_alive
from rag.rag_chain import IndustrialAssistant, get_llm
from rag.vector_store import KnowledgeBase

st.set_page_config(page_title="Industrial AI Assistant · Rocket", page_icon="🏭", layout="wide")

EXAMPLES = [
    "A máquina CNC modelo X200 apresentou erro E045. Qual procedimento devo realizar?",
    "Qual óleo deve ser utilizado no compressor?",
    "Qual procedimento de segurança antes de entrar na área?",
    "Qual o torque de aperto dos parafusos do compressor?",  # não existe na base -> Nível 3
]


# ------------------------------------------------------------------ estado
@st.cache_resource(show_spinner="Carregando modelo de embeddings e banco vetorial...")
def get_kb() -> KnowledgeBase:
    return KnowledgeBase()


def init_state():
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("uploader_key", 0)
    st.session_state.setdefault("pending", None)


def safe_filename(name: str) -> str:
    stem = re.sub(r"[^\w\-. ]", "_", Path(name).stem).strip() or "documento"
    return f"{stem}.pdf"


# --------------------------------------------------------------- render
def render_answer(msg: dict, show_chunks: bool):
    st.markdown(msg["content"])
    if msg.get("sources"):
        st.markdown("\n".join(f"**{s}**  " for s in msg["sources"]))
    if show_chunks and msg.get("chunks"):
        with st.expander(f"🔎 Trechos recuperados ({len(msg['chunks'])})"):
            if msg.get("standalone") and msg["standalone"] != msg.get("question"):
                st.caption(f"Pergunta interpretada com o histórico: *{msg['standalone']}*")
            if msg.get("reason"):
                st.caption(f"Motivo do bloqueio: {msg['reason']}")
            for i, c in enumerate(msg["chunks"], start=1):
                tag = " · código encontrado" if c["keyword_match"] else ""
                st.markdown(f"**Trecho {i}** — `{c['source_file']}` · página {c['page']} · "
                            f"relevância {c['score']:.2f}{tag}")
                st.caption(c["text"])


# ---------------------------------------------------------------- app
init_state()
kb = get_kb()

# Primeira execução: indexa a pasta /knowledge automaticamente
if kb.count() == 0 and list_pdfs(config.KNOWLEDGE_DIR):
    with st.spinner("Primeira execução: indexando a pasta /knowledge..."):
        kb.index_folder(config.KNOWLEDGE_DIR)

with st.sidebar:
    st.header("⚙️ Configuração")
    alive = ollama_alive()
    if alive:
        st.success("Ollama conectado")
    else:
        st.error(f"Ollama não responde em {config.OLLAMA_URL}")

    models = list_models() if alive else []
    if models:
        default = next((i for i, m in enumerate(models) if m.split(":")[0] == config.LLM_MODEL), 0)
        model = st.selectbox("Modelo LLM", models, index=default)
    else:
        model = st.text_input("Modelo LLM", value=config.LLM_MODEL)

    top_k = st.slider("Trechos recuperados (Top K)", 1, 10, config.TOP_K)
    min_rel = st.slider("Relevância mínima", 0.0, 1.0, config.MIN_RELEVANCE, 0.05,
                        help="Abaixo deste valor o trecho é ignorado. Se nenhum trecho passar, "
                             "a resposta é a frase padrão de 'não encontrei'.")
    show_chunks = st.toggle("Mostrar trechos recuperados", value=True)

    st.divider()
    st.header("📚 Base documental")
    files = kb.indexed_files()
    c1, c2 = st.columns(2)
    c1.metric("Documentos", len(files))
    c2.metric("Chunks", kb.count())
    with st.expander("Documentos indexados"):
        for name, n in files.items():
            st.write(f"📄 {name} — {n} chunks")

    uploads = st.file_uploader("Adicionar novos PDFs", type=["pdf"], accept_multiple_files=True,
                               key=f"uploader_{st.session_state.uploader_key}")
    if uploads and st.button("➕ Adicionar à base", type="primary", use_container_width=True):
        config.KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        added = []
        with st.spinner("Lendo PDFs, gerando embeddings e indexando..."):
            for up in uploads:
                dest = config.KNOWLEDGE_DIR / safe_filename(up.name)
                dest.write_bytes(up.getbuffer())
                try:
                    added.append((dest.name, kb.index_file(dest)))
                except Exception as e:
                    st.error(f"Erro ao indexar {up.name}: {e}")
        st.session_state.uploader_key += 1
        st.session_state.flash = "Indexado: " + ", ".join(f"{n} ({c} chunks)" for n, c in added)
        st.rerun()

    if st.button("🔄 Reindexar pasta /knowledge", use_container_width=True):
        with st.spinner("Recriando a base vetorial..."):
            kb.index_folder(config.KNOWLEDGE_DIR, reset=True)
        st.session_state.flash = "Base recriada a partir da pasta /knowledge."
        st.rerun()

    if st.session_state.get("flash"):
        st.success(st.session_state.pop("flash"))

    st.divider()
    if st.button("🧹 Nova conversa", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------- chat
st.title("🏭 Industrial AI Assistant")
st.caption("Rocket Industrial Solutions · Suporte técnico com RAG — respostas somente a partir da documentação "
           "interna, com documento e página de origem.")

if not st.session_state.messages:
    st.markdown("**Experimente:**")
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.pending = ex
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👷" if msg["role"] == "user" else "🏭"):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_answer(msg, show_chunks)

typed = st.chat_input("Descreva o problema ou faça uma pergunta sobre os equipamentos...")
question = typed or st.session_state.pending
st.session_state.pending = None

if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="👷"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🏭"):
        if not alive:
            msg = {"role": "assistant", "content": "⚠️ O Ollama não está ativo. Abra o Ollama (ou rode "
                   "`ollama serve`) e tente novamente.", "sources": [], "chunks": []}
        else:
            try:
                with st.spinner("Consultando a documentação..."):
                    bot = IndustrialAssistant(kb, llm=get_llm(model), top_k=top_k, min_relevance=min_rel)
                    ans = bot.ask(question, history=history)
                msg = {
                    "role": "assistant", "content": ans.text, "question": question,
                    "sources": [s.label() for s in ans.sources],
                    "chunks": [c.to_dict() for c in ans.chunks],
                    "standalone": ans.standalone_question, "reason": ans.reason,
                }
            except Exception as e:
                msg = {"role": "assistant", "content": f"⚠️ Erro ao consultar o modelo `{model}`: {e}\n\n"
                       f"Verifique se ele foi baixado com `ollama pull {model}`.", "sources": [], "chunks": []}
        render_answer(msg, show_chunks)
    st.session_state.messages.append(msg)
