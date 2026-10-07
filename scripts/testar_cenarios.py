"""Roda os cenários do enunciado contra o sistema real (precisa do Ollama ativo).

Uso:  python scripts/testar_cenarios.py [modelo]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import config  # noqa: E402
from rag.rag_chain import IndustrialAssistant, get_llm  # noqa: E402
from rag.vector_store import KnowledgeBase  # noqa: E402

CENARIOS = [
    ("Exemplo 1 - Operador (erro CNC)",
     "A máquina CNC modelo X200 apresentou erro E045. Qual procedimento devo realizar?"),
    ("Exemplo 2 - Técnico de manutenção", "Qual óleo deve ser utilizado no compressor?"),
    ("Histórico - pergunta de acompanhamento", "E qual a quantidade?"),
    ("Exemplo 3 - Funcionário novo", "Qual procedimento de segurança antes de entrar na área?"),
    ("Nível 3 - código inexistente", "O que significa o erro E999 na CNC X200?"),
    ("Nível 3 - fora da base", "Qual o torque de aperto dos parafusos do compressor?"),
    ("Nível 3 - assunto alheio", "Quem ganhou a Copa do Mundo de 2002?"),
]


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else config.LLM_MODEL
    kb = KnowledgeBase()
    if kb.count() == 0:
        print("Base vazia, indexando /knowledge...")
        kb.index_folder(config.KNOWLEDGE_DIR)
    bot = IndustrialAssistant(kb, llm=get_llm(model))

    history = []
    for titulo, pergunta in CENARIOS:
        ans = bot.ask(pergunta, history=history)
        print("=" * 80)
        print(f"{titulo}\nPERGUNTA: {pergunta}")
        if ans.standalone_question != pergunta:
            print(f"(interpretada como: {ans.standalone_question})")
        print("-" * 80)
        print(ans.full_text())
        if ans.reason:
            print(f"[bloqueio: {ans.reason}]")
        history += [{"role": "user", "content": pergunta}, {"role": "assistant", "content": ans.text}]
    print("=" * 80)


if __name__ == "__main__":
    main()
