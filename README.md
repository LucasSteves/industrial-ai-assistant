# 🏭 Industrial AI Assistant — Rocket Industrial Solutions

Assistente de suporte técnico industrial com **RAG (Retrieval-Augmented Generation)**. Responde dúvidas de
operadores e técnicos usando **apenas** a documentação interna da pasta `/knowledge`, sempre indicando
**documento e página** de origem. Se a informação não estiver na base, responde exatamente:

> Não encontrei essa informação na documentação disponível.

Exercício — Unimax Indaiatuba · Prof. Rogerio Morandi.

## Stack

| Etapa | Tecnologia |
|---|---|
| PDF Loader | LangChain `PyPDFLoader` (pypdf) |
| Chunking | LangChain `RecursiveCharacterTextSplitter` (800 caracteres, sobreposição 150, por página) |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (local, via `langchain-huggingface`) |
| Vector DB | ChromaDB persistente (`dados/chroma`), distância de cosseno |
| LLM | Llama 3.2 local via Ollama (`langchain-ollama`), temperatura 0 |
| Interface | Streamlit com chat, histórico e upload de PDFs |

Tudo roda localmente, sem chave de API.

## Como executar (Windows 10/11)

São só dois arquivos, abertos com **dois cliques**:

| Arquivo | Quando | O que faz |
|---|---|---|
| `01_INSTALAR.bat` | **Uma vez** | Baixa e instala tudo o que faltar (veja abaixo) |
| `02_INICIAR.bat` | **Sempre que for usar** | Liga o Ollama e abre o assistente no navegador (`http://localhost:8501`) |

O `01_INSTALAR.bat` não exige nada instalado antes. Ele:

1. Procura o **Python 3.10 a 3.12**. Se não achar, instala o Python 3.12 pelo `winget` (sem precisar de administrador).
2. Cria o ambiente `.venv` e instala as bibliotecas do `requirements.txt`.
3. Procura o **Ollama**. Se não achar, roda o instalador oficial de https://ollama.com. Depois baixa o modelo `llama3.2`.
4. Baixa o modelo de embeddings e indexa a pasta `knowledge`.

Na primeira vez são baixados cerca de **3 GB** (Ollama + `llama3.2` ≈ 2 GB, bibliotecas, embeddings ≈ 470 MB),
então use uma boa conexão. Rodar o instalador de novo é seguro: ele só baixa o que estiver faltando.

Deixe a janela do `02_INICIAR.bat` aberta enquanto usa o sistema; fechá-la encerra o assistente.

Manual (qualquer sistema, com Python 3.10–3.12 e Ollama já instalados):

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2
python scripts/indexar.py
streamlit run app.py
```

Testes (com o `.venv` ativo):

```bash
python -m pytest -q                 # testes automáticos, offline (sem Ollama e sem internet)
python scripts/testar_cenarios.py   # exemplos do enunciado no sistema real (precisa do Ollama)
```

## Como os níveis do enunciado foram atendidos

### Nível 1 — RAG básico
`rag/loader.py` (leitura e chunking) → `rag/vector_store.py` (embeddings e ChromaDB) → `rag/rag_chain.py`
(busca e resposta gerativa com LangChain/LCEL).

### Nível 2 — Sistema profissional
- Upload de novos PDFs pela barra lateral: o arquivo é salvo em `/knowledge` e indexado na hora, sem recriar a base.
- Reenviar um arquivo com o mesmo nome substitui os trechos antigos (IDs determinísticos, sem duplicação).
- Cada resposta mostra as fontes no formato `📄 Fonte: 03_manual_compressor.pdf — Página 12`.
- Painel "Trechos recuperados" mostra cada chunk usado, a página e a relevância.

### Nível 3 — Controle de alucinação
A resposta passa por cinco barreiras e, em qualquer falha, devolve a **string exata** do enunciado, sem fontes:

1. Base vazia.
2. **Código técnico inexistente**: se a pergunta cita um código (ex.: `E999`, `X300`) que não existe nos trechos recuperados, a resposta é bloqueada antes de chamar o LLM.
3. **Relevância mínima**: nenhum trecho acima do limite de similaridade (padrão 0,35, ajustável na tela).
4. **Prompt restritivo**: o LLM só pode usar o contexto e deve devolver a frase padrão quando não encontrar.
5. **Pós-verificação**: se o LLM indicar que não achou, a resposta é normalizada para a frase exata.

### Histórico de conversa
Perguntas de acompanhamento ("E qual a quantidade?") são reescritas pelo LLM como perguntas completas
("Qual a quantidade de óleo do compressor?") antes da busca. Se a reescrita sair inválida, o sistema usa a
pergunta anterior + a atual.

### Busca híbrida (extra)
Modelos de embedding pequenos confundem códigos parecidos (E044 × E045). Por isso, trechos que contêm
literalmente o código citado na pergunta ganham bônus de relevância.

### Página completa no contexto (extra)
Quando uma página longa é dividida em vários trechos, um procedimento pode ficar partido ao meio. Depois da
busca, cada trecho encontrado é unido aos demais trechos da mesma página, então o LLM recebe o procedimento
inteiro (ex.: os 3 passos do erro E045 e a troca do sensor, na pág. 34).

## Base documental (`/knowledge`)

| Arquivo | Conteúdo | Páginas |
|---|---|---|
| `01_manual_cnc.pdf` | Manutenção, operação e códigos de erro da CNC X200 (E045 na pág. 34) | 36 |
| `02_procedimento_seguranca.pdf` | PS-02: NR-12, EPIs, acesso à área (pág. 8), LOTO | 10 |
| `03_manual_compressor.pdf` | Compressor RC-75: especificações, óleo ISO VG 68 (pág. 12), insumos | 16 |
| `04_politica_manutencao.pdf` | PM-04: criticidade, paradas programadas, OS | 6 |
| `05_checklist_operacional.pdf` | CL-05: verificações diárias de turno | 4 |

Os PDFs são **fictícios**, criados para este exercício com as páginas batendo com os exemplos do enunciado. Para usar documentos reais, troque os arquivos e clique em **Reindexar pasta /knowledge**.

## Perguntas para demonstração

| Pergunta | Resultado esperado |
|---|---|
| A máquina CNC modelo X200 apresentou erro E045. Qual procedimento devo realizar? | Desligar, verificar conexão do sensor, teste de calibração; se persistir, trocar o sensor e abrir OS — `01_manual_cnc.pdf`, pág. 34 |
| Qual óleo deve ser utilizado no compressor? | ISO VG 68 — `03_manual_compressor.pdf`, pág. 12 |
| E qual a quantidade? *(logo depois da anterior)* | 22 litros (usa o histórico) |
| Qual procedimento de segurança antes de entrar na área? | Integração NR-12, EPIs... — `02_procedimento_seguranca.pdf`, pág. 8 |
| O que significa o erro E999? | Não encontrei essa informação na documentação disponível. |
| Quem ganhou a Copa do Mundo de 2002? | Não encontrei essa informação na documentação disponível. |
| Qual o torque de aperto dos parafusos do compressor? | Não encontrei essa informação na documentação disponível. |

## Estrutura

```
industrial-ai-assistant/
├── app.py                     # Interface Streamlit (chat, upload, fontes)
├── rag/
│   ├── config.py              # Parâmetros (modelo, chunking, limites, frase NOT_FOUND)
│   ├── loader.py              # PDF Loader + chunking com página
│   ├── vector_store.py        # Embeddings, ChromaDB, busca híbrida
│   ├── rag_chain.py           # Histórico, prompts, anti-alucinação, fontes
│   └── ollama_utils.py        # Status e modelos do Ollama
├── knowledge/                 # Base documental (5 PDFs)
├── scripts/
│   ├── indexar.py             # Indexa /knowledge pelo terminal
│   └── testar_cenarios.py     # Roda os exemplos do enunciado no sistema real
├── tests/test_pipeline.py     # Testes offline (sem Ollama e sem internet)
├── dados/                     # Banco vetorial (criado automaticamente)
├── 01_INSTALAR.bat            # Instala tudo (uma vez)
├── 02_INICIAR.bat             # Abre o assistente
└── requirements.txt
```

## Configuração

Valores em `rag/config.py`, todos sobrescrevíveis por variável de ambiente:

| Variável | Padrão |
|---|---|
| `LLM_MODEL` | `llama3.2` |
| `OLLAMA_URL` | `http://127.0.0.1:11434` |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `150` |
| `TOP_K` | `5` |
| `MIN_RELEVANCE` | `0.35` |

O `llama3.2` (3B) roda em quase qualquer computador. Se a máquina aguentar, `llama3.1:8b` ou `qwen2.5:7b`
seguem melhor as regras de formato e citação (`ollama pull qwen2.5:7b` e selecione na barra lateral).

## Problemas comuns

- **Windows bloqueou o `.bat` ("O Windows protegeu o computador")**: clique em **Mais informações → Executar assim mesmo**.
- **`winget` não encontrado**: instale o Python 3.12 em https://www.python.org/downloads/ (marque "Add Python to PATH") e rode `01_INSTALAR.bat` de novo.
- **"Ollama não responde"**: abra o aplicativo Ollama ou rode `ollama serve`.
- **Modelo não encontrado**: `ollama pull llama3.2`.
- **Primeira execução lenta**: o modelo de embeddings (~470 MB) é baixado uma única vez.
- **Respostas "não encontrei" demais**: reduza a "Relevância mínima" na barra lateral (ex.: 0,25).
- **Base corrompida ou desatualizada**: apague a pasta `dados/` e rode `python scripts/indexar.py`.
