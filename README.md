# IT Service Ticket Classifier — GenAI + RAG + Agents

Resumo do repositório e instruções para reprodução.

Estrutura principal:

- `originals/`: cópia exata do notebook/script original (não modificar).
- `dhauz_ticket_classifier/`: pacote modular com submódulos `data`, `models`, `rag`, `agents`, `interfaces`, `utils`.
- `scripts/`: utilitários para download, construção de vector DB e demo.
- `requirements.txt`, `config.yaml`, `.gitignore`.

Quick start (local):

1. Instale dependências:

```bash
pip install -r requirements.txt
```

2. Configure Kaggle se quiser baixar dataset diretamente (opcional): coloque `kaggle.json` em `~/.kaggle/`.

3. Baixe e processe os dados (ou coloque CSV manualmente em `data/dataset_processed.csv`):

```bash
python scripts/download_data.py
```

4. Construa o banco vetorial Chroma:

```bash
python scripts/build_vector_db.py
```

5. Para demonstração local com Gradio, ajuste `scripts/demo.py` para carregar modelos e executar `python scripts/demo.py`.

Observações:

- O código original está preservado em `originals/` sem mudanças.
- Modularizamos o pipeline em classes: `RAGClassifier` e `HybridClassifier` (padrão configurável no `config.yaml`).
- O agent (Seção 13 do notebook) foi mantido como `TicketAgent` opcional; por padrão o fluxo usa as classes diretamente.
- O `LLM_BATCH_SIZE` padrão é 4 (configurável) — se não houver GPU suficiente use 1.
- Para reproduzir exatamente, use as seeds definidas em `dhauz_ticket_classifier/config.py` ou `config.yaml`.

Colab / Execução em máquina com GPU
----------------------------------

Recomenda-se usar Colab (ou outra VM com GPU) para as etapas pesadas (treino DistilBERT, geração com Qwen). Exemplo de passos:

1. Suba o repositório para o Colab (GitHub ou upload ZIP).
2. Instale dependências:

```bash
pip install -r requirements.txt
```

3. Configure `~/.kaggle/kaggle.json` no ambiente do Colab (upload).

4. Execute (em ordem):

```bash
python scripts/download_data.py       # baixa e pré-processa
python scripts/train_distilbert.py    # treina DistilBERT e salva checkpoint em ./results
python scripts/build_vector_db.py     # cria Chroma DB (embeddings)
python scripts/demo.py               # ajustar para carregar modelos e iniciar Gradio
```

Dicas e limites
- Se usar Qwen2.5-3B local, prefira VM com >=16GB VRAM ou usar instância com `device_map='auto'` e `torch.float16`.
- Para evitar erros OOM, mantenha `llm.batch_size: 1` em `config.yaml` e aumente gradualmente.

Export / Import do Chroma DB
---------------------------------

Para mover o banco vetorial entre máquinas (por exemplo entre sua máquina local e Colab), você pode empacotar o diretório do Chroma DB em um arquivo ZIP usando o demo:

```bash
# No host de origem (onde chroma_db existe)
python scripts/demo.py --chroma-dir data/chroma_db --export-chroma data/chroma_db.zip

# No host destino (Colab): faça upload do data/chroma_db.zip e então
python scripts/demo.py --import-chroma data/chroma_db.zip --import-overwrite
```

Observação: o `--import-overwrite` remove o diretório existente antes de extrair o ZIP.

Comandos práticos (Colab) — incluindo LLM
----------------------------------------

Exemplo de sequência no Colab (assumindo GPU disponível):

```bash
# Instale dependências
pip install -r requirements.txt

# Baixe e preprocese dados
python scripts/download_data.py

# Treine DistilBERT (gera ./results)
python scripts/train_distilbert.py --num-epochs 2

# Construa Chroma DB
python scripts/build_vector_db.py

# (Opcional) Exporte Chroma para zip para transferir
python scripts/demo.py --chroma-dir data/chroma_db --export-chroma data/chroma_db.zip

# Execute demo carregando LLM (padrão LLM setado em config)
python scripts/demo.py --use-llm

# Se preferir especificar outro modelo HF
python scripts/demo.py --use-llm --llm-model "Qwen/Qwen2.5-3B-Instruct"
```

AVISO: carregar um LLM grande localmente requer bastante VRAM. Se ocorrer OOM, reduza `--llm-batch` (ou deixe `--use-llm` desativado) e use o modo `hybrid` ou execute a geração em instância maior.

Avaliação (classification_report e matriz de confusão)
-----------------------------------------------

Comandos para avaliar modelos com 200 amostras reproduzíveis (salva relatório JSON, heatmap PNG e CSV de previsões):

```bash
# Avaliar DistilBERT (val por padrão, salva em ./results)
python scripts/evaluate.py --processed data/dataset_processed.csv --checkpoint ./results --out-dir ./results

# Avaliar RAG/Hybrid (val por padrão). Modo: rag ou hybrid
python scripts/evaluate_rag.py --processed data/dataset_processed.csv --chroma-dir data/chroma_db --checkpoint ./results --out-dir ./results --mode hybrid

# Usar amostra da base de treino em vez da validação
python scripts/evaluate.py --use-train
python scripts/evaluate_rag.py --use-train

# Avaliar RAG com LLM remoto (HuggingFace Inference API)
python scripts/evaluate_rag.py --use-llm-remote --remote-llm-url "https://api-inference.huggingface.co/models/SEU-MODELO" --remote-llm-key $HF_KEY
```

Arquivos gerados (`--out-dir`, padrão `./results`):
- `classification_report.json` / `classification_report_rag.json` — relatório de métricas em JSON
- `class_index_map.json` / `class_index_map_rag.json` — mapeamento índice→classe
- `confusion_matrix_heatmap.png` / `confusion_matrix_rag_heatmap.png` — heatmap da matriz de confusão
- `evaluation_predictions.csv` / `evaluation_rag_predictions.csv` — CSV com textos e previsões
Arquitetura e decisões
----------------------

- Mantivemos o código original intacto em `originals/` (não modificado).
- Implementamos classes `RAGClassifier` e `HybridClassifier` que encapsulam a lógica do notebook.
- O Agent (Seção 13) está disponível como `TicketAgent` e pode ser integrado ao fluxo principal se desejar (o padrão usa as classes diretamente).
- Batching: DistilBERT e recuperação são batched; LLM batch é configurável (padrão 4).
- Seeds: definidas em `dhauz_ticket_classifier/config.py` e `config.yaml` para reprodutibilidade.
