"""Inferência em lote para RAG/Hybrid.

Usage:
  python scripts/rag_infer_batch.py --input data/input.csv --mode hybrid --checkpoint ./results --chroma-dir data/chroma_db --out results/rag_preds.jsonl
"""
from pathlib import Path
import sys
import argparse
import json

# ensure project root is on sys.path when running the script directly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dhauz_ticket_classifier.rag.vector_store import VectorStore
from dhauz_ticket_classifier.models.distilbert_classifier import DistilBERTClassifier
from dhauz_ticket_classifier.rag.classifier import RAGClassifier, HybridClassifier
from dhauz_ticket_classifier.utils.helpers import maybe_load_llm, infer_classes


def read_inputs(input_path: str):
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(input_path)
    if p.suffix.lower() == '.txt':
        with p.open('r', encoding='utf8') as f:
            return [l.strip() for l in f if l.strip()]
    else:
        import pandas as pd
        df = pd.read_csv(p)
        if 'text' not in df.columns:
            raise ValueError("CSV input must contain a 'text' column")
        return df['text'].astype(str).tolist()


def main():
    parser = argparse.ArgumentParser(description='Batch inference for RAG/Hybrid')
    parser.add_argument('--input', type=str, required=True, help='Input TXT (one per line) or CSV with text column')
    parser.add_argument('--checkpoint', type=str, default='./results', help='DistilBERT checkpoint dir')
    parser.add_argument('--chroma-dir', type=str, default='data/chroma_db', help='Chroma DB directory')
    parser.add_argument('--mode', choices=['rag', 'hybrid'], default='hybrid')
    parser.add_argument('--out', type=str, default='results/rag_predictions.jsonl', help='Output JSONL file')
    parser.add_argument('--process-batch', type=int, default=32, help='Number of examples per processing batch')
    parser.add_argument('--llm-batch', type=int, default=4, help='LLM generation batch size')
    parser.add_argument('--use-llm', action='store_true', help='Load local LLM (heavy)')
    parser.add_argument('--llm-model', type=str, default=None, help='Local LLM model name (if --use-llm)')
    parser.add_argument('--use-llm-remote', action='store_true', help='Use remote LLM endpoint instead of local model')
    parser.add_argument('--remote-llm-url', type=str, default=None, help='Remote LLM endpoint URL')
    parser.add_argument('--remote-llm-key', type=str, default=None, help='API key for remote LLM (optional)')
    args = parser.parse_args()

    texts = read_inputs(args.input)
    print(f'Loaded {len(texts)} inputs')

    # Load VectorStore
    vs = VectorStore(persist_dir=args.chroma_dir)
    if not vs.exists():
        raise FileNotFoundError(f'Chroma DB not found at {args.chroma_dir}. Build it with scripts/build_vector_db.py')
    vs.load()

    # Load DistilBERT
    distil = DistilBERTClassifier()
    distil.load(args.checkpoint)

    # Load or configure LLM and classes
    llm = maybe_load_llm(use_llm=args.use_llm, use_remote=args.use_llm_remote, remote_url=args.remote_llm_url, remote_key=args.remote_llm_key, model_name=args.llm_model)
    classes = infer_classes(data_csv='data/dataset_processed.csv', vector_store=vs)

    rag = RAGClassifier(distil, vs, llm, classes=classes)
    hybrid = HybridClassifier(rag)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open('w', encoding='utf8') as fo:
        for i in range(0, len(texts), args.process_batch):
            batch = texts[i:i+args.process_batch]
            if args.mode == 'rag':
                results = rag.classify_batch(batch, batch_size=args.llm_batch)
            else:
                results = hybrid.classify_batch(batch, batch_size=args.llm_batch)

            for text, res in zip(batch, results):
                # res expected to be dict {'class':..., 'justification':...}
                out = {
                    'text': text,
                    'predicted': res.get('class') if isinstance(res, dict) else None,
                    'justification': res.get('justification') if isinstance(res, dict) else str(res),
                    'raw': res
                }
                fo.write(json.dumps(out, ensure_ascii=False) + '\n')

    print('Wrote', out_path)


if __name__ == '__main__':
    main()
