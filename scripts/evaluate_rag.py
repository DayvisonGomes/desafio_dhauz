"""Avaliação do classificador RAG/Hybrid: seleciona N amostras e gera classification_report + heatmap.

Usage:
    python scripts/evaluate_rag.py --processed data/dataset_processed.csv --chroma-dir data/chroma_db --checkpoint ./results --out-dir ./results --mode hybrid --sample-size 200
"""
import argparse
from pathlib import Path
import sys

# ensure project root is on sys.path when running the script directly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from dhauz_ticket_classifier.rag.vector_store import VectorStore
from dhauz_ticket_classifier.models.distilbert_classifier import DistilBERTClassifier
from dhauz_ticket_classifier.models.llm import LLMPipeline
from dhauz_ticket_classifier.rag.classifier import RAGClassifier, HybridClassifier
from dhauz_ticket_classifier.config import TRAIN_TEST_SPLIT_SEED


def load_vector_store(chroma_dir: str, import_zip: str = None, import_overwrite: bool = False):
    vs = VectorStore(persist_dir=chroma_dir)
    if import_zip:
        vs.import_from_zip(import_zip, overwrite=import_overwrite)
    if not vs.exists():
        raise FileNotFoundError(f"Chroma DB not found at {chroma_dir}. Build it with scripts/build_vector_db.py")
    vs.load()
    return vs


def maybe_load_llm(use_llm: bool, use_remote: bool, remote_url: str = None, remote_key: str = None, provider: str = 'hf', model_name: str = None):
    if use_remote:
        class RemoteLLM:
            def __init__(self, url: str, api_key: str = None, provider: str = 'hf'):
                try:
                    import requests
                except Exception:
                    requests = None
                self._requests = requests
                self.url = url
                self.api_key = api_key

            def generate(self, prompts, batch_size=1):
                if self._requests is None:
                    raise RuntimeError('requests not available; install requests')
                headers = {}
                if self.api_key:
                    headers['Authorization'] = f"Bearer {self.api_key}"
                results = []
                for prompt in prompts:
                    payload = {"inputs": prompt}
                    resp = self._requests.post(self.url, headers=headers, json=payload, timeout=60)
                    resp.raise_for_status()
                    data = resp.json()
                    if isinstance(data, list) and len(data) > 0 and 'generated_text' in data[0]:
                        results.append(data)
                    elif isinstance(data, dict) and 'generated_text' in data:
                        results.append({'generated_text': data['generated_text']})
                    else:
                        results.append({'generated_text': str(data)})
                return results

        if not remote_url:
            raise ValueError('Remote LLM requested but no --remote-llm-url provided')
        return RemoteLLM(remote_url, api_key=remote_key, provider=provider)
    elif use_llm:
        if model_name is None:
            raise ValueError('Local LLM requested but no model name provided')
        return LLMPipeline(model_name=model_name)
    else:
        return None


def evaluate(processed_csv: str, chroma_dir: str, checkpoint: str, out_dir: str, mode: str = 'hybrid', sample_size: int = 200, use_train: bool = False, seed: int = TRAIN_TEST_SPLIT_SEED, use_llm: bool = False, use_remote_llm: bool = False, remote_url: str = None, remote_key: str = None, llm_model: str = None):
    p = Path(processed_csv)
    if not p.exists():
        raise FileNotFoundError(f"Processed dataset not found: {processed_csv}")

    df = pd.read_csv(p)
    if 'text' not in df.columns or 'class' not in df.columns:
        raise ValueError("CSV must contain 'text' and 'class' columns")

    train, val = train_test_split(df, test_size=0.2, stratify=df['class'], random_state=seed)
    classes = sorted(train['class'].unique())

    pool = train if use_train else val
    if len(pool) < sample_size:
        raise ValueError(f"Pool has only {len(pool)} rows, cannot sample {sample_size}")
    sample = pool.sample(sample_size, random_state=seed).reset_index(drop=True)

    # Load resources
    vs = load_vector_store(chroma_dir)

    distil = DistilBERTClassifier()
    distil.load(checkpoint)

    llm = maybe_load_llm(use_llm=use_llm, use_remote=use_remote_llm, remote_url=remote_url, remote_key=remote_key, model_name=llm_model)

    rag = RAGClassifier(distil, vs, llm, classes=classes)
    hybrid = HybridClassifier(rag)

    texts = sample['text'].astype(str).tolist()

    if mode == 'rag':
        outputs = rag.classify_batch(texts, batch_size=None)
    else:
        outputs = hybrid.classify_batch(texts, batch_size=None)

    pred_labels = [o.get('class', 'Miscellaneous') if o is not None else 'Miscellaneous' for o in outputs]
    true_labels = sample['class'].tolist()

    # Report and save as JSON
    report_dict = classification_report(true_labels, pred_labels, digits=4, output_dict=True)
    print(classification_report(true_labels, pred_labels, digits=4))

    os.makedirs(out_dir, exist_ok=True)
    report_path = Path(out_dir) / 'classification_report_rag.json'
    with open(report_path, 'w', encoding='utf8') as fh:
        json.dump(report_dict, fh, ensure_ascii=False, indent=2)

    class_map = {str(i): c for i, c in enumerate(classes)}
    mapping_path = Path(out_dir) / 'class_index_map_rag.json'
    with open(mapping_path, 'w', encoding='utf8') as fh:
        json.dump(class_map, fh, ensure_ascii=False, indent=2)

    cm = confusion_matrix(true_labels, pred_labels, labels=classes)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix RAG/Hybrid ({sample_size} samples)')
    heatmap_path = Path(out_dir) / 'confusion_matrix_rag_heatmap.png'
    fig.savefig(heatmap_path, bbox_inches='tight')
    plt.close(fig)

    out_csv = Path(out_dir) / 'evaluation_rag_predictions.csv'
    sample_out = sample.copy()
    sample_out['predicted'] = pred_labels
    sample_out.to_csv(out_csv, index=False)

    print('Saved heatmap:', heatmap_path)
    print('Saved predictions:', out_csv)
    print('Saved classification report:', report_path)
    print('Saved class-index mapping:', mapping_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed', type=str, default='data/dataset_processed.csv')
    parser.add_argument('--chroma-dir', type=str, default='data/chroma_db')
    parser.add_argument('--checkpoint', type=str, default='./results')
    parser.add_argument('--out-dir', type=str, default='./results')
    parser.add_argument('--mode', choices=['rag', 'hybrid'], default='hybrid')
    parser.add_argument('--sample-size', type=int, default=200)
    parser.add_argument('--use-train', action='store_true')
    parser.add_argument('--seed', type=int, default=TRAIN_TEST_SPLIT_SEED)
    parser.add_argument('--use-llm', action='store_true')
    parser.add_argument('--use-llm-remote', action='store_true')
    parser.add_argument('--remote-llm-url', type=str, default=None)
    parser.add_argument('--remote-llm-key', type=str, default=None)
    parser.add_argument('--llm-model', type=str, default=None)
    args = parser.parse_args()

    evaluate(args.processed, args.chroma_dir, args.checkpoint, args.out_dir, mode=args.mode, sample_size=args.sample_size, use_train=args.use_train, seed=args.seed, use_llm=args.use_llm, use_remote_llm=args.use_llm_remote, remote_url=args.remote_llm_url, remote_key=args.remote_llm_key, llm_model=args.llm_model)


if __name__ == '__main__':
    main()
