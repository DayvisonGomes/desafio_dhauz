"""CLI: classify a single ticket (placeholder to wire components)"""
import argparse
from pathlib import Path
import sys

# ensure project root is on sys.path when running the script directly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ticket", type=str, help="Ticket text to classify")
    parser.add_argument("--mode", choices=["rag","hybrid"], default="hybrid")
    parser.add_argument("--checkpoint", type=str, default='./results', help='DistilBERT checkpoint dir')
    parser.add_argument("--chroma-dir", type=str, default='data/chroma_db', help='Chroma DB dir')
    parser.add_argument("--use-llm-remote", action='store_true', help='Use remote LLM for RAG steps')
    parser.add_argument("--remote-llm-url", type=str, default=None, help='Remote LLM URL (if using remote)')
    args = parser.parse_args()

    # Load minimal components and run classification (best-effort)
    print(f"Classifying ticket (mode={args.mode})...")
    try:
        from dhauz_ticket_classifier.models.distilbert_classifier import DistilBERTClassifier
        from dhauz_ticket_classifier.rag.vector_store import VectorStore
        from dhauz_ticket_classifier.rag.classifier import RAGClassifier, HybridClassifier

        distil = DistilBERTClassifier()
        distil.load(args.checkpoint)
        vs = VectorStore(persist_dir=args.chroma_dir)
        if vs.exists():
            vs.load()
        else:
            print('Chroma DB not found at', args.chroma_dir)

        from dhauz_ticket_classifier.utils.helpers import maybe_load_llm, infer_classes

        llm = maybe_load_llm(use_llm=False, use_remote=args.use_llm_remote, remote_url=args.remote_llm_url)

        # Infer classes from processed dataset if available, otherwise fallback
        classes = None
        data_csv = Path('data/dataset_processed.csv')
        if data_csv.exists():
            try:
                import pandas as pd
                df = pd.read_csv(data_csv)
                if 'class' in df.columns:
                    classes = sorted(df['class'].unique())
            except Exception:
                classes = None

        classes = infer_classes(data_csv='data/dataset_processed.csv', vector_store=vs)
        rag = RAGClassifier(distil, vs, llm, classes=classes)
        hybrid = HybridClassifier(rag)
        if args.mode == 'rag':
            out = rag.classify_batch(args.ticket)
        else:
            out = hybrid.classify_batch(args.ticket)
        print('Result:', out)
    except Exception as e:
        print('Failed to run classification:', e)

if __name__ == '__main__':
    main()
