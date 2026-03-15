"""Script: build Chroma vector DB from processed dataset"""
from pathlib import Path
import sys

# ensure project root is on sys.path when running the script directly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from dhauz_ticket_classifier.rag.vector_store import VectorStore


def main(processed: str = "data/dataset_processed.csv", chroma_dir: str = "data/chroma_db"):
    data_path = Path(processed)
    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found. Run scripts/download_data.py first (expected at {data_path})")
    df = pd.read_csv(data_path)
    vs = VectorStore(persist_dir=chroma_dir)
    print("Building vector store from:", data_path)
    vs.create_from_dataframe(df)
    print("Vector store created at:", vs.persist_dir)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build Chroma vector DB from processed CSV')
    parser.add_argument('--processed', type=str, default='data/dataset_processed.csv', help='Path to processed CSV')
    parser.add_argument('--chroma-dir', type=str, default='data/chroma_db', help='Chroma DB persist directory')
    args = parser.parse_args()

    main(processed=args.processed, chroma_dir=args.chroma_dir)
