"""Script: build Chroma vector DB from processed dataset"""
from pathlib import Path
import sys

# ensure project root is on sys.path when running the script directly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from dhauz_ticket_classifier.rag.vector_store import VectorStore


def main():
    data_path = Path("data/dataset_processed.csv")
    if not data_path.exists():
        raise FileNotFoundError("Processed dataset not found. Run scripts/download_data.py first")
    df = pd.read_csv(data_path)
    vs = VectorStore()
    print("Building vector store...")
    vs.create_from_dataframe(df)
    print("Vector store created at:", vs.persist_dir)


if __name__ == '__main__':
    main()
