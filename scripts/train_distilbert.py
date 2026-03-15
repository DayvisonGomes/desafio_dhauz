"""Train DistilBERT on the processed dataset.

This script expects `data/dataset_processed.csv` to exist (run `scripts/download_data.py`).
"""
from pathlib import Path
import sys

# ensure project root is on sys.path when running the script directly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split
from dhauz_ticket_classifier.models.distilbert_classifier import DistilBERTClassifier
from dhauz_ticket_classifier.config import RESULTS_DIR, TRAIN_TEST_SPLIT_SEED
import os


def main(processed_csv: str = "data/dataset_processed.csv", output_dir: str = RESULTS_DIR, num_epochs: int = 2, batch_size: int = 32):
    p = Path(processed_csv)
    if not p.exists():
        raise FileNotFoundError(f"Processed dataset not found at {processed_csv}")

    df = pd.read_csv(p)
    # Ensure columns
    if "text" not in df.columns or "class" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'class' columns")

    train, val = train_test_split(df, test_size=0.2, stratify=df["class"], random_state=TRAIN_TEST_SPLIT_SEED)
    classes = sorted(train["class"].unique())
    train["label"] = train["class"].map({c: i for i, c in enumerate(classes)})
    val["label"] = val["class"].map({c: i for i, c in enumerate(classes)})

    os.makedirs(output_dir, exist_ok=True)

    clf = DistilBERTClassifier()
    clf.train(train, val, classes, output_dir=output_dir, num_epochs=num_epochs, batch_size=batch_size)

    print("Training finished. Checkpoint saved to:", output_dir)


if __name__ == '__main__':
    main()
