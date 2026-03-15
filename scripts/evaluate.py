"""Avaliação: seleciona N amostras fixas e gera classification_report + confusion matrix heatmap.

Usage:
    python scripts/evaluate.py --processed data/dataset_processed.csv --checkpoint ./results --out-dir ./results --sample-size 200
"""
from pathlib import Path
import sys

# ensure project root is on sys.path when running the script directly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

import argparse
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.metrics import classification_report, confusion_matrix
from dhauz_ticket_classifier.utils.helpers import save_classification_report
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split

from dhauz_ticket_classifier.models.distilbert_classifier import DistilBERTClassifier
from dhauz_ticket_classifier.config import TRAIN_TEST_SPLIT_SEED


def evaluate(processed_csv: str, checkpoint: str, out_dir: str, sample_size: int = 200, use_train: bool = False, seed: int = TRAIN_TEST_SPLIT_SEED):
    p = Path(processed_csv)
    if not p.exists():
        raise FileNotFoundError(f"Processed dataset not found: {processed_csv}")

    df = pd.read_csv(p)
    if "text" not in df.columns or "class" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'class' columns")

    train, val = train_test_split(df, test_size=0.2, stratify=df["class"], random_state=seed)
    classes = sorted(train["class"].unique())

    if use_train:
        pool = train
    else:
        pool = val

    if len(pool) < sample_size:
        raise ValueError(f"Pool has only {len(pool)} rows, cannot sample {sample_size}")

    sample = pool.sample(sample_size, random_state=seed).reset_index(drop=True)

    # Load model
    clf = DistilBERTClassifier()
    clf.load(checkpoint)

    texts = sample["text"].astype(str).tolist()

    # Predict in batches
    batch_size = 32
    all_preds = []
    for i in tqdm(range(0, len(texts), batch_size), desc='Batches'):
        batch = texts[i:i+batch_size]
        probs = clf.predict_batch(batch)
        preds = np.argmax(probs, axis=1).tolist()
        all_preds.extend(preds)

    # Map indices to class names
    pred_labels = [classes[p] for p in all_preds]
    true_labels = sample["class"].tolist()

    # Classification report (dict saved to JSON)
    report_dict = classification_report(true_labels, pred_labels, digits=4, output_dict=True)
    print(classification_report(true_labels, pred_labels, digits=4))

    save_info = save_classification_report(report_dict, classes, out_dir=out_dir, prefix='classification_report')
    report_path = save_info['report']
    mapping_path = save_info['class_map']

    # Confusion matrix heatmap
    cm = confusion_matrix(true_labels, pred_labels, labels=classes)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix ({sample_size} samples)")
    heatmap_path = Path(out_dir) / "confusion_matrix_heatmap.png"
    fig.savefig(heatmap_path, bbox_inches="tight")
    plt.close(fig)

    # Save predictions
    out_csv = Path(out_dir) / "evaluation_predictions.csv"
    sample_out = sample.copy()
    sample_out["predicted"] = pred_labels
    sample_out.to_csv(out_csv, index=False)

    print("Saved heatmap:", heatmap_path)
    print("Saved predictions:", out_csv)
    print("Saved classification report:", report_path)
    print("Saved class-index mapping:", mapping_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=str, default="data/dataset_processed.csv")
    parser.add_argument("--checkpoint", type=str, default="./results")
    parser.add_argument("--out-dir", type=str, default="./results")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--use-train", action="store_true", help="Sample from train instead of validation")
    parser.add_argument("--seed", type=int, default=TRAIN_TEST_SPLIT_SEED)
    args = parser.parse_args()

    evaluate(args.processed, args.checkpoint, args.out_dir, sample_size=args.sample_size, use_train=args.use_train, seed=args.seed)


if __name__ == '__main__':
    main()
