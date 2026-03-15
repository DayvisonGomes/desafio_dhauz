"""Run batch inference using a saved DistilBERT checkpoint.

Usage:
  python scripts/infer_batch.py data/input_examples.txt --checkpoint ./results
"""
import argparse
from pathlib import Path
import json
from dhauz_ticket_classifier.models.distilbert_classifier import DistilBERTClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input file (one ticket per line) or CSV with 'text' column")
    parser.add_argument("--checkpoint", type=str, default="./results", help="Checkpoint directory")
    parser.add_argument("--out", type=str, default="predictions.jsonl", help="Output JSONL file")
    args = parser.parse_args()

    p = Path(args.input)
    if not p.exists():
        raise FileNotFoundError(args.input)

    lines = []
    if p.suffix.lower() in [".txt"]:
        with p.open("r", encoding="utf8") as f:
            lines = [l.strip() for l in f if l.strip()]
    else:
        # try csv
        import pandas as pd
        df = pd.read_csv(p)
        if "text" not in df.columns:
            raise ValueError("CSV must have a 'text' column")
        lines = df["text"].astype(str).tolist()

    clf = DistilBERTClassifier()
    clf.load(args.checkpoint)

    batch_size = 32
    results = []
    for i in range(0, len(lines), batch_size):
        batch = lines[i:i+batch_size]
        probs = clf.predict_batch(batch)
        preds = probs.argmax(axis=1)
        for t, pidx in zip(batch, preds):
            results.append({"text": t, "pred_idx": int(pidx)})

    with open(args.out, "w", encoding="utf8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("Wrote", args.out)


if __name__ == '__main__':
    main()
