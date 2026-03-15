"""Script: download dataset and preprocess"""
from pathlib import Path
import os
import sys

# Ensure project root is on sys.path when running this script directly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dhauz_ticket_classifier.data.download import DataDownloader
from dhauz_ticket_classifier.data.preprocessing import TextPreprocessor


def main():
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading dataset...")
    df = DataDownloader.download()
    print(f"Dataset shape: {df.shape}")
    pre = TextPreprocessor()
    df["clean_text"] = df["text"].apply(pre.preprocess)
    out_file = Path(out_dir) / "dataset_processed.csv"
    df.to_csv(out_file, index=False)
    print("Saved to", out_file)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Download and preprocess dataset')
    parser.add_argument('--out-dir', type=str, default='data', help='Output directory for processed CSV')
    args = parser.parse_args()

    # allow custom out-dir
    def run_with_args(out_dir: str = args.out_dir):
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        print("Downloading dataset...")
        df = DataDownloader.download()
        pre = TextPreprocessor()
        df["clean_text"] = df["text"].apply(pre.preprocess)
        saved = out_path / "dataset_processed.csv"
        df.to_csv(saved, index=False)
        print("Saved to", saved)

    run_with_args()
