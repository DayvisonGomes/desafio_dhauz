"""Script: download dataset and preprocess"""
from pathlib import Path
import os
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
    df.to_csv(out_dir / "dataset_processed.csv", index=False)
    print("Saved to", out_dir / "dataset_processed.csv")


if __name__ == '__main__':
    main()
