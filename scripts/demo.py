"""Launch Gradio demo: loads DistilBERT checkpoint, VectorStore and optionally LLM/Agent.

Designed to be safe: LLM loading is optional (`--use-llm`) because it may require large GPU memory.
"""
import argparse
from pathlib import Path
import sys

# ensure project root is on sys.path when running the script directly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import os
import pandas as pd

from dhauz_ticket_classifier.rag.classifier import RAGClassifier, HybridClassifier
from dhauz_ticket_classifier.models.distilbert_classifier import DistilBERTClassifier
from dhauz_ticket_classifier.models.llm import LLMPipeline
from dhauz_ticket_classifier.rag.vector_store import VectorStore
from dhauz_ticket_classifier.interfaces.gradio_app import GradioApp
from dhauz_ticket_classifier.data.download import DataDownloader
from dhauz_ticket_classifier.utils.helpers import maybe_load_llm, infer_classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./results", help="DistilBERT checkpoint dir")
    parser.add_argument("--chroma-dir", type=str, default=None, help="Chroma DB directory (if omitted uses data/chroma_db)")
    parser.add_argument("--use-llm", action="store_true", help="Load LLM (heavy, optional)")
    parser.add_argument("--llm-model", type=str, default=None, help="LLM model name (overrides default in config)")
    parser.add_argument("--use-llm-remote", action="store_true", help="Use remote LLM endpoint instead of local model")
    parser.add_argument("--remote-llm-url", type=str, default=None, help="Remote LLM endpoint URL (full URL to POST to)")
    parser.add_argument("--remote-llm-key", type=str, default=None, help="API key for remote LLM endpoint (optional, can use env var)")
    parser.add_argument("--remote-llm-provider", type=str, choices=['hf','generic'], default='hf', help="Remote provider type: 'hf' for HuggingFace Inference API, 'generic' for a simple POST endpoint")
    parser.add_argument("--use-agent", action="store_true", help="Wrap classifier with LangChain agent (optional)")
    parser.add_argument("--mode", choices=["rag", "hybrid"], default="hybrid", help="Default interface mode")
    parser.add_argument("--export-chroma", type=str, default=None, help="Export Chroma DB to zip path after build")
    parser.add_argument("--import-chroma", type=str, default=None, help="Import Chroma DB from zip path before loading (overwrites if --import-overwrite)")
    parser.add_argument("--import-overwrite", action="store_true", help="Overwrite existing chroma directory when importing zip")
    args = parser.parse_args()

    # Ensure dataset exists to infer classes if needed
    data_csv = Path("data/dataset_processed.csv")
    if not data_csv.exists():
        print("Processed dataset not found locally — attempting to download via kagglehub (may require credentials).")
        try:
            df = DataDownloader.download()
            os.makedirs("data", exist_ok=True)
            df.to_csv(data_csv, index=False)
            print("Saved processed dataset to", data_csv)
        except Exception as e:
            print("Could not download dataset automatically:", e)

    # Load classes from dataset if available
    classes = None
    if data_csv.exists():
        try:
            df = pd.read_csv(data_csv)
            classes = sorted(df["class"].unique())
        except Exception:
            classes = None

    # Load or create VectorStore
    chroma_dir = args.chroma_dir or "data/chroma_db"
    vs = VectorStore(persist_dir=chroma_dir)
    # If user provided a zip to import, extract it first
    if args.import_chroma:
        try:
            print(f"Importing Chroma DB from {args.import_chroma} -> {chroma_dir}")
            vs.import_from_zip(args.import_chroma, overwrite=args.import_overwrite)
        except Exception as e:
            print("Failed to import chroma zip:", e)
            raise
    if vs.exists():
        print("Loading existing Chroma DB from", chroma_dir)
        try:
            vs.load()
        except Exception as e:
            print("Error loading Chroma DB:", e)
    else:
        print("Chroma DB not found. Building from dataset (this may take time)...")
        if data_csv.exists():
            df = pd.read_csv(data_csv)
            vs.create_from_dataframe(df)
            print("Chroma DB created at", vs.persist_directory)
        else:
            raise FileNotFoundError("No dataset available to build Chroma DB. Run scripts/download_data.py")

    # Load DistilBERT checkpoint
    distil = DistilBERTClassifier()
    ckpt = Path(args.checkpoint)
    if ckpt.exists():
        try:
            distil.load(str(ckpt))
            print("Loaded DistilBERT checkpoint from", ckpt)
        except Exception as e:
            print("Failed to load checkpoint:", e)
    else:
        print("Checkpoint not found at", ckpt, "— you may need to train first with scripts/train_distilbert.py")

    # LLM pipeline (optional). Supports local HF model or remote endpoint (--use-llm-remote)
    remote_key = args.remote_llm_key or os.environ.get('REMOTE_LLM_KEY')
    llm = maybe_load_llm(use_llm=args.use_llm, use_remote=args.use_llm_remote, remote_url=args.remote_llm_url, remote_key=remote_key, model_name=args.llm_model)

    # If classes are still unknown, try to infer from vector store metadata
    if classes is None:
        try:
            classes = []
        except Exception:
            classes = ["Miscellaneous"]

    # Instantiate classifiers
    rag = None
    hybrid = None
    if llm is None:
        print("LLM not loaded — RAG classifier will not be functional for LLM steps.")
    try:
        rag = RAGClassifier(distil, vs, llm, classes=classes)
        hybrid = HybridClassifier(rag)
    except Exception as e:
        print("Error instantiating classifiers:", e)

    # Optionally wrap in Agent (user must ensure a compatible 'llm' object for agent creation)
    if args.use_agent:
        try:
            from dhauz_ticket_classifier.agents.ticket_agent import TicketAgent
            agent = TicketAgent(rag_classifier=rag, llm=llm)
            if llm is not None:
                agent.create()
                print("Agent created")
            else:
                print("Agent not created because LLM is not available")
        except Exception as e:
            print("Failed to create agent:", e)

    # Launch Gradio
    if rag is None:
        raise RuntimeError("RAG classifier not instantiated — aborting demo")

    # Optionally export Chroma DB to zip for portability
    if args.export_chroma:
        try:
            print(f"Exporting Chroma DB to {args.export_chroma} ...")
            vs.export_to_zip(args.export_chroma)
            print("Export complete")
        except Exception as e:
            print("Failed to export chroma zip:", e)

    app = GradioApp(rag, hybrid)
    print("Launching Gradio (mode=", args.mode, ")")
    app.launch(mode=args.mode)


if __name__ == '__main__':
    main()
