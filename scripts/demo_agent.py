"""
CLI demo launcher.

Loads:
- DistilBERT checkpoint
- VectorStore
- Optional LLM
- RAG or Hybrid classifier
- LangGraph agent for routing
"""

import argparse
from pathlib import Path
import sys
from typing import TypedDict, Optional

# ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from dhauz_ticket_classifier.models.distilbert_classifier import DistilBERTClassifier
from dhauz_ticket_classifier.models.llm import LLMPipeline

from dhauz_ticket_classifier.rag.vector_store import VectorStore
from dhauz_ticket_classifier.rag.classifier import RAGClassifier, HybridClassifier

from langgraph.graph import StateGraph, END


# =========================
# Arguments
# =========================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./results",
        help="DistilBERT checkpoint directory"
    )

    parser.add_argument(
        "--chroma-dir",
        type=str,
        default="data/chroma_db",
        help="Vector database directory"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/dataset_processed.csv",
        help="Dataset used to infer classes"
    )

    parser.add_argument(
        "--mode",
        choices=["rag", "hybrid"],
        default="hybrid"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9
    )

    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Load LLM for RAG reasoning"
    )
    
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="LLM model name (overrides default in config)"
    )
    
    return parser.parse_args()


# =========================
# LangGraph State
# =========================

class TicketState(TypedDict):

    ticket: str
    prediction: Optional[str]
    justification: Optional[str]
    confidence: Optional[float]
    mode: str


# =========================
# Main
# =========================

def main():

    args = parse_args()

    print("\nLoading DistilBERT...")

    distil = DistilBERTClassifier()
    distil.load(args.checkpoint)

    print("DistilBERT loaded")

    # =========================
    # Load classes
    # =========================

    dataset_path = Path(args.dataset)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    classes = sorted(df["class"].unique())

    print("Detected classes:", classes)

    # =========================
    # Vector Store
    # =========================

    vs = VectorStore(persist_dir=args.chroma_dir)

    if vs.exists():

        print("Loading Chroma DB...")
        vs.load()

    else:

        print("Building Chroma DB from dataset...")
        vs.create_from_dataframe(df)

    # =========================
    # Optional LLM
    # =========================

    llm = None

    if args.use_llm:

        print("Loading LLM...")
        llm = LLMPipeline(model_name=args.llm_model)

    else:

        print("LLM disabled (RAG will not work)")

    # =========================
    # Classifiers
    # =========================

    rag = RAGClassifier(
        distilbert=distil,
        vector_store=vs,
        llm_pipeline=llm,
        classes=classes
    )

    hybrid = HybridClassifier(
        rag_classifier=rag,
        confidence_threshold=args.threshold,
        classes=classes
    )

    print("Classifiers ready")

    # =========================
    # LangGraph Nodes
    # =========================

    def hybrid_node(state: TicketState):

        ticket = state["ticket"]
        bert_res = hybrid.rag.distilbert.predict_with_confidence(ticket)

        print("DistilBERT prediction:", bert_res)
        result = hybrid.classify_batch(ticket)[0]

        return {
            "prediction": result["class"],
            "justification": result["justification"]
        }

    def rag_node(state: TicketState):

        ticket = state["ticket"]

        result = rag.classify_batch(ticket)[0]

        return {
            "prediction": result["class"],
            "justification": result["justification"]
        }

    # =========================
    # Graph
    # =========================

    graph = StateGraph(TicketState)

    graph.add_node("hybrid", hybrid_node)
    graph.add_node("rag", rag_node)

    if args.mode == "hybrid":

        graph.set_entry_point("hybrid")

    else:

        graph.set_entry_point("rag")

    graph.add_edge("hybrid", END)
    graph.add_edge("rag", END)

    agent = graph.compile()

    print("\nAgent ready")
    print("Mode:", args.mode)

    # =========================
    # CLI
    # =========================

    while True:

        ticket = input("\nTicket (type 'exit'): ")

        if ticket.lower() in {"exit", "quit"}:
            break

        for step in agent.stream({
                "ticket": ticket,
                "mode": args.mode
            }):

            node = list(step.keys())[0]

            print(f"\nExecuted node: {node}")
            print("Output:", step[node])

            result = step[node]

        print("\nPrediction:", result["prediction"])
        print("Justification:", result["justification"])


if __name__ == "__main__":

    main()