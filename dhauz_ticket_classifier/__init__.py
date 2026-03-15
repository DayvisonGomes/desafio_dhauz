"""dhauz_ticket_classifier package

Exposes main classes for convenience.
"""
__version__ = "1.0.0"

__all__ = [
    "RAGClassifier",
    "HybridClassifier",
    "DistilBERTClassifier",
    "DataDownloader",
]

# Note: avoid importing heavy submodules at package import time to keep scripts lightweight.
# Consumers should import specific classes when needed, e.g.:
# from dhauz_ticket_classifier.rag.classifier import RAGClassifier
