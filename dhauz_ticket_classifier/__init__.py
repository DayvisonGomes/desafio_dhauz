"""dhauz_ticket_classifier package

Exposes main classes for convenience.
"""
__version__ = "1.0.0"

from .rag.classifier import RAGClassifier, HybridClassifier
from .models.distilbert_classifier import DistilBERTClassifier
from .data.download import DataDownloader

__all__ = [
    "RAGClassifier",
    "HybridClassifier",
    "DistilBERTClassifier",
    "DataDownloader",
]
