"""Configuration and seed utilities"""
import os
import random
import numpy as np
import torch

# Seeds
RANDOM_SEED = 42

def set_seeds(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DB_PATH = os.path.join(DATA_DIR, "chroma_db")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Model names
DISTILBERT_MODEL_NAME = "distilbert-base-uncased"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# Inference/training defaults
TRAIN_TEST_SPLIT_SEED = 42
TRAIN_BATCH_SIZE = 32
LLM_BATCH_SIZE = 4
HYBRID_CONFIDENCE_THRESHOLD = 0.9
