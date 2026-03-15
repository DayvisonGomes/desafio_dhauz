"""RAG retriever utilities ported from notebook functions"""
from typing import List, Union, Dict
import torch
import numpy as np


def bert_class_scores_batch(tickets: Union[str, List[str]], tokenizer, model, classes: List[str], device: str = "cuda") -> List[Dict[str, float]]:
    if isinstance(tickets, str):
        tickets = [tickets]

    inputs = tokenizer(tickets, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
    scores = [dict(zip(classes, p)) for p in probs]
    return scores


def top_k_classes_batch(tickets, tokenizer, model, classes, k=3, device: str = "cuda"):
    scores_batch = bert_class_scores_batch(tickets, tokenizer, model, classes, device)
    top_classes = []
    for scores in scores_batch:
        sorted_classes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_classes.append([c[0] for c in sorted_classes[:k]])
    return top_classes


def retrieve_examples_batch(tickets, vector_db, tokenizer, model, classes, k=3, device: str = "cuda"):
    single_input = False
    if isinstance(tickets, str):
        tickets = [tickets]
        single_input = True

    top_classes = top_k_classes_batch(tickets, tokenizer, model, classes, k, device)
    retrieved_examples = []
    for ticket, candidate_classes in zip(tickets, top_classes):
        examples = []
        for cls in candidate_classes:
            docs = vector_db.retrieve(ticket, class_filter=cls, k=5)
            if isinstance(docs, list):
                examples.extend(docs)
            else:
                examples.append(docs)
        examples = sorted(examples, key=lambda d: d.metadata.get("score", 0), reverse=True)
        retrieved_examples.append(examples[:k])

    if single_input:
        return retrieved_examples[0]
    return retrieved_examples


def bert_topk_with_confidence(tickets, tokenizer, model, classes, k=3, device: str = "cuda"):
    scores_batch = bert_class_scores_batch(tickets, tokenizer, model, classes, device)
    results = []
    for scores in scores_batch:
        sorted_classes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_classes = [c[0] for c in sorted_classes[:k]]
        top_probs = [c[1] for c in sorted_classes[:k]]
        results.append({"top_classes": top_classes, "top_probs": top_probs, "confidence": top_probs[0]})
    return results
