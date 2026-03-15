"""RAG and Hybrid classifiers encapsulating notebook logic"""
from typing import List, Union, Dict
from ..models.distilbert_classifier import DistilBERTClassifier
from ..models.llm import LLMPipeline
from .vector_store import VectorStore
from .retriever import retrieve_examples_batch, bert_topk_with_confidence
from ..utils.json_parser import extract_json
from ..config import HYBRID_CONFIDENCE_THRESHOLD, LLM_BATCH_SIZE
from langchain_core.messages import HumanMessage, SystemMessage


class RAGClassifier:
    def __init__(self, distilbert: DistilBERTClassifier, vector_store: VectorStore, llm_pipeline: LLMPipeline, classes: List[str], llm_batch_size: int = LLM_BATCH_SIZE):
        self.distilbert = distilbert
        self.vector_store = vector_store
        self.llm_pipeline = llm_pipeline
        self.classes = classes
        self.llm_batch_size = llm_batch_size

    def classify_batch(self, tickets: Union[str, List[str]], batch_size: int = None) -> List[Dict]:
        if isinstance(tickets, str):
            tickets = [tickets]
        if batch_size is None:
            batch_size = self.llm_batch_size

        prompts = []
        for ticket in tickets:
            retrieved = retrieve_examples_batch(ticket, self.vector_store, self.distilbert.tokenizer, self.distilbert.model, self.classes)
            examples = ""
            for doc in retrieved:
                examples += f"""\nTicket: {doc.metadata['original_text']}\nClass: {doc.metadata['class']}\n"""

            user_prompt = f"""Classify the IT support ticket.\n\nTicket:\n{ticket}\n\nChoose EXACTLY ONE category from:\n{', '.join(self.classes)}\n\nReturn ONLY this JSON and NOTHING ELSE:\n\n{{"class": "<category>", "justification": "<short reason>"}}"""

            messages = [SystemMessage(content=f"You are an AI assistant specialized in IT support ticket classification.\n\nONLY Possible categories:\n{', '.join(self.classes)}\n\nExamples:\n{examples}"), HumanMessage(content=user_prompt)]

            # Build prompt using tokenizer if available
            try:
                prompt = self.llm_pipeline.tokenizer.apply_chat_template([
                    {"role": "system", "content": messages[0].content},
                    {"role": "user", "content": messages[1].content}
                ], tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt = messages[1].content

            prompts.append(prompt)

        outputs = self.llm_pipeline.generate(prompts, batch_size=batch_size)

        results = []
        for out in outputs:
            if isinstance(out, list):
                generated = out[0].get("generated_text", "")
            else:
                generated = out.get("generated_text", "") if isinstance(out, dict) else str(out)
            parsed = extract_json(generated)
            if parsed is None or parsed.get("class") not in self.classes:
                parsed = {"class": "Miscellaneous", "justification": "Invalid model output"}
            results.append(parsed)

        return results


class HybridClassifier:
    def __init__(self, rag_classifier: RAGClassifier, confidence_threshold: float = HYBRID_CONFIDENCE_THRESHOLD, classes: List[str] = None):
        self.rag = rag_classifier
        self.confidence_threshold = confidence_threshold
        self.classes = classes or rag_classifier.classes

    def classify_batch(self, tickets: Union[str, List[str]], batch_size: int = None) -> List[Dict]:
        if isinstance(tickets, str):
            tickets = [tickets]

        bert_results = bert_topk_with_confidence(tickets, self.rag.distilbert.tokenizer, self.rag.distilbert.model, self.classes)

        final_results = []
        rag_tickets = []
        rag_candidates = []
        rag_indices = []

        for i, (ticket, bert_res) in enumerate(zip(tickets, bert_results)):
            confidence = bert_res["confidence"]
            top_classes = bert_res["top_classes"]
            if confidence >= self.confidence_threshold:
                final_results.append({"class": top_classes[0], "justification": "High confidence DistilBERT prediction"})
            else:
                rag_tickets.append(ticket)
                rag_candidates.append(top_classes)
                rag_indices.append(i)
                final_results.append(None)

        if len(rag_tickets) > 0:
            rag_outputs = self._rag_classifier_batch_hybrid(rag_tickets, rag_candidates, batch_size=batch_size)
            for idx, out in zip(rag_indices, rag_outputs):
                final_results[idx] = out

        return final_results

    def _rag_classifier_batch_hybrid(self, tickets: List[str], candidate_classes_batch: List[List[str]], batch_size: int = None):
        prompts = []
        for ticket, candidate_classes in zip(tickets, candidate_classes_batch):
            retrieved = retrieve_examples_batch(ticket, self.rag.vector_store, self.rag.distilbert.tokenizer, self.rag.distilbert.model, self.classes)
            examples = ""
            for doc in retrieved:
                examples += f"""\nTicket: {doc.metadata['original_text']}\nClass: {doc.metadata['class']}\n"""

            user_prompt = f"""Classify the IT support ticket.\n\nTicket:\n{ticket}\n\nChoose EXACTLY ONE category from:\n\n1. {candidate_classes[0]}\n2. {candidate_classes[1]}\n3. {candidate_classes[2]}\n\nReturn ONLY this JSON:\n\n{{"class": "<category>", "justification": "<short reason>"}}"""

            messages = [SystemMessage(content=f"You are an AI assistant specialized in IT support ticket classification.\n\nPossible categories:\n{', '.join(candidate_classes)}\n\nExamples:\n{examples}"), HumanMessage(content=user_prompt)]

            try:
                prompt = self.rag.llm_pipeline.tokenizer.apply_chat_template([
                    {"role": "system", "content": messages[0].content},
                    {"role": "user", "content": messages[1].content}
                ], tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt = messages[1].content

            prompts.append(prompt)

        outputs = self.rag.llm_pipeline.generate(prompts, batch_size=batch_size or self.rag.llm_batch_size)
        results = []
        for out in outputs:
            if isinstance(out, list):
                generated = out[0].get("generated_text", "")
            else:
                generated = out.get("generated_text", "") if isinstance(out, dict) else str(out)
            parsed = extract_json(generated)
            if parsed is None:
                parsed = {"class": "Miscellaneous", "justification": "Invalid model output"}
            results.append(parsed)
        return results
