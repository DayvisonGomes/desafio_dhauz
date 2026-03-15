"""Gradio interface wrapper for the classifier"""
import gradio as gr


class GradioApp:
    def __init__(self, rag_classifier, hybrid_classifier=None):
        self.rag = rag_classifier
        self.hybrid = hybrid_classifier

    def classify_rag(self, ticket: str):
        result = self.rag.classify_batch([ticket])[0]
        return result.get('class'), result.get('justification')

    def classify_hybrid(self, ticket: str):
        if self.hybrid:
            result = self.hybrid.classify_batch([ticket])[0]
        else:
            result = self.rag.classify_batch([ticket])[0]
        return result.get('class'), result.get('justification')

    def launch(self, mode: str = "hybrid", **kwargs):
        fn = self.classify_hybrid if mode == "hybrid" else self.classify_rag
        demo = gr.Interface(
            fn=fn,
            inputs=gr.Textbox(lines=4, label="IT Support Ticket"),
            outputs=[gr.Label(label="Predicted Class"), gr.Textbox(label="Justification")],
            examples=[
                "User cannot connect to VPN after password reset",
                "Laptop screen broken need replacement",
                "Create account for new employee",
                "Need access to Oracle reports"
            ],
            title="IT Ticket Classifier (RAG)",
            description="Hybrid classifier using DistilBERT + Retrieval + LLM reasoning.",
            **kwargs
        )
        demo.launch()
