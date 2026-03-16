"""DistilBERT training and inference helper"""
from typing import List
import numpy as np
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score


class DistilBERTClassifier:
    def __init__(self, model_name: str = "distilbert-base-uncased", device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        self.model = None

    def train(self, train_df, val_df, classes: List[str], output_dir: str = "./results", num_epochs: int = 2, batch_size: int = 32):       
        train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
        val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))

        def tokenize(example):
            return self.tokenizer(example["text"], padding="max_length", truncation=True, max_length=256)

        train_ds = train_ds.map(tokenize, batched=True)
        val_ds = val_ds.map(tokenize, batched=True)

        train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        self.model = DistilBertForSequenceClassification.from_pretrained(self.model_name, num_labels=len(classes))

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            num_train_epochs=num_epochs,
            logging_steps=100,
            fp16=torch.cuda.is_available(),
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=1)
            return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds, average="macro")}

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics
        )

        trainer.train()

        # Save final model and tokenizer
        try:
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
        except Exception:
            pass

    def predict_batch(self, texts: List[str]):
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        self.model.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        return probs

    def predict_with_confidence(self, text: str, top_k: int = 3, classes: List[str] = None):

        probs = self.predict_batch([text])[0]
        sorted_idx = np.argsort(probs)[::-1]
        top_idx = sorted_idx[:top_k]

        top_classes = [classes[i] for i in top_idx]
        prediction = top_classes[0]
        dict_conf_pred = {classes[i]: float(probs[i]) for i in top_idx}

        return prediction, dict_conf_pred, top_classes

    def save(self, output_dir: str):
        """Save model and tokenizer to `output_dir`"""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def load(self, checkpoint_dir: str):
        """Load model and tokenizer from `checkpoint_dir`"""
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(checkpoint_dir, local_files_only=True)
        self.model = DistilBertForSequenceClassification.from_pretrained(checkpoint_dir, local_files_only=True)
        self.model.to(self.device)

