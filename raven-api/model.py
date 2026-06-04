import os
from dataclasses import dataclass
from typing import Iterable

DEFAULT_THRESHOLD = 0.5


@dataclass
class Prediction:
    label: str
    score: float
    needs_review: bool
    source: str


class DemoFallbackModel:
    source = "raven-api-demo-fallback"

    def __init__(self) -> None:
        self.threshold = float(os.getenv("RAVEN_THRESHOLD", DEFAULT_THRESHOLD))
        self.review_terms = {
            "aggressive",
            "attack",
            "bully",
            "hate",
            "hurt",
            "insult",
            "moderator",
            "not cool",
            "review",
            "reviewed",
            "stupid",
            "threat",
            "abuse",
            "harass",
            "violent",
        }

    def predict_one(self, text: str) -> Prediction:
        normalized = text.lower()
        hits = sum(1 for term in self.review_terms if term in normalized)
        score = min(0.96, 0.12 + hits * 0.28 + min(len(text) / 500, 0.18))
        needs_review = hits > 0 or score >= self.threshold
        return Prediction(
            label="review" if needs_review else "safe",
            score=round(score, 4),
            needs_review=needs_review,
            source=self.source,
        )

    def predict_batch(self, texts: Iterable[str]) -> list[Prediction]:
        return [self.predict_one(text) for text in texts]


class TransformersModel:
    source = "raven-local-model"

    def __init__(self, model_dir: str) -> None:
        import torch
        import torch.nn.functional as functional
        from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

        self.torch = torch
        self.functional = functional
        self.threshold = float(os.getenv("RAVEN_THRESHOLD", DEFAULT_THRESHOLD))
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

    def predict_one(self, text: str) -> Prediction:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )

        with self.torch.no_grad():
            outputs = self.model(**inputs)

        probs = self.functional.softmax(outputs.logits, dim=1)[0]
        toxic_score = float(probs[1])
        needs_review = toxic_score >= self.threshold

        return Prediction(
            label="review" if needs_review else "safe",
            score=round(toxic_score, 4),
            needs_review=needs_review,
            source=self.source,
        )

    def predict_batch(self, texts: Iterable[str]) -> list[Prediction]:
        texts = list(texts)
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )

        with self.torch.no_grad():
            outputs = self.model(**inputs)

        probs = self.functional.softmax(outputs.logits, dim=1)
        predictions = []
        for row in probs:
            toxic_score = float(row[1])
            needs_review = toxic_score >= self.threshold
            predictions.append(
                Prediction(
                    label="review" if needs_review else "safe",
                    score=round(toxic_score, 4),
                    needs_review=needs_review,
                    source=self.source,
                )
            )

        return predictions


def load_model():
    model_dir = os.getenv("RAVEN_MODEL_DIR")
    if model_dir and os.path.isdir(model_dir):
        return TransformersModel(model_dir)
    return DemoFallbackModel()
