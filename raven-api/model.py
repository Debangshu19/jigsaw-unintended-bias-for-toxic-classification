import os
from dataclasses import dataclass
from typing import Iterable

DEFAULT_THRESHOLD = 0.5
TOXIC_LABEL_HINTS = (
    "toxic",
    "severe",
    "obscene",
    "threat",
    "insult",
    "identity",
    "hate",
    "abuse",
    "offensive",
)
SAFE_LABEL_HINTS = ("safe", "clean", "neutral", "normal", "non_toxic", "not_toxic", "non-toxic")


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
    def __init__(self, model_ref: str, source: str) -> None:
        import torch
        import torch.nn.functional as functional
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.functional = functional
        self.threshold = float(os.getenv("RAVEN_THRESHOLD", DEFAULT_THRESHOLD))
        self.model_ref = model_ref
        self.source = source
        self.tokenizer = AutoTokenizer.from_pretrained(model_ref)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_ref)
        self.model.eval()
        self.id2label = {
            int(index): str(label).lower().replace(" ", "_")
            for index, label in getattr(self.model.config, "id2label", {}).items()
        }

    def predict_one(self, text: str) -> Prediction:
        return self.predict_batch([text])[0]

    def _toxic_score(self, logits) -> float:
        label_count = int(logits.shape[-1])

        if label_count == 1:
            return float(self.torch.sigmoid(logits)[0])

        toxic_indexes = [
            index
            for index in range(label_count)
            if any(hint in self.id2label.get(index, "") for hint in TOXIC_LABEL_HINTS)
            and not any(hint in self.id2label.get(index, "") for hint in SAFE_LABEL_HINTS)
        ]

        if toxic_indexes and len(toxic_indexes) > 1:
            probs = self.torch.sigmoid(logits)
            return max(float(probs[index]) for index in toxic_indexes)

        probs = self.functional.softmax(logits.unsqueeze(0), dim=1)[0]
        toxic_index = toxic_indexes[0] if toxic_indexes else min(1, label_count - 1)
        return float(probs[toxic_index])

    def _prediction(self, score: float) -> Prediction:
        needs_review = score >= self.threshold
        return Prediction(
            label="review" if needs_review else "safe",
            score=round(score, 4),
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

        return [self._prediction(self._toxic_score(row)) for row in outputs.logits]


def load_model():
    model_dir = os.getenv("RAVEN_MODEL_DIR")
    if model_dir and os.path.isdir(model_dir):
        return TransformersModel(model_dir, "raven-local-model")

    model_id = os.getenv("RAVEN_MODEL_ID")
    if model_id:
        return TransformersModel(model_id, "raven-hf-model")

    return DemoFallbackModel()
