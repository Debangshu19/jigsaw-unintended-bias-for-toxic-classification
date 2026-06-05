import os
import json
from dataclasses import dataclass, field
from typing import Iterable

DEFAULT_THRESHOLD = 0.5
AI_GATEWAY_URL = "https://ai-gateway.vercel.sh/v1/chat/completions"
AI_GATEWAY_MODEL = os.getenv("RAVEN_AI_GATEWAY_MODEL", "google/gemini-3.5-flash")
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


CATEGORY_LABELS = {
    "toxic": "Toxic",
    "severe_toxic": "Severe toxic",
    "severe_toxicity": "Severe toxic",
    "obscene": "Obscene",
    "threat": "Threat",
    "insult": "Insult",
    "identity_hate": "Identity hate",
    "identity_attack": "Identity hate",
}


@dataclass
class Prediction:
    label: str
    score: float
    needs_review: bool
    source: str
    categories: dict = field(default_factory=dict)
    top_category: str = None


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

    def explain_one(self, text: str) -> dict:
        prediction = self.predict_one(text)
        words = []
        for word in text.split()[:60]:
            cleaned = word.lower().strip(".,!?\"'();:")
            hit = any(term in cleaned for term in self.review_terms)
            words.append({"w": word, "s": 0.3 if hit else 0.0})
        out = prediction.__dict__.copy()
        out["categories"] = {"toxic": prediction.score} if prediction.score else {}
        out["top_category"] = "toxic" if prediction.needs_review else None
        out["words"] = words
        return out


class AiGatewayFallbackModel:
    source = "raven-ai-gateway-fallback"

    def __init__(self) -> None:
        self.api_key = os.getenv("AI_GATEWAY_API_KEY")
        self.threshold = float(os.getenv("RAVEN_THRESHOLD", DEFAULT_THRESHOLD))
        self.model_ref = AI_GATEWAY_MODEL
        if not self.api_key:
            raise RuntimeError("AI_GATEWAY_API_KEY is not configured")

    def predict_one(self, text: str) -> Prediction:
        import httpx

        response = httpx.post(
            AI_GATEWAY_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_ref,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are Raven, a strict toxicity classifier for public social media comments. "
                            "Classify hate speech, harassment, insults, severe profanity, threats, and identity attacks as review. "
                            "Classify normal disagreement, criticism without abuse, and neutral comments as safe. "
                            "Return only compact JSON with keys label, score, needs_review. "
                            "label must be safe or review. score must be a number from 0 to 1."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
            timeout=18,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        data = json.loads(content)
        score = max(0.0, min(1.0, float(data.get("score", 0))))
        needs_review = bool(data.get("needs_review", score >= self.threshold))
        label = "review" if needs_review else "safe"

        return Prediction(
            label=label,
            score=round(score, 4),
            needs_review=needs_review,
            source=self.source,
        )

    def predict_batch(self, texts: Iterable[str]) -> list[Prediction]:
        return [self.predict_one(text) for text in texts]

    def explain_one(self, text: str) -> dict:
        prediction = self.predict_one(text)
        out = prediction.__dict__.copy()
        out["categories"] = {"toxic": prediction.score} if prediction.score else {}
        out["top_category"] = "toxic" if prediction.needs_review else None
        out["words"] = [{"w": word, "s": 0.0} for word in text.split()[:60]]
        return out


class ResilientModel:
    def __init__(self, primary, fallback) -> None:
        self.primary = primary
        self.fallback = fallback
        self.source = getattr(primary, "source", "raven-primary-model")
        self.fallback_source = getattr(fallback, "source", "unknown")
        self.threshold = getattr(primary, "threshold", getattr(fallback, "threshold", DEFAULT_THRESHOLD))
        self.model_ref = getattr(primary, "model_ref", None)

    def predict_one(self, text: str) -> Prediction:
        try:
            return self.primary.predict_one(text)
        except Exception:
            return self.fallback.predict_one(text)

    def predict_batch(self, texts: Iterable[str]) -> list[Prediction]:
        texts = list(texts)
        try:
            return self.primary.predict_batch(texts)
        except Exception:
            return self.fallback.predict_batch(texts)

    def explain_one(self, text: str) -> dict:
        try:
            return self.primary.explain_one(text)
        except Exception:
            return self.fallback.explain_one(text)


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

    def _toxic_indexes(self, label_count: int) -> list:
        return [
            index
            for index in range(label_count)
            if any(hint in self.id2label.get(index, "") for hint in TOXIC_LABEL_HINTS)
            and not any(hint in self.id2label.get(index, "") for hint in SAFE_LABEL_HINTS)
        ]

    def _category_scores(self, logits) -> dict:
        label_count = int(logits.shape[-1])
        toxic_indexes = self._toxic_indexes(label_count)
        # Multi-label heads (e.g. toxic-bert): independent sigmoid per category.
        if len(toxic_indexes) > 1:
            probs = self.torch.sigmoid(logits)
            return {self.id2label.get(i, f"label_{i}"): round(float(probs[i]), 4) for i in toxic_indexes}
        # Single toxic head (binary fine-tuned model): one category.
        return {"toxic": round(self._toxic_score(logits), 4)}

    def _top_category(self, categories: dict):
        if not categories:
            return None
        # Prefer the most specific type (insult, threat, identity hate…) over the
        # generic "toxic" umbrella, when one is meaningfully present.
        specific = {k: v for k, v in categories.items() if k != "toxic"}
        if specific:
            best = max(specific, key=specific.get)
            if specific[best] >= 0.15:
                return best
        return max(categories, key=categories.get)

    def _prediction_from_logits(self, logits) -> Prediction:
        score = self._toxic_score(logits)
        categories = self._category_scores(logits)
        top_category = self._top_category(categories)
        needs_review = score >= self.threshold
        return Prediction(
            label="review" if needs_review else "safe",
            score=round(score, 4),
            needs_review=needs_review,
            source=self.source,
            categories=categories,
            top_category=top_category,
        )

    def _logits_for(self, texts: list):
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        with self.torch.no_grad():
            return self.model(**inputs).logits

    def predict_batch(self, texts: Iterable[str]) -> list[Prediction]:
        logits = self._logits_for(list(texts))
        return [self._prediction_from_logits(row) for row in logits]

    def explain_one(self, text: str) -> dict:
        words = text.split()[:60]
        if not words:
            return {**self.predict_one(text).__dict__, "words": []}

        # Leave-one-out occlusion: how much does removing each word drop the toxicity?
        variants = [text] + [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
        logits = self._logits_for(variants)
        base = self._toxic_score(logits[0])
        word_scores = [
            {"w": word, "s": round(float(base - self._toxic_score(logits[i + 1])), 4)}
            for i, word in enumerate(words)
        ]

        out = self._prediction_from_logits(logits[0]).__dict__.copy()
        out["words"] = word_scores
        return out


def load_model():
    fallback = DemoFallbackModel()
    if os.getenv("AI_GATEWAY_API_KEY"):
        try:
            fallback = AiGatewayFallbackModel()
        except Exception:
            fallback = DemoFallbackModel()

    model_dir = os.getenv("RAVEN_MODEL_DIR")
    if model_dir and os.path.isdir(model_dir):
        try:
            return ResilientModel(TransformersModel(model_dir, "raven-local-model"), fallback)
        except Exception:
            return fallback

    model_id = os.getenv("RAVEN_MODEL_ID")
    if model_id:
        try:
            return ResilientModel(TransformersModel(model_id, "raven-hf-model"), fallback)
        except Exception:
            return fallback

    return fallback
