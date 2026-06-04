import argparse
import json

import torch
import torch.nn.functional as functional
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast


def parse_args():
    parser = argparse.ArgumentParser(description="Run a Raven DistilBERT checkpoint on text.")
    parser.add_argument("--model-dir", default="models/raven-distilbert")
    parser.add_argument("--text", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    inputs = tokenizer(args.text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = functional.softmax(outputs.logits, dim=1)[0]
    toxic_score = float(probs[1])
    result = {
        "label": "review" if toxic_score >= args.threshold else "safe",
        "score": round(toxic_score, 4),
        "needs_review": toxic_score >= args.threshold,
        "source": "raven-local-model",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
