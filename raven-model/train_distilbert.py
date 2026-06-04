import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import (
    DataCollatorWithPadding,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for Raven toxicity detection.")
    parser.add_argument("--train-csv", required=True, help="CSV containing comment text and toxicity labels.")
    parser.add_argument("--output-dir", default="models/raven-distilbert", help="Where to save the Raven model.")
    parser.add_argument("--model-name", default="distilbert-base-uncased", help="Base Hugging Face model.")
    parser.add_argument("--text-column", default="comment_text", help="Text column name.")
    parser.add_argument("--label-column", default="target", help="Label column name.")
    parser.add_argument("--positive-threshold", type=float, default=0.5, help="Convert float toxicity target to class 1.")
    parser.add_argument("--validation-size", type=float, default=0.1, help="Validation split size.")
    parser.add_argument("--sample-size", type=int, default=0, help="Optional deterministic sample for quick runs.")
    parser.add_argument("--max-length", type=int, default=128, help="Tokenizer max length.")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    return parser.parse_args()


def load_frame(args):
    df = pd.read_csv(args.train_csv)
    missing = {args.text_column, args.label_column} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[[args.text_column, args.label_column]].dropna()
    df = df.rename(columns={args.text_column: "text", args.label_column: "raw_label"})
    df["text"] = df["text"].astype(str)
    df["label"] = (df["raw_label"].astype(float) >= args.positive_threshold).astype(int)
    df = df[["text", "label"]]

    if args.sample_size and args.sample_size < len(df):
        df = df.sample(args.sample_size, random_state=42)

    return df.reset_index(drop=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    try:
      probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
      metrics["roc_auc"] = roc_auc_score(labels, probs[:, 1])
    except ValueError:
      pass

    return metrics


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_frame(args)
    train_df, val_df = train_test_split(
        df,
        test_size=args.validation_size,
        random_state=42,
        stratify=df["label"] if df["label"].nunique() == 2 else None,
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name)
    model = DistilBertForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_dataset = Dataset.from_pandas(train_df, preserve_index=False).map(tokenize, batched=True)
    val_dataset = Dataset.from_pandas(val_df, preserve_index=False).map(tokenize, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics_path = output_dir / "eval_metrics.json"
    metrics_path.write_text(pd.Series(metrics).to_json(indent=2), encoding="utf-8")
    print(f"Saved Raven model to {output_dir}")
    print(f"Saved evaluation metrics to {metrics_path}")


if __name__ == "__main__":
    main()
