"""Fine-tune MuRIL on HASOC-2019 Hindi (task_1: HOF vs NOT) and export a
standard Hugging Face sequence-classification model that the Raven API can load
directly via RAVEN_HI_MODEL_DIR. Single classification head -> drop-in.

Run (from repo root, inside raven-api/.venv):
    python raven-codemixed/train_muril_hi.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data", "hasoc2019", "hindi_dataset.tsv")
OUT = os.path.join(ROOT, "models", "raven-muril-hi")
BACKBONE = os.getenv("MURIL_BACKBONE", "google/muril-base-cased")
EPOCHS = int(os.getenv("MURIL_EPOCHS", "3"))
MAX_LEN = 128
SEED = 42


class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    p, r, f, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": p,
        "recall": r,
        "f1": f,
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def main():
    df = pd.read_csv(DATA, sep="\t", quoting=3, engine="python")
    df = df[df["task_1"].isin(["HOF", "NOT"])].copy()
    df["label"] = (df["task_1"] == "HOF").astype(int)
    df["text"] = df["text"].astype(str)
    print(f"rows={len(df)}  toxic={df.label.sum()}  non_toxic={(df.label==0).sum()}")

    train_df, tmp_df = train_test_split(df, test_size=0.30, random_state=SEED, stratify=df["label"])
    val_df, test_df = train_test_split(tmp_df, test_size=0.50, random_state=SEED, stratify=tmp_df["label"])
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    tok = AutoTokenizer.from_pretrained(BACKBONE)

    def enc(texts):
        return tok(list(texts), truncation=True, padding=True, max_length=MAX_LEN)

    train_ds = TextDataset(enc(train_df.text), train_df.label.tolist())
    val_ds = TextDataset(enc(val_df.text), val_df.label.tolist())
    test_ds = TextDataset(enc(test_df.text), test_df.label.tolist())

    model = AutoModelForSequenceClassification.from_pretrained(
        BACKBONE,
        num_labels=2,
        id2label={0: "safe", 1: "toxic"},
        label2id={"safe": 0, "toxic": 1},
    )

    args = TrainingArguments(
        output_dir=os.path.join(HERE, "_muril_ckpt"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.06,
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="no",
        seed=SEED,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    print("VAL :", trainer.evaluate(val_ds))
    print("TEST:", trainer.evaluate(test_ds))

    os.makedirs(OUT, exist_ok=True)
    trainer.save_model(OUT)
    tok.save_pretrained(OUT)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
