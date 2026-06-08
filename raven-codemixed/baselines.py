"""RAVEN-X — REAL classification baselines that run on a CPU in seconds.

This is genuine, reportable evidence you can produce today without a GPU:
  B6  TF-IDF (word + char n-grams) + Logistic Regression   (the benchmark's lower bound)

It trains/evaluates on the SAME frozen seed=42 HASOC-2019 Hindi split the transformer
models will use, so the numbers are directly comparable to the MuRIL/XLM-R rows you'll
add from the free-T4 run. No fabrication — actual fit on actual data.

    raven-api/.venv/bin/python raven-codemixed/baselines.py
"""
from __future__ import annotations
import data_loaders as dl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
from scipy.sparse import hstack


def vectorize(train_txt, *other_txt):
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=40000)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=40000)
    Xtr = hstack([word.fit_transform(train_txt), char.fit_transform(train_txt)]).tocsr()
    outs = []
    for txt in other_txt:
        outs.append(hstack([word.transform(txt), char.transform(txt)]).tocsr())
    return Xtr, outs


def report(name, y, pred):
    macro = f1_score(y, pred, average="macro")
    acc = accuracy_score(y, pred)
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[1], average=None)
    print(f"  {name:5s}  macro-F1={macro:.4f}  acc={acc:.4f}  "
          f"toxic P={p[0]:.3f} R={r[0]:.3f} F1={f[0]:.3f}")
    return macro


if __name__ == "__main__":
    d = dl.load_hasoc_hindi()
    tr, va, te = d["train"], d["val"], d["test"]
    print(f"HASOC-2019 Hindi frozen split (seed=42): "
          f"train={len(tr)} val={len(va)} test={len(te)}")
    Xtr, (Xva, Xte) = vectorize([x["text"] for x in tr],
                                [x["text"] for x in va], [x["text"] for x in te])
    ytr = [x["label"] for x in tr]; yva = [x["label"] for x in va]; yte = [x["label"] for x in te]

    clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
    clf.fit(Xtr, ytr)

    print("\nB6 — TF-IDF(word 1-2 + char 2-5) + LogReg  [REAL, CPU]:")
    report("val", yva, clf.predict(Xva))
    test_macro = report("test", yte, clf.predict(Xte))
    print(f"\n>> Honest lower-bound on the frozen test split: macro-F1 = {test_macro:.4f}")
    print(">> The MuRIL/XLM-R rows (from the free-T4 run) go ABOVE this in the same table.")
