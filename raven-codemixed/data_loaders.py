"""RAVEN-X — reusable corpus loaders (design-independent, stdlib only).

Two corpora feed the cross-lingual rationale-transfer model:
  * HASOC-2019 Hindi  -> code-mixed posts with POST-LEVEL labels (the classification signal)
  * HateXplain (EN)   -> posts with TOKEN-LEVEL rationale masks (the explanation signal)

These loaders return plain Python dicts so they stay independent of the chosen
backbone/tokenizer. Tokenization + subword<->rationale alignment happen later,
once the methodology (backbone, heads) is finalized.

Verified facts (see eda.py output, 2026-06-07):
  HASOC Hindi labeled = 4665 rows; official test is BLIND -> seeded stratified split.
  HateXplain = 20148 posts; 56.6% carry rationales; official divisions provided.
"""
from __future__ import annotations
import csv, json, os, random
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# ---- HASOC-2019 Hindi -------------------------------------------------------
HASOC_DIR = os.path.join(DATA, "hasoc2019")
# binary task_1: HOF (hate-or-offensive) = 1, NOT = 0
BIN = {"NOT": 0, "HOF": 1}
# fine-grained task_2
FINE = {"NONE": 0, "HATE": 1, "OFFN": 2, "PRFN": 3}


def _read_hasoc_tsv(path):
    out = []
    with open(path, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        header = next(r)
        labeled = len(header) >= 5
        for row in r:
            if not labeled or len(row) < 5 or row[2] not in BIN:
                continue
            out.append({
                "id": row[0],
                "text": row[1],
                "label": BIN[row[2]],          # binary HOF/NOT
                "fine": FINE.get(row[3], 0),   # hate/offn/prfn/none
                "target": row[4],              # TIN/UNT/NONE
            })
    return out


def load_hasoc_hindi(seed=42, ratios=(0.70, 0.15, 0.15)):
    """Stratified seeded split of the labeled set (official test is blind)."""
    rows = _read_hasoc_tsv(os.path.join(HASOC_DIR, "hindi_dataset.tsv"))
    by_label = {}
    for ex in rows:
        by_label.setdefault(ex["label"], []).append(ex)
    train, val, test = [], [], []
    for label, items in sorted(by_label.items()):
        rng = random.Random(seed)
        rng.shuffle(items)
        n = len(items)
        a, b = int(n * ratios[0]), int(n * ratios[0] + n * ratios[1])
        train += items[:a]
        val += items[a:b]
        test += items[b:]
    random.Random(seed).shuffle(train)
    return {"train": train, "val": val, "test": test}


# ---- HateXplain (English, token rationales) ---------------------------------
HX_DIR = os.path.join(DATA, "hatexplain")
# collapse to the same toxic/non-toxic axis as HASOC: hate|offensive -> 1, normal -> 0
HX3 = {"normal": 0, "offensive": 1, "hatespeech": 2}
HX_BIN = {"normal": 0, "offensive": 1, "hatespeech": 1}


def _majority(labels):
    return Counter(labels).most_common(1)[0][0]


def _union_rationale(rationales, n_tokens):
    """OR the per-annotator 0/1 masks into one token-level rationale mask."""
    mask = [0] * n_tokens
    for r in rationales or []:
        for j in range(min(n_tokens, len(r))):
            if r[j]:
                mask[j] = 1
    return mask


def load_hatexplain():
    with open(os.path.join(HX_DIR, "dataset.json"), encoding="utf-8") as f:
        data = json.load(f)
    with open(os.path.join(HX_DIR, "post_id_divisions.json"), encoding="utf-8") as f:
        div = json.load(f)
    by_id = {}
    for pid, v in data.items():
        toks = v.get("post_tokens", [])
        labels = [a["label"] for a in v["annotators"]]
        maj = _majority(labels)
        by_id[pid] = {
            "id": pid,
            "tokens": toks,
            "text": " ".join(toks),
            "label3": HX3[maj],                 # normal/offensive/hate
            "label": HX_BIN[maj],               # toxic vs not (aligns to HASOC)
            "rationale": _union_rationale(v.get("rationales", []), len(toks)),
            "has_rationale": any(sum(r) > 0 for r in v.get("rationales", []) or []),
        }
    splits = {}
    for name, ids in div.items():
        splits[name] = [by_id[i] for i in ids if i in by_id]
    return splits


if __name__ == "__main__":  # self-test
    h = load_hasoc_hindi()
    print("HASOC Hindi:", {k: len(v) for k, v in h.items()},
          "| train label balance:", dict(Counter(x["label"] for x in h["train"])))
    x = load_hatexplain()
    print("HateXplain:", {k: len(v) for k, v in x.items()})
    ex = next(e for e in x["train"] if e["has_rationale"])
    print("  example rationale-bearing post:")
    print("   tokens:", ex["tokens"][:12], "...")
    print("   mask  :", ex["rationale"][:12], "...  label3=", ex["label3"])
    print("  rationale coverage in train:",
          f"{sum(e['has_rationale'] for e in x['train'])}/{len(x['train'])}")
