"""RAVEN-X — exploratory data audit for the two corpora.

Verifies (does NOT assume) the structure & label distributions of:
  1. HASOC-2019 Hindi (code-mixed, post-level labels, the spine)
  2. HateXplain (English, token-level rationales, the explanation donor)

Run:  raven-api/.venv/bin/python raven-codemixed/eda.py
Reads only local gitignored data; prints stats. No network, no training.
"""
import csv, json, os, statistics
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def hr(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def hasoc(path, name):
    rows, malformed = [], 0
    with open(path, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        header = next(r)
        for row in r:
            if len(row) < 5 or row[2] not in ("NOT", "HOF"):
                malformed += 1
                continue
            rows.append(row)
    print(f"\n[{name}] header={header}")
    if len(header) < 3:
        print(f"  >> BLIND test set (no labels): {malformed} text-only rows. "
              f"Official 2019 gold labels are not public in this mirror.")
        return rows
    print(f"  usable rows: {len(rows)}   malformed/skipped: {malformed}")
    for i, task in enumerate(("task_1 (HOF/NOT)", "task_2 (type)", "task_3 (target)"), start=2):
        print(f"  {task:22s}: {dict(Counter(x[i] for x in rows))}")
    lens = [len(x[1].split()) for x in rows]
    print(f"  text length (words): mean={statistics.mean(lens):.1f} max={max(lens)} "
          f"p95~{sorted(lens)[int(len(lens) * 0.95)]}")
    return rows


def stratified_split(rows, label_idx=2, seed=42, ratios=(0.70, 0.15, 0.15)):
    """Reproducible stratified split (stdlib only) since the official gold test is blind."""
    import random
    by_label = {}
    for row in rows:
        by_label.setdefault(row[label_idx], []).append(row)
    tr, va, te = [], [], []
    for label, items in sorted(by_label.items()):
        rng = random.Random(seed)
        rng.shuffle(items)
        n = len(items)
        n_tr, n_va = int(n * ratios[0]), int(n * ratios[1])
        tr += items[:n_tr]
        va += items[n_tr:n_tr + n_va]
        te += items[n_tr + n_va:]
    print(f"\n  STRATIFIED split (seed={seed}, {ratios}):")
    for nm, sp in (("train", tr), ("val", va), ("test", te)):
        print(f"    {nm:5s} n={len(sp):4d}  HOF/NOT={dict(Counter(x[2] for x in sp))}")
    return tr, va, te


def hatexplain():
    p = os.path.join(DATA, "hatexplain", "dataset.json")
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    print(f"  posts: {len(d)}")
    sample_key = next(iter(d))
    sample = d[sample_key]
    print(f"  record keys: {list(sample.keys())}")
    print(f"  annotators/post field example: {sample['annotators']}")
    # majority label
    maj, rat_token_frac, rationale_present, ntoks = Counter(), [], 0, []
    for v in d.values():
        labels = [a["label"] for a in v["annotators"]]
        m = Counter(labels).most_common(1)[0][0]
        maj[m] += 1
        toks = v.get("post_tokens", [])
        ntoks.append(len(toks))
        rats = v.get("rationales", [])
        if rats and any(sum(r) > 0 for r in rats):
            rationale_present += 1
            # union mask over annotators
            L = len(toks)
            union = [0] * L
            for r in rats:
                for j in range(min(L, len(r))):
                    union[j] = union[j] or r[j]
            if L:
                rat_token_frac.append(sum(union) / L)
    print(f"  majority label dist: {dict(maj)}")
    print(f"  posts with >=1 non-empty rationale: {rationale_present} ({100*rationale_present/len(d):.1f}%)")
    if rat_token_frac:
        print(f"  fraction of tokens marked rationale (union): mean={statistics.mean(rat_token_frac):.3f}")
    print(f"  tokens/post: mean={statistics.mean(ntoks):.1f} max={max(ntoks)}")
    # divisions
    dp = os.path.join(DATA, "hatexplain", "post_id_divisions.json")
    with open(dp, encoding="utf-8") as f:
        div = json.load(f)
    print(f"  splits: " + ", ".join(f"{k}={len(v)}" for k, v in div.items()))


if __name__ == "__main__":
    hr("HASOC-2019 Hindi (code-mixed) — the spine")
    base = os.path.join(DATA, "hasoc2019")
    labeled = hasoc(os.path.join(base, "hindi_dataset.tsv"), "hindi labeled set")
    hasoc(os.path.join(base, "hasoc2019_hi_test.tsv"), "hindi official test")
    stratified_split(labeled)

    hr("HateXplain (English) — token-rationale donor")
    hatexplain()

    hr("DONE — both corpora verified locally")
