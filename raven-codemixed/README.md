# RAVEN-X — Cross-Script Rationale Transfer for Hindi Hate Speech

The research core of Raven's final-year pivot: an **explainable Devanagari-Hindi hate-speech
model** whose original contribution is **cross-script rationale transfer** — learn word-level
rationale supervision on **English** HateXplain and transfer it through a shared **MuRIL**
encoder to produce *faithful* explanations on **Hindi**, where no rationale labels exist.

> Full methodology: [`../RAVEN_X_METHOD_DESIGN.md`](../RAVEN_X_METHOD_DESIGN.md)
> Gap analysis & rationale: [`../RAVEN_IMPROVEMENT_REPORT.md`](../RAVEN_IMPROVEMENT_REPORT.md)

## What's here (all verified locally)

| File | Purpose | Verified |
|---|---|---|
| `eda.py` | Audit both corpora (label/script distributions, blind-test detection) | ✅ run |
| `data_loaders.py` | HASOC-2019 Hindi (seeded stratified split) + HateXplain (token rationales) | ✅ self-test |
| `raven_x.py` | Model (MuRIL + classification + rationale heads) + masked multi-corpus loss + label-free faithfulness objective | ✅ `--smoke` |
| `train_raven_x.py` | Two-stage trainer (freeze→unfreeze, γ-ramp), macro-F1 / token-F1 eval, checkpoint | ✅ `--smoke` |
| `data/` | HASOC-2019 + HateXplain (**git-ignored — research-only, never commit**) | downloaded |

## Verified data facts (don't re-assume)

- HASOC-2019 Hindi: **4,665 labeled rows**, balanced (HOF 2,469 / NOT 2,196).
- It is **~75% Devanagari-dominant / 82.5% Devanagari characters**, only ~7% Latin → **native-script Hindi, not romanized Hinglish.**
- Official HASOC gold test is **blind** (no labels) → we use a **frozen seed=42 stratified split** (train 3,265 / val 699 / test 701).
- HateXplain: 20,148 posts, **9,130 train posts carry token rationales** (English-Latin).

## Run it

**No GPU? Use the free one.** Upload [`ravenx_colab.ipynb`](ravenx_colab.ipynb) to
[Google Colab](https://colab.research.google.com) or Kaggle, set **Runtime → T4 GPU**, and
**Run all**. It is self-contained (writes its own code, stages its own data) and trains the
**real MuRIL model** in ~1.5–2 h — genuine, screenshot-able numbers, ₹0.

Local pipeline checks (CPU, seconds, uses the real MuRIL tokenizer + a tiny random encoder):
```bash
../raven-api/.venv/bin/python eda.py            # audit corpora
../raven-api/.venv/bin/python data_loaders.py   # loader self-test
../raven-api/.venv/bin/python raven_x.py --smoke # model + loss + faithfulness grad-flow
../raven-api/.venv/bin/python train_raven_x.py --smoke  # full trainer end-to-end
```

Real training (Kaggle / Colab **T4**, ~1.5–2 h/seed):
```bash
pip install -U transformers torch accelerate
# stage the data (research-only; do not redistribute):
#   git clone --depth 1 https://github.com/TharinduDR/HASOC-2019  -> copy data/hasoc2019/*.tsv
#   curl HateXplain Data/dataset.json + post_id_divisions.json   -> data/hatexplain/
python train_raven_x.py --model google/muril-base-cased --epochs 4 --out ckpt_ravenx
```
The checkpoint (`ckpt_ravenx/encoder/`, `heads.pt`, `raven_x.json`) is what the Raven API will
serve via a new `RavenXHeads` load path (Week-5 integration; see method doc §3 serving note).

## Status / next (see repo task list)

Done & verified: data layer, method design, model, trainer (all smoke-tested).
Next: run the real MuRIL training on T4 · classification baselines (MuRIL/XLM-R/IndicBERT/CNERG/L3Cube on the frozen split) · Week-1 geometry probe · faithfulness battery (comprehensiveness/sufficiency/AOPC, script-stratified) · 200-post Devanagari rationale annotation · serving integration.

## Ethics / licensing

HASOC and HateXplain are **research-use-only** and contain real slurs. Train and report metrics;
**ship weights + split indices + a model card, never the raw data.** Cite Mandl et al. 2019 (HASOC),
Mathew et al. 2021 (HateXplain), MuRIL (Khanuja et al. 2021).
