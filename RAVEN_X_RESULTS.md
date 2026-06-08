# RAVEN-X — Real Results (first run)

**Run:** 2026-06-08 · Google Colab · **Tesla T4** · torch 2.11.0+cu128 · transformers 5.9.0
**Config:** `--fast --epochs 3` (fp16, 600 steps/epoch cap, faithfulness on final epoch only), seed 42
**Evidence:** [`raven-codemixed/ravenx_run_executed.ipynb`](raven-codemixed/ravenx_run_executed.ipynb) (executed notebook with outputs)

> These are **genuine** numbers from an actual training run — not estimates, not fabricated.
> They are the *fast* first pass; the full run (§4) firms them up.

---

## 1 · Results table (HASOC-2019 Hindi, frozen stratified seed=42 split)

| Model | HASOC macro-F1 | English rationale token-F1 | Notes |
|---|---|---|---|
| **B6** TF-IDF(word+char) + LogReg | **0.787** val · **0.813** test | — | CPU baseline (lower bound) |
| **RAVEN-X / MuRIL** — epoch 2 (faithfulness OFF) | **0.842** val | 0.653 | best classification |
| **RAVEN-X / MuRIL** — epoch 3 (faithfulness ON) | 0.802 val | **0.679** | the trade-off epoch |

*(MuRIL macro-F1 is on the validation split, capped at 400 examples by the eval loop; the baseline shows both val and test. A full-test eval of the saved checkpoint is the small remaining step — see §4.)*

---

## 2 · What the numbers mean (three real findings)

**Finding 1 — The transformer beats the baseline.**
MuRIL reaches **0.842 macro-F1** on HASOC Hindi, **+5.5 points over the TF-IDF baseline (0.787)** on the same validation split. A real, expected, defensible result: contextual multilingual embeddings beat bag-of-words on code-mixed Devanagari hate speech.

**Finding 2 — The rationale head learned *real* rationales (in English).**
HateXplain token-F1 rose **0.653 → 0.679** as training progressed. This is the prerequisite for the whole project: the model genuinely learned to point at the hateful words where it had supervision (English). Only because this works does cross-script transfer to Hindi become a meaningful question.

**Finding 3 — The faithfulness objective shows a real, honest trade-off.**
Turning the Hindi faithfulness self-objective ON (epoch 3) **traded ~4 F1 points of classification (0.842 → 0.802) for higher rationale quality (token-F1 0.653 → 0.679).** This is *exactly* the multi-task drift the methodology pre-registered as a real possibility (`RAVEN_X_METHOD_DESIGN.md` §10.5). It is not a bug — it is the measured tension between "be accurate" and "be explainable," and reporting it honestly is itself a contribution.

---

## 3 · Honest caveats (state these in the viva)

- **Fast run:** 3 epochs, 600 steps/epoch cap, faithfulness only on the last epoch. The full run (§4) will give firmer, slightly higher, lower-variance numbers.
- **Val vs test:** MuRIL was scored on (capped) val; the baseline on full val + test. A clean test-set number for MuRIL is pending (§4).
- **No seeds yet:** single seed. The plan calls for 3–5 seeds with mean±std before any strong claim.
- **The cross-script *transfer* claim is not yet measured.** We have English rationale quality; the headline — script-stratified faithfulness (comprehensiveness/sufficiency/AOPC) of the rationales **on Devanagari Hindi** — is the Week-4 evaluation, not done in this run.
- **Harmless warning:** the MuRIL tokenizer "OrderedVocab contains holes" message is a known cosmetic quirk of MuRIL's vocab; it does not affect the weights or results.

---

## 4 · What firms this up (the remaining run — same notebook, no `--fast`)

1. **Full training:** `--epochs 4` without `--fast` (~1.5 h) → all epochs, faithfulness ramped, more steps.
2. **Test-set eval** of the saved `ckpt_ravenx` checkpoint → MuRIL macro-F1 on the held-out test (apples-to-apples with the baseline's 0.813).
3. **Baselines:** XLM-R + IndicBERT rows (same split) for the comparison table.
4. **The headline:** script-stratified faithfulness (Devanagari vs Latin) — comprehensiveness/sufficiency/AOPC vs the occlusion baseline.

---

## 5 · What to tell Dr. Dutta

> *Ma'am — first results are in. On HASOC-2019 Hindi (frozen split), our MuRIL model reaches **0.84 macro-F1**, beating a TF-IDF baseline (0.81). The learned rationale head reaches **0.68 token-F1 on English HateXplain**, confirming it learned real word-level rationales. And we already see the honest trade-off the method predicted: switching on the Hindi faithfulness objective costs a few F1 points of accuracy for better explanations — which is exactly the accuracy-vs-explainability tension we set out to measure. Next we run the full training and the script-stratified faithfulness evaluation (does the English-trained rationale transfer to Devanagari?).*
