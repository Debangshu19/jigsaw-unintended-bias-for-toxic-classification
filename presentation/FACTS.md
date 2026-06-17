# CANONICAL FACTS — single source of truth for the Raven presentation

> Every number below was read directly from this repo's code, notebook cell outputs, or
> committed reports, OR from the cited external paper/solution. **Do not invent, round
> beyond what is shown, or state any metric not on this page.** If a needed number is not
> here, write "not measured" — never guess. Always keep the two evaluation regimes
> (RNN-family vs DistilBERT) separate and say they are NOT directly comparable.

---

## 0. The project in one paragraph
Raven is a toxic / hate-speech comment-detection system. It triages comments so a human
moderator only reads the risky ones ("needs review"), instead of reading everything. It ships
as **one model + one API + three clients**: a FastAPI inference service (`raven-api`), a
React website with a live demo (`raven-web`), a Chrome Manifest-V3 extension that flags
comments inline on X / YouTube / Instagram (`extension`), and an Expo mobile shell
(`raven-mobile`). All three clients speak one API contract. The API has a resilient fallback
chain (real model → LLM-via-gateway → transparent keyword heuristic) so a demo never hard-crashes.

## 1. Dataset (English track)
- **Jigsaw "Unintended Bias in Toxicity Classification"** (Civil Comments corpus).
- Total comments: **1,804,871**. Raw class balance: **1,660,537 non-toxic vs 144,334 toxic**
  (~**11.5 : 1**, ~8% toxic) — heavily imbalanced.
- Label: continuous `target` in [0,1]; **binarised at 0.5** (`target ≥ 0.5 → toxic`).
- Identity columns (muslim, black, white, female, …) enable **bias measurement**.
- Two evaluation regimes appear in the repo (KEEP SEPARATE):
  - **Regime A (RNN/classical):** trained/evaluated on the large split; reported with the
    **Jigsaw bias-aware AUC** (a weighted blend of overall AUC + per-identity-subgroup AUCs).
    RNN preprocessing: Keras tokenizer vocab **100,000**, **max_sequence_length 300**,
    pretrained word embeddings of shape **(276,289 × 300)**.
  - **Regime B (DistilBERT):** trained on a **balanced down-sample of 50,000 per class =
    100,000**, stratified **80/20 → 80,000 train / 20,000 val (10k per class)**; reported with
    plain accuracy/F1/ROC-AUC on the balanced 20k val set. Tokenizer = DistilBertTokenizerFast
    (WordPiece, 30,522 vocab), **max_length 128**.

## 2. Regime A — Classical + RNN results (REAL, from `jigsaw-unintended-bias-in-toxic-classification.ipynb`
   and `jigsaw_unintended_bias_toxic_classification_STARTING_UPTO_LOGISTIC.ipynb`)
Metric column "Bias-AUC" = the Jigsaw final bias-aware AUC. "Val acc" = best validation accuracy.

| Model | Architecture detail | Val acc | Bias-AUC |
|---|---|---|---|
| TF-IDF + Logistic Regression (SGD, log-loss, alpha=1e-5) | bag-of-words | 0.87 | **0.9061** |
| Single LSTM | LSTM 128 units, 10 epochs, pretrained 300-d emb | 0.9469 | **0.8927** |
| BiLSTM | 2× BiLSTM 128u + GlobalMaxPool + Dense | 0.8114 | **0.8992** |
| Weighted BiLSTM | same + class weights | 0.8561 | **0.9056** |
| BiGRU + Attention | BiGRU 64u + attention | 0.9517 | **0.9181** |
| BiGRU-Conv1D + Attention | BiGRU 64u + attention + Conv1D | 0.9514 | **0.9190** |

- **Toxic-class recall on the imbalanced set (documented in `RAVEN_REPORT_AND_VIVA.md` §3.6):**
  TF-IDF+LogReg **toxic recall ≈ 0.448**; Single LSTM **toxic recall ≈ 0.452**;
  BiLSTM **toxic recall ≈ 0.914**. (i.e. the high-accuracy simple models *miss more than half of
  toxic comments*; this is the key RNN failure on imbalance.)
- Embeddings are **pretrained, context-free word vectors** (one fixed vector per word).
- ATTRIBUTION: these RNN/classical baselines were **primarily a teammate's contribution
  (Debangshu Dey)**; the DistilBERT transformer + the entire product/engineering stack is the
  author's (Niladri Hazra) track. Use the RNNs as a comparison family; do not claim the author
  trained the LSTMs.

## 3. Regime B — DistilBERT fine-tuned (REAL, from `distillBert_Fine_Tunning_file_nad.ipynb` (1 epoch)
   and `final_distill_bert_eval.ipynb` (3 epoch))
Model: `DistilBertForSequenceClassification` from `distilbert-base-uncased`, num_labels=2.
Architecture: WordPiece tokenizer (30,522 vocab, max_len 128) → embeddings (~23.8M) →
**6 transformer blocks** (12 attention heads, FFN 768→3072→768; ~42.5M) → first-token [CLS]
hidden state (768-d) → pre_classifier Linear(768→768)+ReLU+Dropout(0.2) → classifier Linear(768→2)
→ softmax → P(toxic). **~66.96M total params, 6 layers, 768 hidden dim, 12 heads.**
Training: HuggingFace Trainer, AdamW, lr ≈ 5e-5 linear schedule, per-device batch 16, GPU (Kaggle).

| Metric (balanced 20k val) | 1-epoch (2,500 steps) | 3-epoch (7,500 steps) — saved final model |
|---|---|---|
| Accuracy | 0.8895 | **0.8851** |
| Precision (toxic) | 0.873 | **0.875** |
| Recall (toxic) | 0.912 | **0.899** |
| F1 (toxic) | 0.892 | **0.887** |
| Validation loss | **0.537** | 0.818 |
| Training loss | 0.6075 | 0.4207 |
| ROC-AUC | not reported | **0.9515** |
| PR-AUC (avg precision) | not reported | **0.9495** |

Confusion matrix (3-epoch, rows=actual, cols=pred; 10k per class):
TN **8712** / FP **1288** / FN **1011** / TP **8989**.
(1-epoch confusion: TN 8667 / FP 1333 / FN 878 / TP 9122.)
Sanity probes: "You are stupid" → toxic P=0.999; "I hate you" → toxic P=0.964;
"Have a wonderful day" → non-toxic P=0.9996.
**Overfitting note (honest):** 3 epochs LOWERED training loss but RAISED validation loss
(0.537 → 0.818) with no accuracy gain vs 1 epoch → textbook mild overfitting; ~1 epoch was
near-optimal for this 100k subset.

## 4. "Distillation vs fine-tuning" — the precise truth (do not overstate)
- Knowledge distillation (Hinton et al., 2015): train a small *student* to imitate a large
  *teacher*'s soft logits.
- DistilBERT (Sanh et al., HuggingFace, 2019) is the *result* of distilling BERT-base
  (~110M, 12 layers) into ~66M, 6 layers — **40% smaller, ~60% faster, keeps ~97%** of BERT's
  GLUE performance.
- **What we did: we took the ALREADY-distilled DistilBERT and FINE-TUNED it on Jigsaw.** We did
  NOT run a teacher→student distillation loop ourselves. Never say "I distilled BERT." Correct
  phrasing: "I used a distilled backbone for efficiency and fine-tuned it."

## 5. The "bigger model" step ("DistilBERT large" as the user phrased it)
- There is **no measured larger-model run in this repo.** DistilBERT only exists as a base
  (66M) model; there is no official "DistilBERT-large".
- The HONEST framing of this step: the natural way to raise the capacity ceiling above
  distilbert-base is a **larger transformer** — e.g. BERT-base (110M, 12 layers) or BERT-large
  (~340M, 24 layers, 16 heads, 1024 hidden) — i.e. *undo the compression* and use a full-size
  encoder. Present this as REASONING + literature, explicitly labelled "not run by us", never as
  our own measured number.
- Bias-variance reasoning: a larger model lowers approximation error (bias) but raises variance
  / overfitting risk on small data and costs more compute; a *single* larger model raises the
  ceiling but does not remove single-model variance or correlated errors.

## 6. The ensemble / "simple averaging" step (what we are bringing in now)
EXTERNAL GROUNDING (cite as literature, not our measurement):
- **Qishen Ha — 8th place, Jigsaw Unintended Bias in Toxicity Classification (Kaggle, 2019):**
  title "8th Place Solution (4 models simple avg)" → **four diverse models combined by SIMPLE
  AVERAGING of predicted probabilities.** (We have the public title + competition-wide writeups;
  we do NOT have the four exact architectures verbatim, so say "four diverse models" and do not
  invent their names.)
- Competition consensus (from public solution writeups, e.g. Ceshine Lee's notes):
  fine-tuned **BERT was the best *single* model**; top teams **ensembled diverse models**
  (BERT-base, BERT-large, GPT-2, and LSTM-based models) by averaging / weighted-averaging
  (e.g. 2nd place "power 3.5 weighted sum"). Bias-metric tricks: **sample weighting** (up-weight
  toxic + identity examples; target weight × log(annotator_count+2)) and **negative
  down-sampling** (drop 50% negatives after epoch 1). No top team relied on a single model.
- **Multi-sample dropout (Inoue, 2019, arXiv:1905.09788):** within ONE training iteration, draw
  *k* dropout masks, push the input through each (duplicated layers share weights), and **average
  the per-sample losses** → faster convergence and better generalization at minimal extra cost
  (validated on ImageNet / CIFAR-10 / CIFAR-100). Frame it as an *intra-model* "mini-ensemble".
  We do NOT have the paper's exact error-reduction percentages — describe the mechanism, do not
  quote unverified numbers.

ENSEMBLE MATH (this is standard theory — state it as theory, derived, not measured):
- Bias–variance: for K i.i.d. models each with variance σ², the simple average has the SAME bias
  but variance **σ²/K**.
- With pairwise correlation ρ between model errors, variance of the average =
  **ρσ² + (1−ρ)σ²/K**. As K→∞ it tends to ρσ². ⇒ the gain comes from **low correlation
  (diversity)**, which is why diverse architectures (transformer + RNN + larger transformer) are
  averaged rather than copies of one model.
- Ambiguity decomposition (Krogh & Vedelsby, 1995): ensemble error =
  (average individual error) − (average ambiguity/diversity) ⇒ diversity strictly reduces error
  for convex losses (also follows from Jensen's inequality).
- Our concrete plan: average the probabilities of our **DistilBERT + a larger transformer + an
  RNN-family model** (we already have the RNN family and DistilBERT). Present the EXPECTED benefit
  from this theory + the proven Kaggle result; clearly label it as the proposed next step, using
  our REAL single-model numbers as the starting point. Do NOT report a fabricated ensemble score.

## 7. RAVEN-X — the original contribution (Hindi, cross-script rationale transfer) — REAL
   (from `RAVEN_X_METHOD_DESIGN.md`, `RAVEN_X_RESULTS.md`, executed notebook)
- Motivation: supervisor Dr. Arpita Dutta is 2nd author of the survey "A Survey on Automatic
  Online Hate Speech Detection in Low-Resource Languages" (arXiv:2411.19017); she asked the team
  to propose their OWN method in low-resource / code-mixed Hindi-English, not reproduce an English
  Jigsaw model.
- Data truth (measured on disk, HASOC-2019 Hindi, 4,665 rows): **82.5% of script characters are
  Devanagari**; 74.9% of posts are Devanagari-dominant, 7.3% Latin-dominant → it is **native-script
  Hindi, not romanized Hinglish.** A Devanagari slur and its English equivalent share **zero
  subword tokens**, so any cross-script transfer is **purely representational** (through MuRIL's
  shared meaning space). Official gold test is **blind** → we use a **frozen stratified seed=42
  split: train 3,265 / val 699 / test 701.**
- Method: ONE shared **MuRIL-base** encoder (12 layers, 768-d, ~236M params) + two heads —
  CLASSIFY (toxic/not, supervised by Hindi HASOC + English) and RATIONALE (which words are toxic,
  768→256→1 per word, supervised ONLY on English HateXplain) — trained with a **per-head-masked
  multi-task loss** (each loss term applies only where its label exists → never trains on a
  missing label) plus a **label-free faithfulness self-objective** on Hindi (comprehensiveness /
  sufficiency via attention-level masking + full-encoder re-run).
- Mixed batch of 16 = 8 HASOC-Hindi + 8 HateXplain-English. AdamW, discriminative LR 2e-5
  encoder / 1e-4 heads, max_len 128, fp16, T4 GPU.
- REAL first-run results (Colab T4, `--fast --epochs 3`, seed 42):
  - TF-IDF(word+char)+LogReg baseline: macro-F1 **0.787 val / 0.813 test**.
  - **MuRIL (RAVEN-X), epoch 2 (faithfulness off): macro-F1 0.842 val** — best classification,
    **+5.5 pts over the baseline** on the same val split.
  - English rationale token-F1 (HateXplain): **0.653 → 0.679** across training.
  - Faithfulness trade-off: turning the Hindi faithfulness objective ON (epoch 3) traded
    **0.842 → 0.802 macro-F1** for **token-F1 0.653 → 0.679** — the measured accuracy-vs-
    explainability tension the method pre-registered.
- Honest caveats: fast single-seed run; MuRIL val capped at 400 examples by the eval loop; the
  cross-script faithfulness *transfer* to Devanagari is the pending Week-4 headline, not yet
  measured; clean test-set MuRIL number pending.
- IMPORTANT honesty rule: **the English DistilBERT 0.885 is NOT a baseline for RAVEN-X** — different
  language, task, label scheme, and class balance (a "category error"). It is only *motivation*.

## 8. Tone & writing rules (apply everywhere)
- Plain, humble, factual — a diligent student explaining their work to a teacher. Short sentences.
- NO marketing / AI-buzzwords: avoid "delve, leverage, robust, seamless, cutting-edge, harness,
  unlock, game-changer, powerful, revolutionary, elevate, realm, tapestry."
- Say what is **measured by us** vs **established in the literature** vs **proposed next step** —
  explicitly, every time.
- Never claim we distilled BERT; never compare DistilBERT-English to MuRIL-Hindi as baselines.
- British/Indian spelling is fine. Numbers exactly as on this page.

## 9. References (use these; do not invent others)
- Jigsaw Unintended Bias in Toxicity Classification, Kaggle, 2019.
- Qishen Ha, "8th Place Solution (4 models simple avg)", Jigsaw competition writeup, Kaggle, 2019.
- H. Inoue, "Multi-Sample Dropout for Accelerated Training and Better Generalization",
  arXiv:1905.09788, 2019.
- V. Sanh et al., "DistilBERT, a distilled version of BERT", 2019.
- A. Vaswani et al., "Attention Is All You Need", 2017.
- G. Hinton, O. Vinyals, J. Dean, "Distilling the Knowledge in a Neural Network", 2015.
- S. Hochreiter, J. Schmidhuber, "Long Short-Term Memory", 1997.
- A. Krogh, J. Vedelsby, "Neural Network Ensembles, Cross Validation, and Active Learning", 1995.
- S. Khanuja et al., "MuRIL: Multilingual Representations for Indian Languages", 2021.
- B. Mathew et al., "HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection", 2021.
- S. Das, A. Dutta, et al., "A Survey on Automatic Online Hate Speech Detection in Low-Resource
  Languages", arXiv:2411.19017, 2024.
</content>
</invoke>
