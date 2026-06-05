# Raven — Final-Year Project Report & Viva Preparation

**Author:** Niladri Hazra
**Project:** Raven — an end-to-end toxic / hate-speech comment detection system
**Repo components:** `raven-model` + notebooks (ML), `raven-api` (FastAPI inference), `raven-web` (website + live demo), `extension` (Chrome MV3), `raven-mobile` (Expo app)

> This document is written for **you** to study before the viva. Every number and claim
> in it comes from the actual code and notebook outputs in this repo. There is **no
> exaggeration** — where the work is "fine-tuning, not distillation," it says so plainly,
> and gives you the *correct, still-impressive* way to describe it so you are never caught out.

---

## Part 0 — The 30-second honest pitch (memorise this)

> "Raven is a toxic-comment detection system. The core model is **DistilBERT — a compressed,
> knowledge-distilled version of BERT — which I fine-tuned on the Jigsaw toxicity dataset** to
> classify a comment as *toxic / needs-review* or *safe*. It reaches **~0.885 accuracy and 0.95 ROC-AUC**
> on a balanced 20k validation set. I then wrapped that model in a **FastAPI inference service** and
> built **three front-ends on top of one API contract** — a React website with a live demo, a
> Chrome Manifest-V3 extension that flags toxic comments inline on X / YouTube / Instagram, and an
> Expo mobile app. The whole system has a graceful fallback chain so the demo never crashes."

That sentence is 100% defensible. Nothing in it is a lie.

---

## Part 1 — The single most important thing: "distillation" vs "fine-tuning"

You keep using the word **"distill"**. An ML professor *will* probe this, so get the framing exactly right. Here is the truth, and the truth is still impressive.

### What knowledge distillation actually is
Knowledge distillation (Hinton et al., 2015) is a **training technique** where a small *student*
model is trained to imitate the soft output probabilities (logits) of a large *teacher* model, so
the student keeps most of the teacher's accuracy at a fraction of the size/cost.

### What DistilBERT is
**DistilBERT (Sanh et al., HuggingFace, 2019)** is the *result* of doing exactly that to BERT:
- Teacher: BERT-base, ~110M parameters.
- Student: DistilBERT, **~66M parameters** — **40% smaller, ~60% faster, keeps ~97%** of BERT's language understanding (GLUE).

### What **you** actually did
You took the **already-distilled** DistilBERT and **fine-tuned** it (supervised transfer learning)
on the Jigsaw toxicity data. **You did not run a teacher→student distillation loop yourself.**

### So how do you talk about it? (say exactly this)
> "I used a **distilled** model — DistilBERT — *because* it is small and fast, and I **fine-tuned**
> it on toxic-comment data. The distillation was done by the original authors; my contribution is
> **selecting the distilled backbone, fine-tuning it for toxicity, and serving it end-to-end.**
> I leveraged distillation for efficiency; I did not train a student against a teacher myself."

That is honest, it shows you *understand* what distillation is, and it explains *why you chose a
distilled model* (deployability — small enough to serve on a college budget and even run in a
browser later). **Never say "I distilled BERT into a smaller model" — that would be false.**

If a professor asks "could you actually distill it yourself?" → see Part 10 (you *can*, and the
path is short — that turns this from a weak spot into a strong forward-looking answer).

---

## Part 2 — What Raven is (the product)

Raven answers one question for a moderator or platform: **"Which of these comments actually need a
human to look at them?"** Instead of asking a person to read everything, Raven scores each comment
and surfaces only the risky ones.

It ships as **one model + one API + three clients**:

| Surface | What it does | Tech |
|---|---|---|
| `raven-api` | Loads the model, exposes `/predict` & `/predict-batch` | FastAPI + PyTorch + HuggingFace Transformers |
| `raven-web` | Landing page + "Raven Lab" live demo (paste comments → scored) | Vite + React + plain CSS |
| `extension` | Auto-flags toxic comments **inline** on X / YouTube / Instagram | Chrome Manifest V3 (content script + service worker) |
| `raven-mobile` | Paste/scan comments → review-queue UI on phone | Expo / React Native |

All three clients speak the **same API contract**, so the model only has to exist in one place.

---

## Part 3 — The ML core (this is the heart of your viva)

### 3.1 Dataset
- **Jigsaw "Unintended Bias in Toxicity Classification"** (Civil Comments corpus), ~**1.8M** comments.
- Each comment has a continuous `target` toxicity score in **[0, 1]** (fraction of human raters who
  found it toxic), plus fine-grained columns (`obscene`, `threat`, `insult`, `identity_attack`, …)
  and identity columns (`muslim`, `black`, `white`, `female`, …) used for **bias measurement**.

### 3.2 Preprocessing (exact steps from the notebook)
1. Keep `comment_text` + `target`.
2. **Binarise**: `label = 1 (toxic) if target ≥ 0.5 else 0 (non-toxic)`.
3. The full data is **heavily imbalanced**: **1,660,537 non-toxic vs 144,334 toxic** (~**11.5 : 1**, ~8% toxic).
4. **Balance by down-sampling**: take **50,000 per class → 100,000** balanced comments
   (`groupby('target').sample(50000, random_state=42)`).
5. **Stratified 80/20 split** → **80,000 train / 20,000 validation** (10k per class in val).
6. **Tokenise** with `DistilBertTokenizerFast` (WordPiece, 30,522 vocab), `max_length=128`,
   truncation + padding.

> **Why balancing matters (good viva point):** on the raw 92/8 split, a model that predicts
> "non-toxic" for everything already scores ~92% accuracy. Reporting 88.5% on a **balanced** set is
> a *harder, more honest* number than 94% on the imbalanced set.

### 3.3 Architecture — DistilBERT for sequence classification (~67M params)
Loaded as `DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)`.

```
input text
  → WordPiece tokenizer (max_len 128) → input_ids + attention_mask
  → Embeddings (word 30522×768 + positional 512×768 + LayerNorm)      ~23.8M params
  → 6 × Transformer block (multi-head self-attention, 12 heads, FFN 768→3072→768)  ~42.5M params
  → take first-token ([CLS]) hidden state (768-d)
  → pre_classifier  Linear(768→768) + ReLU + Dropout(0.2)             ~0.59M params
  → classifier      Linear(768→2)  →  2 logits                        ~1.5K params
  → softmax → P(toxic)
  → label = "review" if P(toxic) ≥ 0.5 else "safe"
```
- **6 transformer layers** (BERT-base has 12 — that's the distillation saving), **768 hidden dim**,
  **12 attention heads**, **~66.96M total params**.
- The DistilBERT backbone is pretrained; the **classification head is newly initialised** and learned
  during fine-tuning (the notebook's load report shows `classifier.*` as MISSING = freshly created).

### 3.4 Training setup
- **HuggingFace `Trainer`**, optimiser **AdamW** with Trainer defaults (lr ≈ **5e-5**, linear
  schedule), **per-device batch size 16**, `max_length=128`.
- Ran on **GPU** (CUDA, Kaggle; PyTorch 2.10, Transformers 5.0).
- Two runs in the repo:
  - `distillBert_Fine_Tunning_file_nad.ipynb` → **1 epoch** (~10 min, 2,500 steps).
  - `final_distill_bert_eval.ipynb` → **3 epochs** (~30 min, 7,500 steps) — the **final** model,
    saved as `distilbert_toxicity_model` (and `trainer.save_model` + `tokenizer.save_pretrained`).
- A clean, parameterised re-implementation also exists as a script: **`raven-model/train_distilbert.py`**
  (argparse CLI, stratified split, `compute_metrics`, saves `eval_metrics.json`).

### 3.5 Results — the numbers to memorise

**Final 3-epoch DistilBERT, on the 20,000-comment balanced validation set:**

| Metric | Value |
|---|---|
| Accuracy | **0.885** |
| Precision (toxic) | **0.875** |
| Recall (toxic) | **0.899** |
| F1 (toxic) | **0.887** |
| **ROC-AUC** | **0.9515** |
| PR-AUC (avg precision) | **0.9495** |

**Confusion matrix** (rows = actual, cols = predicted; 10k per class):

```
                 pred non-toxic   pred toxic
actual non-toxic     8712 (TN)     1288 (FP)
actual toxic         1011 (FN)     8989 (TP)
```

Sanity checks on raw text:
- `"You are stupid"` → toxic, P=0.999
- `"I hate you"` → toxic, P=0.964
- `"Have a wonderful day"` → non-toxic, P=0.9996

**The 1-epoch run** scored slightly *higher* on val (acc 0.889, F1 0.892, recall 0.912) with a
**much lower validation loss (0.537 vs 0.818)**.

> **Sophisticated honesty point — overfitting:** training longer (3 epochs) *lowered training loss*
> but *raised validation loss* and did **not** improve validation accuracy. That is textbook **mild
> overfitting**. The right takeaway (say this): *"For this balanced 100k subset, ~1 epoch was already
> near-optimal; extra epochs mostly memorised the training set. With more data or early-stopping on
> val loss I'd pick the 1-epoch checkpoint."* Saying this shows you read your own curves.

### 3.6 Baselines (team context — be honest about who did what)
The repo also contains **classical and RNN baselines** (TF-IDF + Logistic Regression, LSTM, BiLSTM,
weighted BiLSTM, GRU-Conv1D with attention). **These deep-learning baselines were primarily a
teammate's contribution (Debangshu Dey);** your model track is the **DistilBERT transformer** plus
the **entire product/engineering stack**. Use the baselines as a comparison family, but don't claim
you trained the LSTMs.

Indicative baseline numbers (evaluated on the **full ~361k imbalanced** validation set with the
Jigsaw **bias-aware AUC** metric):

| Model | Accuracy | Toxic recall | Final bias metric |
|---|---|---|---|
| TF-IDF + Logistic Regression | 0.942 | 0.448 | 0.868 |
| Single LSTM | 0.947 | 0.452 | 0.893 |
| BiLSTM | 0.811 | 0.914 | 0.899 |

> **Critical fairness caveat (say this before anyone catches it):** the baselines were measured on
> the **imbalanced 361k** set, the DistilBERT on a **balanced 20k** set — so accuracies are **not
> directly comparable**. On the imbalanced set, high accuracy with **low toxic recall** (LogReg
> catches <45% of toxic comments) is exactly the failure transformers fix: DistilBERT gets **~90%
> recall on toxic** while staying balanced. The transformer's win is **recall on the minority class**,
> which is the class that actually matters for moderation.

---

## Part 4 — From notebook to product: the inference API (`raven-api`)

This is where "I trained a model" becomes "I shipped a system." It's a **FastAPI** service
(`app.py`) backed by a **resilient model loader** (`model.py`).

### 4.1 Endpoints (Pydantic-validated)
- `GET /health` → `{ ok, source, model_ref, fallback_source }`
- `GET /metadata` → name, source, threshold, endpoint list
- `POST /predict` → `{ text }` → `{ label, score, needs_review, source }`
- `POST /predict-batch` → `{ texts: [...] }` (1–100) → `{ predictions: [...] }`
- Inputs validated: `text` 1–5000 chars, batch ≤ 100. **CORS open** so the website/extension can call it.

### 4.2 The resilient model chain (a genuinely strong engineering story)
`load_model()` builds a **primary + fallback** cascade (`ResilientModel`): it tries the real model,
and on *any* exception transparently serves a fallback — so a demo never hard-crashes.

```
Primary (best available):
  TransformersModel  ← your fine-tuned DistilBERT
     • RAVEN_MODEL_DIR  → load your local exported checkpoint  (source: "raven-local-model")
     • RAVEN_MODEL_ID   → pull a HF model, e.g. unitary/toxic-bert (source: "raven-hf-model")
Fallback (if primary errors / not configured):
  AiGatewayFallbackModel  → an LLM classifier via Vercel AI Gateway, strict JSON, temp 0
  DemoFallbackModel       → transparent keyword heuristic (source ends in "fallback")
```

How `TransformersModel` scores a comment: tokenise (max_len 128) → forward pass under
`torch.no_grad()` → for a 2-class head, **softmax → P(class 1) = toxic score**; it also handles
multi-label heads (sigmoid + max over toxic-labelled indices) so it works with off-the-shelf models
like `unitary/toxic-bert`. Threshold (default **0.5**, `RAVEN_THRESHOLD`) maps score → `review`/`safe`.

### 4.3 Honesty built into the product (use this — it impresses)
`docs/raven-plan.md` states the rule explicitly: **only call it the "Raven model" when serving a
Raven-owned fine-tuned model**; if an external API is used as a fallback, don't misrepresent it as
proprietary. The UI maps internal `source` strings to friendly names (`Raven engine` / `Raven
fallback` / `Demo fallback`) but the **real source is always returned by `/health` and shown**.

> **Current-state honesty (important):** the trained checkpoint is **not committed** (`models/` is
> git-ignored; it lives on Kaggle/Drive). So out-of-the-box the API runs the **demo fallback** until
> you set `RAVEN_MODEL_DIR` to your exported DistilBERT folder (or `RAVEN_MODEL_ID=unitary/toxic-bert`
> as a stronger ready-made stand-in). **For the live viva demo, export your checkpoint and point
> `RAVEN_MODEL_DIR` at it** so you're demoing *your* model, not the heuristic.

---

## Part 5 — The website (`raven-web`)
- **Vite + React + plain CSS** (no Tailwind/component libs), a marketing landing page (hero,
  how-it-works, features) plus a **"Raven Lab"** playground.
- `ravenClient.js` is the demo brain: `scoreWithApi()` POSTs lines to `/predict-batch`; if the API
  is down, `scoreText()` provides a **browser-side keyword fallback** that **labels itself
  `browser-demo-fallback`** in the UI — so the demo always works *and* never pretends a model is
  running when it isn't.
- One comment per line → batch scored → result panel shows the verdict **and the source**.

## Part 6 — The browser extension (`extension`)
- **Manifest V3**, content script (`content.js`) + background **service worker** + popup.
- On **X/Twitter, YouTube, Instagram**, it finds comment/tweet DOM nodes
  (`article[data-testid="tweet"]`, `ytd-comment-view-model`, …), batches their text (25/batch, cap 40),
  sends to the API via the service worker, and **attaches an inline pill** — *Safe / Borderline /
  Toxic* + **% toxic** — next to each comment. Toxic nodes get a highlight.
- Engineering details worth name-dropping:
  - **`MutationObserver` + scroll listener** with debounce → re-scans single-page-app feeds as you
    scroll (X/YouTube never do full page loads).
  - **`WeakMap` prediction cache per node** → React re-renders re-attach the pill without re-calling
    the API.
  - **Built with DOM APIs (no `innerHTML`)** specifically to survive **Trusted-Types CSP** on x.com.
  - Same **API-down keyword fallback** so it degrades gracefully.

## Part 7 — The mobile app (`raven-mobile`)
- **Expo / React Native**, same `/predict-batch` contract (`EXPO_PUBLIC_RAVEN_API_URL`), paste/scan
  comments → **review-queue** UI, same offline fallback approach. It's an app **shell/scaffold** —
  describe it as "the same engine reached from a 4th surface," not a finished store app.

---

## Part 8 — System architecture (one picture in words)

```
                       ┌───────────────────────────────┐
   Jigsaw CSV  ──train──►  DistilBERT fine-tune (notebook / train_distilbert.py)
                       └───────────────┬───────────────┘
                                       │ export checkpoint
                                       ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  raven-api  (FastAPI)                                          │
   │   load_model(): DistilBERT ─► AI-Gateway LLM ─► keyword demo   │
   │   POST /predict  ·  POST /predict-batch  ·  GET /health        │
   └───────────────┬───────────────┬───────────────┬──────────────┘
                   │               │               │
            ┌──────▼─────┐  ┌──────▼──────┐  ┌──────▼──────┐
            │ raven-web  │  │  extension  │  │ raven-mobile│
            │ React demo │  │  MV3 inline │  │ Expo queue  │
            └────────────┘  └─────────────┘  └─────────────┘
```

---

## Part 9 — Limitations & ethics (have these ready — examiners love them)
1. **Trained on a 100k balanced subset**, not all 1.8M — chosen for compute budget; more data would help.
2. **English-only**, `distilbert-base-uncased` → loses case (SHOUTING), weak on code-switching, slang, emojis, sarcasm.
3. **Label noise**: `target ≥ 0.5` is a hard cut on a soft human-rater score; borderline comments are inherently fuzzy.
4. **Bias**: the Jigsaw task *exists because* toxicity models over-flag identity terms ("I am a gay
   man" scored toxic). The dataset's bias columns let you *measure* this; you have not yet *mitigated*
   it in the DistilBERT model — say so honestly. (The baseline notebooks compute the bias-aware AUC.)
5. **128-token cap** truncates long comments.
6. **Toxicity ≠ harm**: false positives censor legitimate speech, false negatives miss abuse —
   that's why Raven frames output as **"needs review," assisting a human, not auto-deleting.**

## Part 10 — Future work (and the answer to "could you actually distill?")
- **Actually do distillation / compression** (turns the weak spot into a project extension):
  use `unitary/toxic-bert` (or a larger toxicity model) as a **teacher** and train DistilBERT as a
  **student on its soft logits** — *that* would be real knowledge distillation you performed. Or
  **quantise to INT8 / export to ONNX** for a 2–4× speedup.
- **Run in the browser**: ONNX-quantised DistilBERT via Transformers.js → no server needed.
- **Bias mitigation**: re-weight identity subgroups, optimise the Jigsaw bias-aware metric.
- **Multi-label** (threat / insult / obscene / identity-attack) instead of binary.
- **Calibration & threshold tuning** per platform (precision-recall trade-off).
- **Commit the eval artifacts** (confusion matrix PNG, `eval_metrics.json`) for reproducibility.

## Part 11 — Your contribution statement (say this if asked "what did *you* do?")
> "I owned the **transformer model track and the entire product**: I fine-tuned DistilBERT on Jigsaw
> toxicity data, evaluated it (accuracy/F1/ROC-AUC/confusion matrix), wrapped it in a **FastAPI
> inference service with a resilient fallback chain**, and built **three clients on one API contract**
> — a React website with a live demo, a Manifest-V3 browser extension that flags comments inline on
> social platforms, and an Expo mobile app. Teammates contributed the classical/RNN baselines that I
> use as comparison." Honest, specific, and a lot of surface area.

---

## Part 12 — 42 viva questions with model answers

### A. Problem, motivation, scope
**Q1. What problem does Raven solve?**
It triages user comments so moderators only read the risky ones — it scores each comment for
toxicity and flags those that "need review," instead of a human reading everything.

**Q2. Why is this an ML problem and not just a banned-word list?**
Toxicity is contextual: "you killed it!" is praise, "I am a Muslim" is neutral, and abuse is often
implicit with no slur. A keyword list has terrible recall and over-flags identity terms; a learned
model captures context. (Our own demo fallback is a keyword heuristic on purpose — it's the weak
baseline that shows why the model is needed.)

**Q3. Who is the user and what's the output?**
A moderator/platform. Output per comment: `label` (safe/review), `score` (P(toxic) 0–1),
`needs_review` boolean, and the `source` that produced it.

**Q4. Why "needs review" rather than auto-delete?**
Because both error types are costly — false positives censor legitimate speech, false negatives miss
abuse — so Raven is decision-support for a human, not an automatic censor.

**Q5. What's the scope you actually built vs. future?**
Built: fine-tuned model, inference API, website demo, working browser extension, mobile shell.
Future: real distillation/quantisation, browser-local inference, bias mitigation, multi-label.

### B. Data
**Q6. What dataset and why?**
Jigsaw "Unintended Bias in Toxicity Classification" (~1.8M Civil Comments). It's large, real-world,
labelled with a graded toxicity score, and includes identity columns for bias analysis.

**Q7. The label is a float in [0,1] — how did you get a class?**
Binarised at 0.5: `target ≥ 0.5 → toxic (1)`, else non-toxic (0).

**Q8. Isn't a hard 0.5 cut lossy?**
Yes — it throws away the graded signal and makes ~0.5 comments arbitrary. Alternatives: regression
on the raw score, or multi-label using the obscene/threat/insult columns. I chose binary for a clear,
demoable first product.

**Q9. Class balance?**
Raw data is ~11.5:1 (1,660,537 non-toxic vs 144,334 toxic, ~8% toxic).

**Q10. How did you handle the imbalance?**
Down-sampled to **50k per class → 100k balanced**, then stratified 80/20 split (80k train / 20k val).

**Q11. Why down-sample instead of class weights / oversampling?**
Simplicity and speed for a college-compute budget, and it gives a clean balanced accuracy number.
Trade-off: I discard a lot of non-toxic data. Class weights or focal loss on the full set is the
better-scaling alternative (future work).

**Q12. Train/val/test?**
80k train / 20k val (stratified, seed 42). Jigsaw also ships a separate test set; the baselines were
additionally checked on the full ~361k held-out set. The 20k val is balanced (10k/10k).

### C. Model & "distillation" (the trap zone — be precise)
**Q13. Which model and why DistilBERT specifically?**
`distilbert-base-uncased` fine-tuned for sequence classification. I chose it because it's a
**distilled, compressed** transformer — ~66M params, 40% smaller and ~60% faster than BERT-base while
keeping ~97% of its performance — so it's **deployable** on a small budget and could even run in a
browser later. It captures context far better than the RNN/TF-IDF baselines.

**Q14. Did you perform knowledge distillation yourself?**
No — and I'm careful with the word. **DistilBERT is already the product of distillation** (BERT
teacher → DistilBERT student, by HuggingFace). I **fine-tuned** that distilled model on toxicity
data. My contribution is choosing the distilled backbone, fine-tuning it, and serving it. I did not
train a student against a teacher's soft labels.

**Q15. Then explain what knowledge distillation *is*.**
Train a small "student" to match a large "teacher's" soft probability outputs (often with a
temperature on the softmax), so the student inherits the teacher's behaviour at much lower cost. The
soft targets carry more information ("dark knowledge") than hard labels.

**Q16. Could you actually distill a model for this task?**
Yes — I'd use a strong toxicity model (e.g. `unitary/toxic-bert`) as **teacher**, run it over
unlabelled comments to get soft logits, and train DistilBERT as **student** on a KL-divergence +
cross-entropy loss. That's a concrete extension. (See Part 10.)

**Q17. What's the architecture end-to-end?**
Tokenizer (WordPiece, 30,522 vocab, max 128) → embeddings → **6 transformer blocks** (12-head
self-attention, FFN 768→3072→768) → first-token (CLS) hidden state → pre-classifier Linear(768→768)
+ ReLU + dropout → classifier Linear(768→2) → softmax → P(toxic).

**Q18. How many parameters, and where's the saving vs BERT?**
~67M total. BERT-base has 12 transformer layers; DistilBERT has **6** — that halved depth is the
main distillation saving, plus dropping BERT's token-type embeddings and pooler.

**Q19. The pretrained checkpoint had no classifier — what got trained?**
The classification head (`pre_classifier`, `classifier`) is **newly initialised** and learned during
fine-tuning; the transformer backbone weights are also updated (full fine-tuning, not frozen).

**Q20. Why uncased? Any downside?**
`uncased` lowercases everything — smaller vocab, robust to casual capitalisation. Downside: it loses
the signal in ALL-CAPS shouting; a cased model might catch that.

**Q21. Why max_length 128?**
Most comments are short; 128 tokens covers the large majority while keeping training/inference fast
and memory low. Longer comments are truncated — a known limitation.

**Q22. Why softmax over 2 classes and not a single sigmoid?**
Either works for binary. The 2-logit softmax matches `num_labels=2` and gives a clean P(toxic) =
softmax(logits)[1]. My API also supports sigmoid/multi-label heads so it can serve off-the-shelf
multi-label toxicity models.

### D. Training & evaluation
**Q23. Optimiser, learning rate, batch size, epochs?**
HuggingFace Trainer (AdamW, lr ≈ 5e-5 with linear schedule), per-device batch 16, max_len 128.
Trained 1 epoch and 3 epochs; the 3-epoch model is the saved one, but see overfitting below.

**Q24. Final metrics?**
On the 20k balanced val set: **accuracy 0.885, precision 0.875, recall 0.899, F1 0.887, ROC-AUC
0.9515, PR-AUC 0.9495**. Confusion matrix: TN 8712, FP 1288, FN 1011, TP 8989.

**Q25. Why report F1 / AUC and not just accuracy?**
Accuracy hides class behaviour and is misleading under imbalance. F1 balances precision/recall on
the toxic class; ROC-AUC/PR-AUC are threshold-independent and show ranking quality (0.95 AUC = strong
separation).

**Q26. Precision vs recall — which matters more here?**
Depends on policy. For safety you favour **recall** (don't miss abuse); to avoid over-censoring you
favour **precision**. Raven exposes a **tunable threshold** (`RAVEN_THRESHOLD`) so a platform picks
its own operating point on the PR curve. My current model is fairly balanced (P 0.875 / R 0.899).

**Q27. You trained 3 epochs but say 1 was better — explain.**
The 1-epoch run had higher val accuracy/F1 and **much lower val loss (0.537 vs 0.818)**. Training
longer cut *training* loss but raised *validation* loss with no accuracy gain — **mild overfitting**.
The honest choice is the ~1-epoch checkpoint with early stopping on val loss.

**Q28. How do you know it's not memorising?**
Metrics are on a **held-out, stratified** val set the model never trained on; the overfitting signal
(rising val loss) is exactly what I'd watch, and I'd early-stop on it.

**Q29. What does the confusion matrix tell you about error type?**
Slightly more false positives (1288) than false negatives (1011) — it leans toward over-flagging,
which for a "needs review" tool is the safer direction. Recall on toxic (0.90) > on the matrix's FN
rate.

**Q30. How does it beat the baselines?**
On the imbalanced set, TF-IDF+LogReg gets high accuracy but **<45% toxic recall** — it misses most
toxic comments. The transformer understands context, so it catches ~90% of toxic comments while
staying balanced. (Caveat: different test distributions, so I compare *behaviour*, not raw accuracy.)

### E. Deployment, API, engineering
**Q31. How does a trained notebook become a usable service?**
`trainer.save_model()` exports the weights + tokenizer to a folder; the **FastAPI** service
(`raven-api`) loads that folder (`RAVEN_MODEL_DIR`), tokenises requests, runs a `no_grad` forward
pass, applies softmax + threshold, and returns JSON via `/predict` and `/predict-batch`.

**Q32. What's the fallback chain and why have one?**
`ResilientModel` tries the real DistilBERT first; on any error it serves an LLM-via-AI-Gateway
classifier, and if that's unconfigured, a transparent keyword heuristic. So a live demo **never hard
crashes**, and the response always reports its true `source`.

**Q33. Isn't a keyword fallback "cheating"?**
No — it's clearly labelled `*-fallback` in every response and in the UI, and it's only for when the
model server is unreachable. The docs explicitly forbid presenting a fallback as the real model.

**Q34. How does the extension flag comments on a live site?**
A content script finds comment DOM nodes on X/YouTube/Instagram, batches their text, sends it through
the service worker to the API, and attaches an inline Safe/Borderline/Toxic pill with a % score. A
MutationObserver + scroll listener re-scan as the SPA loads more comments; a WeakMap caches
predictions per node.

**Q35. Why a service worker / background script instead of fetching from the content script?**
Manifest V3 routes cross-origin requests through the background service worker (cleaner CORS/permission
model), and it centralises the API call so the content script stays focused on the DOM.

**Q36. What was the hardest engineering bug?**
x.com enforces **Trusted-Types CSP**, which blocks `innerHTML`. I had to build every injected element
(pills, SVG icon) with `document.createElement`/`createElementNS` so the extension isn't blocked.

**Q37. How do all three clients stay consistent?**
They share **one API contract** (`/predict-batch` → `{predictions:[{label,score,needs_review,source}]}`)
and each has the same client-side fallback shape, so the model logic lives in exactly one place.

### F. Ethics, bias, limitations, big-picture
**Q38. What are the ethical risks?**
Over-flagging suppresses legitimate speech (especially identity terms — the exact bias Jigsaw was
built to expose); under-flagging lets abuse through. That's why output is "needs review," assisting a
human, with a tunable threshold — not silent auto-deletion.

**Q39. Does your model have identity bias?**
Likely some — toxicity models notoriously over-score sentences containing identity terms. I have the
Jigsaw bias columns to **measure** subgroup AUC, but I have **not yet mitigated** bias in the
DistilBERT model. Mitigation (subgroup re-weighting, the bias-aware metric) is planned.

**Q40. Biggest limitations?**
English-only, uncased (loses shouting), 128-token cap, binary (no toxicity type), trained on a 100k
subset, and a hard-threshold label. None are fatal for a prototype; all are clear next steps.

**Q41. What would you do with more time/compute?**
Train on the full dataset with class weights, do real teacher→student distillation + INT8/ONNX
quantisation, run the model in-browser via Transformers.js, add multi-label heads, and add bias
mitigation + calibration.

**Q42. In one line, what did you personally build and learn?**
"I fine-tuned a distilled transformer for toxicity and shipped it as a real product — one model, one
API with graceful fallback, three clients — and I learned the gap between a 0.95-AUC notebook and a
system that survives a live web page."

---

### Quick-reference cheat sheet (numbers to never forget)
- Base model: **DistilBERT** (`distilbert-base-uncased`), ~**67M** params, **6** layers, **768** dim, **12** heads.
- Data: **Jigsaw**, binarised at 0.5, balanced to **50k/50k = 100k**, **80k/20k** split, **max_len 128**.
- Final model: **acc 0.885 · F1 0.887 · precision 0.875 · recall 0.899 · ROC-AUC 0.9515 · PR-AUC 0.9495**.
- Confusion: **TN 8712 / FP 1288 / FN 1011 / TP 8989**.
- "I **fine-tuned a distilled** model. I did **not** perform distillation myself." ← the one sentence that keeps you honest.
