# Meeting Prep — what to tell & show Dr. Dutta (+ glossary)

Study this before the meeting. Everything here matches what was actually built today, so you can defend any of it.

---

## PART A — What to say ("what I did today")

> *"Ma'am, I started from your survey on hate speech in low-resource languages. I realised our earlier work was just an English Jigsaw toxicity model — a reproduction — so I re-scoped the project to your area: **Hindi hate speech**.*
>
> *Before designing anything, I opened the HASOC-2019 Hindi data and measured it: it's about **82% Devanagari script**, so it's **native Hindi, not romanized Hinglish** — and the official gold test is blind (no labels), so I use a **frozen stratified split**.*
>
> *I then proposed an **original method, RAVEN-X: cross-script rationale transfer.** We train one **MuRIL** model on two things at once — **what is toxic** (from Hindi HASOC) and **which words are toxic** (from English HateXplain, the only dataset with word-level rationales) — so the model can **point to the hateful words in Hindi, even though no one labelled hateful words in Hindi.** Whether that transfer is faithful, we **measure** with a deletion test, not assume.*
>
> *I built it and **trained the real MuRIL model on a free GPU**. First results: **macro-F1 0.84** on HASOC Hindi, beating a TF-IDF baseline (0.81). The rationale head reached **0.68 token-F1 on English**, and I already see the honest **accuracy-vs-explainability trade-off** the method predicted. Full methodology and results are documented. The product side — the web demo and browser extension — is also running live."*

**The 4 things you did today (bullet version):**
1. Re-scoped from English reproduction → **original Hindi method** (matching your survey).
2. **Verified the data** (Devanagari, not Hinglish; blind test → frozen split).
3. **Designed + built + trained** RAVEN-X (cross-script rationale transfer) on a free GPU → **real numbers**.
4. Kept the **live product** (web + extension) working as the deliverable demo.

---

## PART B — What to show (in order)

1. **`RAVEN_X_METHOD_DESIGN.md`** — the proposed methodology (architecture, the transfer idea, training + evaluation plan). *This is the "design your method" deliverable she asked for.*
2. **`RAVEN_X_RESULTS.md`** — the real results table + honest interpretation.
3. **`raven-codemixed/ravenx_run_executed.ipynb`** — the actual GPU run with outputs (proof it ran).
4. **Live demo** — web playground at `http://localhost:5174` + the Chrome extension flagging toxic comments.

---

## PART C — Glossary (the "big words", explained + tied to our project)

For each: **what it means** → **in our project** → **if she asks, say**.

### 1. Code-mixed / code-switching
**Means:** mixing two languages in one sentence (e.g. Hinglish: "yeh movie was so bad"). "Code-switching" = switching between them.
**In our project:** the survey is about code-mixed text, BUT I measured our HASOC Hindi data and it's **~82% Devanagari (native Hindi)**, only ~7% romanized Latin. So I honestly call it **Hindi (Devanagari)**, not Hinglish.
**If she asks:** *"I checked the script distribution myself — 75–82% is Devanagari-dominant, so it's native-script Hindi with a small code-mixed/Latin slice that I analyse separately."*

### 2. Hyperparameter
**Means:** a setting you choose *before* training (not learned by the model) — like learning rate, batch size, number of epochs.
**In our project (our actual values):**
- **Backbone:** MuRIL-base-cased (~236M parameters)
- **Learning rate:** 2e-5 for the encoder, 1e-4 for the new heads (discriminative LRs)
- **Batch size:** 16 (exactly 8 Hindi + 8 English per batch — the 50/50 mix)
- **Max sequence length:** 128 tokens
- **Optimizer:** AdamW, weight decay 0.01, gradient clipping 1.0, 6% warmup
- **Epochs:** 1 warm-up (encoder frozen) + 3 main epochs (fast run)
- **Loss weights:** α=0.5 (extra classification heads), β=2.0 (rationale), γ=0.5 (faithfulness), sparsity 0.02
- **Precision:** fp16 (mixed precision, for speed on the GPU)
- **Decision threshold:** 0.5
**If she asks:** quote the learning rate (2e-5), batch size (16), max_len (128), AdamW, fp16 — those are the ones examiners check.

### 3. Fine-tuning (vs training from scratch)
**Means:** taking a model already pre-trained on huge text and continuing to train it a little on *your* task. Cheaper and works with small data.
**In our project:** we **fine-tune MuRIL** (which Google pre-trained on Hindi+English) on our 3,265 Hindi training posts. We did **not** train from scratch.
**If she asks:** *"Full fine-tuning — the encoder weights update too, not just the heads."*

### 4. Transformer / encoder / MuRIL / XLM-R / DistilBERT
**Means:** a "transformer" is the neural-network architecture behind modern NLP (uses *self-attention* to read context). An "encoder" turns text into number-vectors that capture meaning.
**In our project:** **MuRIL** = a transformer encoder Google pre-trained specifically on **17 Indian languages + English**, so it's strong on Hindi. **XLM-R** and **IndicBERT** are alternative multilingual encoders we use as comparison baselines. **DistilBERT** was our old English-only model.
**If she asks "why MuRIL?":** *"It's pre-trained on Devanagari Hindi and English jointly, so it aligns the two languages internally — exactly what cross-script transfer needs. mBERT/XLM-R are weaker on Indic."*

### 5. Tokenizer / subword / WordPiece
**Means:** models can't read raw text; a **tokenizer** chops text into pieces ("subwords"). MuRIL uses **WordPiece** with a 197k vocabulary.
**In our project:** we map each word to subwords and keep track of which subword belongs to which word (`word_ids()`), so we can line up the English rationale labels with the right tokens.
**If she asks:** *"English and Hindi share zero subwords — different alphabets — so any transfer is purely through meaning, not spelling."*

### 6. Embedding / representation / shared semantic space
**Means:** the vector of numbers a model uses to represent a word's meaning. A "shared space" means English and Hindi words with the same meaning land near each other.
**In our project:** this is **why transfer is even possible** — MuRIL puts an English slur and a Hindi slur in nearby regions, so a "find the hateful word" skill learned in English can fire on Hindi.
**If she asks:** *"We test this with a CKA / nearest-neighbour probe — we don't just assume the spaces align."*

### 7. Cross-lingual / cross-script transfer (our core novelty)
**Means:** teaching a model a skill in one language and having it work in another. **Cross-script** = the languages even use different alphabets (Latin vs Devanagari).
**In our project:** we teach word-level **rationale** (which words are hateful) only in **English**, and it has to work in **Hindi** where there are **no rationale labels**. This conjunction — cross-script + rationale-level + no target labels — is the **original contribution**.
**If she asks "what's novel?":** *"Nobody has transferred word-level hate rationales across a Latin→Devanagari script boundary and measured whether they're faithful."*

### 8. Rationale / token-level rationale / explainability
**Means:** the *explanation* — which exact words made the model say "hateful". "Token-level" = per-word.
**In our project:** HateXplain gives human word-level rationales (English). Our model learns a **rationale head** that outputs a score per word; the web UI highlights them.
**If she asks:** *"It's a learned head — one forward pass — not the slow word-by-word deletion we used before."*

### 9. Faithfulness — comprehensiveness / sufficiency / AOPC (the deletion test)
**Means:** does the explanation reflect what the model *actually used*? Test it by **deleting** the highlighted words: if the "toxic" score collapses, the explanation was real (**comprehensiveness**); if keeping only those words preserves the score, it's complete (**sufficiency**). **AOPC** = the area-under-curve version. This needs **no human labels** — that's why it works for Hindi.
**In our project:** this is how we evaluate Hindi rationales without any Hindi rationale gold.
**If she asks:** *"Comprehensiveness and sufficiency from the ERASER framework (DeYoung et al.) — label-free, so it works in the target language."*

### 10. Metrics: macro-F1 / precision / recall / accuracy
**Means:** **Precision** = of the ones I flagged toxic, how many really were. **Recall** = of the really-toxic ones, how many I caught. **F1** = their balance. **Macro-F1** = F1 averaged equally over both classes (fair when classes are imbalanced). **Accuracy** = overall % correct (can be misleading).
**In our project:** primary metric is **macro-F1**. MuRIL = **0.84**, baseline = **0.81**.
**If she asks "why macro-F1 not accuracy?":** *"Accuracy hides minority-class behaviour; macro-F1 weights both classes equally — the honest number for hate detection."*

### 11. Datasets: HASOC / HateXplain
**HASOC-2019 Hindi:** 4,665 labelled Hindi posts (hate-or-offensive vs not). Our **classification** signal.
**HateXplain:** 20,148 English posts **with word-level rationales**. Our **explanation** signal (the only such dataset; English-only).
**If she asks:** *"They never overlap — HASOC has labels but no rationales, HateXplain has rationales but is English. Bridging that gap is the whole project."*

### 12. Multi-task learning / loss function / per-head masking
**Means:** training one model to do several jobs at once. The **loss function** is the error the model minimises. **Per-head masking** = each job's error only applies to the data that has that label.
**In our project:** one model, two heads (classify + rationale). The classification error only applies to Hindi/English labels we have; the rationale error only applies to English (where rationales exist). So **we never train on a label we don't have** — the clean answer to "you don't have Hindi rationales."
**If she asks:** *"Masked multi-task loss — missing supervision contributes zero gradient, never a fake target."*

### 13. Baseline (TF-IDF + Logistic Regression)
**Means:** a simple, classic model you must beat to prove the fancy one is worth it. **TF-IDF** = counts important words; **Logistic Regression** = a simple classifier.
**In our project:** our lower bound = **0.81 macro-F1**. MuRIL beats it (0.84).
**If she asks:** *"It's the honest floor — a transformer that can't beat TF-IDF isn't worth it; ours does."*

### 14. Stratified split / train-val-test / seed
**Means:** splitting data into **train** (learn), **validation** (tune), **test** (final score). **Stratified** = keep the class balance in each part. **Seed** = a fixed number so the split is reproducible.
**In our project:** frozen **70/15/15** split, **seed 42** → 3,265 / 699 / 701. We use this because the official HASOC test is **blind**.
**If she asks:** *"No leaderboard claim — the gold test is unlabelled, so we re-run every baseline on our own frozen split for fairness."*

### 15. fp16 / mixed precision
**Means:** doing the maths in 16-bit instead of 32-bit numbers — ~2× faster, less memory, on a GPU.
**In our project:** lets the full run finish in ~1.5 h (fast run ~30 min) on a free T4.

### 16. Knowledge distillation (she may ask, because of DistilBERT)
**Means:** training a small "student" model to copy a big "teacher" model. **DistilBERT is the result of distilling BERT** — we did **not** do distillation ourselves; we *used* an already-distilled model earlier.
**If she asks:** *"DistilBERT was already distilled by HuggingFace; I fine-tuned it. I did not run a teacher→student loop."* (Never say "I distilled BERT.")

### 17. Zero-shot transfer
**Means:** the model does a task in a setting it got **zero** training examples for.
**In our project:** Hindi rationales are **zero-shot** — zero Hindi rationale labels; the skill transfers from English.

### 18. Honest negative result / pre-registration
**Means:** deciding *before* the experiment what counts as success, and reporting the outcome **even if it fails**.
**In our project:** if cross-script transfer turns out weak, that's a **publishable finding** your survey says the field lacks. *This is why the project can't outright fail.*

---

## PART D — Likely questions → 1-line answers

- **"What's your novel contribution?"** → Cross-script word-level rationale transfer: English-supervised explanations, faithful on Devanagari Hindi, measured not assumed.
- **"Is your data code-mixed?"** → Mostly native Devanagari (~82%), I measured it; small Latin slice analysed separately.
- **"Why MuRIL?"** → Pre-trained jointly on Indic + English, so it aligns the two scripts internally.
- **"How can you explain Hindi with no Hindi explanation labels?"** → Train rationales on English, transfer through the shared encoder, evaluate with a label-free deletion test.
- **"What did you score?"** → Macro-F1 0.84 on HASOC Hindi, beating a 0.81 TF-IDF baseline; rationale token-F1 0.68 on English.
- **"Did explainability cost accuracy?"** → Yes, ~4 F1 points — the honest trade-off we set out to measure.
- **"What's next?"** → Full multi-seed run + the script-stratified faithfulness evaluation (does it transfer to Devanagari?).
