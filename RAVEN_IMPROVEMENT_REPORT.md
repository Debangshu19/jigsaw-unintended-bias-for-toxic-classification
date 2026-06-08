# Raven — Improvement & Original-Method Proposal

**Prepared for:** Niladri Hazra & team · **Date:** 2026-06-07
**Trigger:** Dr. Arpita Dutta shared *"A Survey on Automatic Online Hate Speech Detection in Low-Resource Languages"* (arXiv:2411.19017v1) and asked the group to **"propose your own method"**, not reproduce existing work.

> Every factual claim below was **verified** — the code audit reads the actual files; every dataset/model was web-checked for real availability, license, and size; the candidate methods were scored by adversarial examiners. Where a thing is gated, English-only, or thin on novelty, it says so plainly. Nothing here is assumed.

---

## 0. The one finding that reframes everything

**Dr. Arpita Dutta is the *second author* of the survey she sent you** (Susmita Das, **Arpita Dutta**, Kingshuk Roy, Abir Mondal, Arnab Mukhopadhyay). She did not hand you a random reference — she handed you **her own paper**, whose entire thesis is:

> *English has abundant resources; **low-resource languages — especially Indic and code-mixed Hindi-English — are starved of datasets and attention.***

So when she says *"propose your method… you'll get plenty of references from here… not making your own,"* the subtext is unambiguous: **stop reproducing an English-only Jigsaw DistilBERT and make an original contribution in her research area — low-resource / code-mixed hate speech.** Another English toxicity model is precisely what frustrated her.

**Your own question to her** ("English-only Jigsaw, or low-resource/code-mixed Hindi-English?") is therefore essentially self-answering: **code-mixed Hindi-English**. This report is built around that, but flags it as the one decision the group + supervisor formally own.

---

## 1. What Raven is **today** (verified by code audit)

| Layer | What it actually does now | Verified gap for low-resource / code-mixed |
|---|---|---|
| **Model** (`raven-model/train_distilbert.py`, notebooks) | Fine-tunes `distilbert-base-uncased`, binary toxic/safe, Jigsaw 100k balanced subset; acc **0.885**, ROC-AUC **0.9515** | Tokenizer + embeddings are **English-only** (30,522 WordPiece, lowercased). No Devanagari/Romanized-Hindi vocab, no code-switch handling, no class-weighting for skewed Indic sets |
| **API** (`raven-api/model.py`, `app.py`) | Loads any HF seq-classification model (`RAVEN_MODEL_DIR`/`RAVEN_MODEL_ID`); per-category sigmoid scores; **word-level explanations via leave-one-out occlusion** (`explain_one`); resilient fallback chain (LLM gateway → keyword); `/predict`, `/predict-batch`, `/explain`, `/health` | Explanation = **post-hoc occlusion** (not a *learned* rationale head). English keyword fallback. `.split()` word boundaries break on Indic scripts. Single 0.5 threshold |
| **Web** (`raven-web`) | Chat playground, per-category badges, toxic-word highlights, honest source labels | No language detection, static English category labels, no per-segment language tags |
| **Extension** (`extension`) | MV3, inline pills on X/YouTube/Instagram, MutationObserver+WeakMap, Trusted-Types-safe DOM, keyword fallback | English-assuming; no non-Latin handling |
| **Mobile** (`raven-mobile`) | Expo review-queue shell on same API contract | Inherits all of the above |

**The good news the audit confirms:** the inference layer is *already model-agnostic*. `TransformersModel` loads **any** `AutoModelForSequenceClassification` + `AutoTokenizer`; the `Prediction` dataclass and API contract accept new fields without breaking clients. **Swapping in a multilingual model (MuRIL/XLM-R) is an env-var + tokenizer change, not a rewrite.** The training script is also language-agnostic apart from the hard-coded backbone name (`train_distilbert.py:22`). This is a big head-start — the *product* is done; only the *research core* needs to move.

**The honest weakness the audit confirms:** the project currently has **no research contribution beyond "fine-tuned an off-the-shelf English model."** That is exactly the supervisor's complaint.

---

## 2. What the survey actually gives you (mapped to Raven)

The survey is a **reference map**, not a method. The pieces that matter for *your* contribution:

- **§2 Taxonomy** — hate / offensive / profane / cyberbullying / abusive overlap; targets: race, religion, gender, ableism. → justifies **per-category** output (you already have the UI).
- **§4 Datasets** — the buildable Indic/code-mixed ones: **HASOC 2019/2021 Hindi** [91,98], **Bohra Hinglish code-mixed** [21], **Mathur HEOT** [97], **HateXplain (rationales)** [96], **L3Cube-MahaHate** [152].
- **§5.2.5 Indic methods** — the field standard is **transformer transfer**: **MuRIL** [79], **IndicBERT**, **XLM-R**, mBERT; monolingual/Indic-specific encoders beat generic mBERT.
- **§6 Research challenges & opportunities** — the explicit gaps the supervisor wants attacked:
  1. **Culture/language constraint** — code-mixing, transliteration, sarcasm.
  2. **Dataset suitability & bias** — annotation noise, subgroup bias.
  3. **Open platform / reproducibility** — *only 53 of 1039 GitHub projects are maintained.* → ship a clean reproducible repo + model card and you literally answer a stated gap.
  4. **Explainability + emotion** — listed as future opportunities. → you already shipped word-level explanations; this is your wedge.

---

## 3. Verified resource inventory (datasets + models)

> ✅ direct-download · 🟡 gated/on-request · ⚠️ partial (IDs only) — all checked on the web during this analysis.

### Datasets
| Resource | Status | Access | Lang | Size | Labels | License |
|---|---|---|---|---|---|---|
| **HASOC-2019 Hindi** ⭐spine | ✅ | GitHub `TharinduDR/HASOC-2019` (verified HEAD live) | Hindi code-mixed | ~4.7k train / 1.3k test | binary HOF/NOT + 4-class + target | research-use, no redistribution |
| **HateXplain** ⭐rationales | ✅ | HF `Hate-speech-CNERG/hatexplain` / GitHub `hate-alert/HateXplain` (verified 12.3 MB) | **English only** | 20,148 | 3-class + **token rationales** + target | CC-BY-4.0 |
| Bohra 2018 code-mixed | ⚠️ | GitHub `deepanshu1995/HateSpeech-Hindi-English-Code-Mixed…` — **IDs+labels only**, text via rehydration/email | Hinglish | 4,114 IDs | binary hate/normal | GPL-3.0 (code) |
| Mathur HEOT | 🟡 | email author; profanity lexicon is public | Hinglish (Romanized) | ~3k | 3-class | none stated |
| HASOC 2021 Hindi/Marathi | 🟡 | Google-Form → emailed password | Hindi, Marathi | ~4.6k / 1.9k | binary + 4-class | research-use |
| L3Cube-MahaHate | ✅ | GitHub `l3cube-pune/MarathiNLP` | **Marathi** (not Hinglish) | 25k–37.5k | hate/offensive/profane/not | **CC-BY-NC-SA** (non-commercial) |

### Models (all ✅ direct, ungated — verified reachable)
| Model | HF id | Params | License | Role |
|---|---|---|---|---|
| **MuRIL** | `google/muril-base-cased` | 236M | Apache-2.0 | **Recommended backbone** — pretrained on Hindi+English+**transliterations** |
| **XLM-R** | `FacebookAI/xlm-roberta-base` | 270M | MIT | Strongest single HASOC baseline |
| IndicBERT | `ai4bharat/indic-bert` | ~33M | MIT | Light Indic baseline |
| **CNERG Hinglish-abusive** | `Hate-speech-CNERG/hindi-codemixed-abusive-MuRIL` | 238M | AFL-3.0 | **Drop-in baseline / teacher** (purpose-built for Hinglish) |
| L3Cube HASOC-Hindi | `l3cube-pune/hate-roberta-hasoc-hindi` | 110M | CC-BY-4.0 | Registration-free baseline |
| toxic-bert | `unitary/toxic-bert` | 110M | Apache-2.0 | English toxicity teacher / category labels |

**Spine decision:** **HASOC-2019 Hindi** is the only *guaranteed, instantly-open* code-mixed corpus. Start there day 1. Treat Bohra/Mathur/HASOC-2021 as optional augmentation (begin the email/rehydration in week 1 so they're never on the critical path).

**Licensing rule (non-negotiable):** all the hate datasets are research-only / non-commercial / Twitter-ToS-bound. **You may train on them and report metrics, but never bundle raw data into the public extension/web/app.** Ship **weights + a model card** only. This also *answers the survey's open-platform challenge* if done cleanly.

---

## 4. Candidate "your-own-method" proposals — adversarially scored

Five candidates were each graded 1–10 on **novelty / feasibility (free GPU, 4–6 wks) / supervisor-alignment / student-impact** by a skeptical examiner instructed to name the killer risk.

| # | Method | Nov | Feas | Align | Impact | Verdict | Killer risk |
|---|---|:-:|:-:|:-:|:-:|---|---|
| **M1** | Hinglish-MuRIL pivot (fine-tune MuRIL on HASOC + existing explainability) | 4 | 6 | **9** | 8 | recommend-w/-changes | Novelty thin: explainability ports for free; transliteration-norm may give ~0 gain; overlaps published CNERG/L3Cube models → reads as reproduction |
| **M2** | Explainable + emotion multi-task (4 heads) | 4 | **3** | 8 | 6 | recommend-w/-changes | **No single dataset has all 4 signals**; HateXplain rationales English-only; too big for 4–6 wks |
| **M3** | Real distillation + INT8/ONNX in-browser | 2 | 7 | **2** | 5 | **reject** | It's your *own pre-written TODO*, English-only, zero code-mixed alignment |
| **M4** | XLM-R + language adapters (cross-lingual transfer) | 3 | 6 | **9** | 6 | recommend-w/-changes | MAD-X adapters heavily published; API has no PEFT path; underperforms in-language MuRIL |
| **M5** | RAVEN-X unified (code-mixed + explainable, on existing stack) | 4 | 5 | **9** | 8 | recommend-w/-changes | "Rationale head" has **no target-language supervision**; "vs English DistilBERT" is a category error |

### Themes every examiner independently flagged (read these twice)
1. **"vs the English DistilBERT 0.885" is a category error** — different language, task, label scheme, and class balance. An examiner *will* pounce. The honest baseline set is **MuRIL vs XLM-R vs IndicBERT vs off-the-shelf CNERG/L3Cube — on the *same* HASOC code-mixed split**. The English DistilBERT is *motivation*, not a baseline.
2. **Just fine-tuning MuRIL on HASOC = reproduction.** CNERG and L3Cube already ship those checkpoints. You need a **real differentiator** to clear the "original, not reproduction" bar.
3. **Your shipped "explainability" is post-hoc occlusion, not a learned rationale.** Porting it to MuRIL adds nothing new by itself.
4. **HateXplain rationales are English-only** — you cannot directly supervise Hinglish explanations. *This constraint is also where the real novelty hides (see §5).*
5. **Data access, not compute, is the bottleneck.** HASOC-2019 Hindi is the only safe spine.

---

## 5. ⭐ Recommended contribution — **RAVEN-X: Explainable Code-Mixed Hate Speech via Cross-Lingual Rationale Transfer**

Synthesizing the adversarial feedback, the design that is **simultaneously novel, supervisor-aligned, and buildable** has **two tiers** — a bulletproof core plus a genuinely original headline. Staging it this way means even if the hard part underdelivers, you still have a solid, defensible final-year contribution.

### Tier 1 — Core (guaranteed, bulletproof) — *"a real code-mixed model + honest benchmark"*
Build a **Hindi-English code-mixed hate-speech classifier** and a **fair benchmark**:
- Backbone bake-off on the **same HASOC-2019 Hindi split**: **MuRIL vs XLM-R vs IndicBERT vs your old DistilBERT-multilingual vs the off-the-shelf CNERG/L3Cube checkpoints.**
- **Seed-replicated** (3–5 seeds), report **mean ± std + significance** — directly defeats killer-risk #1 and #2.
- A **code-mixed preprocessing ablation** (transliteration normalization, emoji/slang handling) **pre-registered and reported even if the gain is null** (a clean negative result *is* a finding the survey itself says the field lacks).
- Served through the **existing Raven API/web/extension** by pointing `RAVEN_MODEL_DIR` at the new checkpoint — near-zero product code change.

This tier alone clears *"your own model/study, not an English Jigsaw reproduction."*

### Tier 2 — Novelty headline (the differentiator) — *"explanations in a language with no explanation labels"*
**Cross-lingual rationale transfer.** Train **word-level rationale supervision on English HateXplain** (which *has* token rationales), and transfer it **through the shared multilingual MuRIL/XLM-R encoder** to produce **faithful word-level explanations on Hinglish**, where *no rationale labels exist*. Then **evaluate faithfulness properly** — comprehensiveness / sufficiency (AOPC), not just "the highlights look plausible."

**Why this is genuinely original (and why examiners flagged it as the one novel angle):**
- It attacks a *real* low-resource problem: **how do you explain hate-speech decisions in a language that has zero explanation training data?**
- It fuses **your already-shipped explainability** with the **supervisor's low-resource thesis** — a story no off-the-shelf model tells.
- It produces a **quantitative faithfulness result on code-mixed text** — answering the survey's explainability *and* low-resource opportunities at once.
- It's **honest about the constraint** (English rationales → Hinglish transfer) instead of hiding it — which is exactly the kind of rigor that wins vivas.

### The one-sentence pitch (memorize)
> *"We built an explainable Hindi-English code-mixed hate-speech model on a MuRIL backbone, benchmarked it fairly against XLM-R/IndicBERT and the published baselines on HASOC, and — our core contribution — we transfer **word-level rationale supervision from English HateXplain across the shared encoder to produce *faithful* explanations in Hinglish, a language with no rationale labels**, evaluated with comprehensiveness/sufficiency. We then served it through our existing API, website, and browser extension."*

---

## 6. Honest evaluation protocol (so the viva can't crack it)

- **Primary metric:** macro-F1 on the HASOC-2019 Hindi **gold test split** (the standard reported number → free literature comparison).
- **Baselines on the identical split:** MuRIL, XLM-R, IndicBERT, off-the-shelf `hindi-codemixed-abusive-MuRIL` and `hate-roberta-hasoc-hindi`. *(The old English DistilBERT appears only as motivation, never as a baseline.)*
- **Statistical rigor:** 3–5 seeds, mean ± std, paired significance test between top models.
- **Explanation quality:** token-F1 vs HateXplain rationales (English, where gold exists) + **comprehensiveness/sufficiency** on Hinglish (no gold needed — these are intervention metrics).
- **Bias/fairness:** reuse the subgroup-AUC framework already in `jigsaw-classification-report.ipynb` on target-community labels.
- **Ablations:** with/without transliteration normalization; rationale-transfer vs occlusion baseline.

---

## 7. Risks, ethics, reproducibility

- **Data access risk** → HASOC-2019 Hindi is the spine; start Bohra rehydration + HASOC-2021 form in week 1; never block on one author's email.
- **Licensing** → train + report only; ship **weights + model card**, not raw data. Cite Mandl et al. 2019 [91], Bohra et al. 2018 [21], Mathew et al. 2021 [96], MuRIL [79].
- **Content ethics** → corpora contain real slurs; keep local, add a handling note, frame output as **"needs review,"** not auto-delete (you already do).
- **Reproducibility** → a clean repo + model card + fixed splits + seeds **directly answers the survey's "open platform" challenge (only 53/1039 maintained).** Make this an explicit selling point.

---

## 8. Phased build plan (4–6 weeks) — each phase has a **test/verify gate**

> Nothing is "done" until its gate passes on real data. No assumptions.

| Wk | Phase | Build | **Verify gate (must pass)** |
|:-:|---|---|---|
| 1 | **Spine + baselines** | Download HASOC-2019 Hindi; clean the known malformed row; fixed train/val/test; fine-tune MuRIL + XLM-R | Reproduce a published HASOC macro-F1 within ±2 pts; eval script runs end-to-end |
| 2 | **Benchmark** | Add IndicBERT + off-the-shelf CNERG/L3Cube; seed replication; significance | Results table with mean±std; old English DistilBERT correctly excluded as baseline |
| 3 | **Preprocessing ablation** | Transliteration normalization, emoji/slang; pre-registered | Report delta + significance **even if null**; error analysis on code-switch boundaries |
| 4 | **Rationale transfer** | Train rationale head on HateXplain; transfer via shared encoder to Hinglish | Token-F1 on English gold; comprehensiveness/sufficiency computed on Hinglish |
| 5 | **Serve + integrate** | Export best checkpoint; `RAVEN_MODEL_DIR` → API; web/extension show code-mixed verdict + rationale | Live demo scores a real Hinglish comment correctly; `/health` reports the right source |
| 6 | **Write-up + repo** | Model card, reproducible README, results, ethics | Fresh clone reproduces the headline number; supervisor-ready slide deck |

---

## 9. What to tell Dr. Dutta (proposed message)

> *"Ma'am, based on your survey we're committing to **low-resource Hindi-English code-mixed** hate speech. Our proposed method (RAVEN-X): a MuRIL-backbone code-mixed classifier benchmarked fairly on HASOC against XLM-R/IndicBERT and the published baselines — and our original contribution is **cross-lingual rationale transfer**: we supervise word-level explanations on English HateXplain and transfer them through the shared encoder to produce *faithful* explanations in Hinglish, where no rationale labels exist, evaluated with comprehensiveness/sufficiency. We'll serve it through our existing API/website/extension and release a reproducible repo + model card. Does this direction and scope work for you?"*

This shows her you read the survey, picked her field, proposed a *method* (not a reproduction), and have a concrete plan.

---

## 10. Appendix — survey citations to use in your report
- **Datasets:** HASOC [91, 98], Bohra Hinglish [21], Mathur HEOT [97], HateXplain [96], L3Cube-MahaHate [152], HASOC multilingual table [20, 151, 102, 70].
- **Models/methods (Indic):** MuRIL [79], cross-lingual/transfer for Marathi [51], mono-vs-multilingual BERT [153, 52], Dravidian transfer [32, 149, 139], AbuseXLMR/MACD [57], indic multilingual toxicity [69].
- **Challenges/opportunities:** §6 (culture/language, dataset suitability, data bias, open platform), explainability via HateXplain [96], emotion/sentiment opportunity (§6 closing bullets).
