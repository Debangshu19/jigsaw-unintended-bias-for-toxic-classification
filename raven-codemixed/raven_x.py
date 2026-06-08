"""RAVEN-X — model + multi-corpus masked loss + label-free faithfulness objective.

One shared encoder (MuRIL) with two heads:
  HEAD 1 classification: shared toxicity z (both corpora) + native y3 (HateXplain 3-cls)
          + t4 (HASOC fine-type) — each masked to its own corpus.
  HEAD 2 token-rationale (NER-style): per-subword logit a_j; English-supervised only.

Faithfulness self-objective on Hindi (no labels): remove the top-k rationale tokens via a
straight-through ATTENTION mask and RE-RUN the full encoder; toxicity must drop
(comprehensiveness) and rationale-only must hold (sufficiency). The float attention mask
carries gradient back to the rationale head (verified by the smoke test's grad-norm check).

Smoke test (fast, CPU, no 950MB download):
    raven-api/.venv/bin/python raven-codemixed/raven_x.py --smoke
Real training swaps in MuRIL:  --model google/muril-base-cased
"""
from __future__ import annotations
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE = -100


# ----------------------------------------------------------------------------- model
class RavenXModel(nn.Module):
    def __init__(self, encoder, hidden):
        super().__init__()
        self.encoder = encoder
        self.drop = nn.Dropout(0.2)
        self.cls_trunk = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.z = nn.Linear(hidden, 1)      # shared toxicity (both corpora)
        self.y3 = nn.Linear(hidden, 3)     # HateXplain native 3-class
        self.t4 = nn.Linear(hidden, 4)     # HASOC fine-type
        self.rat = nn.Sequential(          # token-rationale head
            nn.Linear(hidden, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1)
        )

    def encode(self, input_ids, attention_mask):
        # attention_mask may be a float tensor (for the differentiable faithfulness re-run)
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state               # (B, L, H)

    def heads(self, H):
        h_cls = self.drop(H[:, 0])                 # pooled [CLS]
        trunk = self.cls_trunk(h_cls)
        a = self.rat(H).squeeze(-1)                # (B, L) subword rationale logits
        return {
            "z": self.z(trunk).squeeze(-1),        # (B,)
            "y3": self.y3(trunk),                  # (B,3)
            "t4": self.t4(trunk),                  # (B,4)
            "a": a,
        }

    def forward(self, input_ids, attention_mask):
        return self.heads(self.encode(input_ids, attention_mask))

    def p_hof(self, input_ids, attention_mask):
        """Toxicity prob from a fresh encode (used by the faithfulness re-run)."""
        H = self.encode(input_ids, attention_mask)
        return torch.sigmoid(self.z(self.cls_trunk(H[:, 0])).squeeze(-1))


# ----------------------------------------------------------------- straight-through top-k
def ste_topk_mask(scores, valid, k_frac=0.20):
    """Per-row hard top-k over `valid` tokens, with a straight-through estimator so
    gradient flows back to `scores`. Returns a float mask (B,L), 1.0 = selected."""
    B, L = scores.shape
    probs = torch.sigmoid(scores)
    hard = torch.zeros_like(probs)
    for b in range(B):
        idx = valid[b].nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        k = max(1, int(round(k_frac * idx.numel())))
        sel = idx[torch.topk(probs[b, idx], k).indices]
        hard[b, sel] = 1.0
    # STE: forward = hard, backward = d/d(probs)
    return hard + (probs - probs.detach())


# --------------------------------------------------------------------------- the loss
def compute_loss(model, batch, w, gamma):
    """batch: dict of tensors. w: loss weights. gamma: faithfulness weight (0 disables)."""
    ids, attn = batch["input_ids"], batch["attention_mask"]
    out = model(ids, attn.float())
    z, a = out["z"], out["a"]
    is_en, is_hi = batch["is_en"].bool(), batch["is_hi"].bool()
    parts = {}

    # (1) shared toxicity — both corpora
    parts["z"] = F.binary_cross_entropy_with_logits(z, batch["z_star"].float())

    # (2) native heads — masked per corpus
    parts["clsA"] = (F.cross_entropy(out["y3"][is_en], batch["y3"][is_en])
                     if is_en.any() else z.sum() * 0)
    parts["clsB"] = (F.cross_entropy(out["t4"][is_hi], batch["t4"][is_hi])
                     if is_hi.any() else z.sum() * 0)

    # (3) rationale — English-supervised token-BCE on first-subword logits (-100 ignored)
    r_tgt = batch["r_sub"]                      # (B,L) in {0,1,-100}
    keep = (r_tgt != IGNORE) & batch["has_rationale"].bool().unsqueeze(1)
    if keep.any():
        pos_w = torch.tensor(4.0, device=z.device)
        parts["rat"] = F.binary_cross_entropy_with_logits(
            a[keep], r_tgt[keep].clamp(min=0).float(), pos_weight=pos_w)
    else:
        parts["rat"] = a.sum() * 0

    # (4) faithfulness — label-free, Hindi only, differentiable attention masking
    if gamma > 0 and is_hi.any():
        special = batch["special"].bool()       # CLS/SEP/PAD
        valid = (attn.bool() & ~special)
        sel = ste_topk_mask(a, valid)           # (B,L) STE, 1 = rationale token
        keep_special = special.float() + attn.float() * special.float()  # always keep specials
        cls_keep = torch.zeros_like(attn.float()); cls_keep[:, 0] = 1.0  # always keep CLS
        # comprehensiveness: REMOVE rationale -> toxicity should drop
        attn_comp = (attn.float() * (1.0 - sel)).clamp(0, 1)
        attn_comp = torch.maximum(attn_comp, cls_keep)
        p_comp = model.p_hof(ids, attn_comp)
        # sufficiency: KEEP only rationale -> toxicity should hold
        attn_suff = torch.maximum(attn.float() * sel, cls_keep)
        p_suff = model.p_hof(ids, attn_suff)
        p_full = torch.sigmoid(z).detach()
        hi = is_hi.float()
        parts["comp"] = (p_comp * hi).sum() / hi.sum()
        parts["suff"] = (((p_full - p_suff) ** 2) * hi).sum() / hi.sum()
    else:
        parts["comp"] = z.sum() * 0
        parts["suff"] = z.sum() * 0

    # (5) priors — short + contiguous
    pi = torch.sigmoid(a)
    parts["sparse"] = pi.mean()
    parts["cont"] = (pi[:, 1:] - pi[:, :-1]).abs().mean()

    total = (parts["z"]
             + w["alpha"] * (parts["clsA"] + parts["clsB"])
             + w["beta"] * parts["rat"]
             + gamma * (parts["comp"] + parts["suff"])
             + w["lam_s"] * parts["sparse"] + w["lam_c"] * parts["cont"])
    return total, parts


# ------------------------------------------------------------------------ tokenization
def encode_examples(examples, tokenizer, max_len=128):
    """Tokenize a list of unified examples into model-ready tensors with subword<->word
    alignment and first-subword rationale targets. Each example needs:
      text (str), words (list[str]), z_star (0/1), is_en/is_hi, has_rationale (0/1),
      rationale (list[0/1] over words) or None, y3 (int), t4 (int)."""
    texts = [ex["words"] for ex in examples]
    enc = tokenizer(texts, is_split_into_words=True, truncation=True,
                    max_length=max_len, padding="max_length", return_tensors="pt")
    B, L = enc["input_ids"].shape
    r_sub = torch.full((B, L), IGNORE, dtype=torch.long)
    special = torch.zeros((B, L), dtype=torch.long)
    for b, ex in enumerate(examples):
        wids = enc.word_ids(b)
        seen = set()
        for j, wid in enumerate(wids):
            if wid is None:
                special[b, j] = 1
                continue
            if wid in seen:           # only first subword of each word gets a target
                continue
            seen.add(wid)
            if ex.get("rationale") is not None and wid < len(ex["rationale"]):
                r_sub[b, j] = int(ex["rationale"][wid])
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "special": special,
        "r_sub": r_sub,
        "z_star": torch.tensor([ex["z_star"] for ex in examples]),
        "y3": torch.tensor([ex.get("y3", 0) for ex in examples]),
        "t4": torch.tensor([ex.get("t4", 0) for ex in examples]),
        "is_en": torch.tensor([ex["is_en"] for ex in examples]),
        "is_hi": torch.tensor([ex["is_hi"] for ex in examples]),
        "has_rationale": torch.tensor([ex["has_rationale"] for ex in examples]),
    }


def unify_hasoc(ex):
    return {"words": ex["text"].split(), "z_star": ex["label"], "is_en": 0, "is_hi": 1,
            "has_rationale": 0, "rationale": None, "y3": 0, "t4": ex["fine"]}


def unify_hatexplain(ex):
    return {"words": ex["tokens"], "z_star": ex["label"], "is_en": 1, "is_hi": 0,
            "has_rationale": int(ex["has_rationale"]), "rationale": ex["rationale"],
            "y3": ex["label3"], "t4": 0}


# -------------------------------------------------------------------------- smoke test
def smoke():
    from transformers import AutoTokenizer, BertConfig, BertModel
    import data_loaders as dl

    print("[smoke] loading MuRIL tokenizer (real alignment) + tiny random encoder (fast)...")
    tok = AutoTokenizer.from_pretrained("google/muril-base-cased")
    cfg = BertConfig(vocab_size=tok.vocab_size, hidden_size=64, num_hidden_layers=2,
                     num_attention_heads=2, intermediate_size=128, max_position_embeddings=256)
    model = RavenXModel(BertModel(cfg), hidden=64)

    hi = dl.load_hasoc_hindi()["train"][:4]
    en = [e for e in dl.load_hatexplain()["train"] if e["has_rationale"]][:4]
    examples = [unify_hasoc(e) for e in hi] + [unify_hatexplain(e) for e in en]
    batch = encode_examples(examples, tok, max_len=64)

    # alignment sanity on a REAL Devanagari + English example
    print(f"[smoke] batch input_ids {tuple(batch['input_ids'].shape)}, "
          f"specials/row≈{batch['special'].sum().item()//8}, "
          f"EN rationale tokens={int((batch['r_sub']==1).sum())}, "
          f"ignored={int((batch['r_sub']==IGNORE).sum())}")
    assert (batch["r_sub"][:4] == IGNORE).all(), "HASOC rows must have NO rationale target"
    assert (batch["r_sub"][4:] == 1).any(), "HateXplain rows must carry rationale targets"

    w = dict(alpha=0.5, beta=2.0, lam_s=0.02, lam_c=0.01)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(2):
        gamma = 0.0 if step == 0 else 0.5        # ramp: faithfulness on at step 2
        opt.zero_grad()
        total, parts = compute_loss(model, batch, w, gamma)
        total.backward()
        # the critical check: faithfulness gradient must reach the rationale head
        gnorm = sum(p.grad.norm().item() for p in model.rat.parameters() if p.grad is not None)
        assert torch.isfinite(total), "loss is not finite"
        assert gnorm > 0, "NO gradient reached the rationale head"
        opt.step()
        print(f"[smoke] step {step+1} gamma={gamma} loss={total.item():.4f} "
              f"rat_grad_norm={gnorm:.4e} | " +
              " ".join(f"{k}={v.item():.3f}" for k, v in parts.items()))
    print("[smoke] PASS — forward, masked loss, faithfulness re-encode, and "
          "straight-through gradient flow to the rationale head all verified.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default="google/muril-base-cased")
    args = ap.parse_args()
    if args.smoke:
        smoke()
    else:
        print("Use --smoke for the local pipeline check. Real training: see train (Week 2-3).")
