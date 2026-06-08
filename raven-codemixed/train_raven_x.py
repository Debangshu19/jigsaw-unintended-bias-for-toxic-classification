"""RAVEN-X trainer — multi-corpus, two-stage, faithfulness-regularized.

Local smoke (CPU, tiny encoder, real MuRIL tokenizer):
    raven-api/.venv/bin/python raven-codemixed/train_raven_x.py --smoke
Real run (Kaggle/Colab T4):
    python train_raven_x.py --model google/muril-base-cased --epochs 4 --out ckpt_ravenx

Saves a checkpoint dir the Raven API can serve (encoder/ + heads.pt + raven_x.json).
Dependency-light: metrics computed by hand (no sklearn needed).
"""
from __future__ import annotations
import argparse, contextlib, json, os, random
import torch
import raven_x as rx
import data_loaders as dl


# ----------------------------------------------------------------------- metrics (no sklearn)
def _f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def macro_f1_binary(preds, golds):
    f = []
    for c in (0, 1):
        tp = sum(p == c and g == c for p, g in zip(preds, golds))
        fp = sum(p == c and g != c for p, g in zip(preds, golds))
        fn = sum(p != c and g == c for p, g in zip(preds, golds))
        f.append(_f1(tp, fp, fn))
    return sum(f) / 2


def token_f1(pred_mask, gold_mask):
    tp = sum(p and g for p, g in zip(pred_mask, gold_mask))
    fp = sum(p and not g for p, g in zip(pred_mask, gold_mask))
    fn = sum((not p) and g for p, g in zip(pred_mask, gold_mask))
    return _f1(tp, fp, fn)


# ------------------------------------------------------------------------------ batching
def stratified_batches(hi, en, bs, seed, max_batches=None):
    """Each batch = bs/2 HASOC (oversampled w/ replacement) + bs/2 HateXplain."""
    half = bs // 2
    rng = random.Random(seed)
    en = en[:]
    rng.shuffle(en)
    hi_pool = hi[:]
    rng.shuffle(hi_pool)
    n = len(en) // half
    if max_batches:
        n = min(n, max_batches)
    hp = 0
    for b in range(n):
        en_chunk = en[b * half:(b + 1) * half]
        hi_chunk = []
        for _ in range(half):
            if hp >= len(hi_pool):
                rng.shuffle(hi_pool); hp = 0
            hi_chunk.append(hi_pool[hp]); hp += 1
        yield ([rx.unify_hasoc(x) for x in hi_chunk] +
               [rx.unify_hatexplain(x) for x in en_chunk])


# ------------------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, tok, hi_val, en_val, max_len, device, cap=400):
    model.eval()
    # HASOC macro-F1
    preds, golds = [], []
    for i in range(0, min(len(hi_val), cap), 16):
        ex = [rx.unify_hasoc(x) for x in hi_val[i:i + 16]]
        bt = rx.encode_examples(ex, tok, max_len)
        z = model(bt["input_ids"].to(device), bt["attention_mask"].float().to(device))["z"]
        preds += (z > 0).long().tolist()
        golds += [e["z_star"] for e in ex]
    mf1 = macro_f1_binary(preds, golds)
    # HateXplain token-F1 (first-subword positions with a real rationale target)
    pm, gm = [], []
    en_r = [e for e in en_val if e["has_rationale"]][:cap]
    for i in range(0, len(en_r), 16):
        ex = [rx.unify_hatexplain(x) for x in en_r[i:i + 16]]
        bt = rx.encode_examples(ex, tok, max_len)
        a = model(bt["input_ids"].to(device), bt["attention_mask"].float().to(device))["a"]
        keep = bt["r_sub"] != rx.IGNORE
        pm += (torch.sigmoid(a.cpu())[keep] > 0.5).long().tolist()
        gm += bt["r_sub"][keep].tolist()
    tf1 = token_f1(pm, gm)
    model.train()
    return mf1, tf1


# ------------------------------------------------------------------------------ trainer
def build_model(name, tok, tiny, device):
    if tiny:
        from transformers import BertConfig, BertModel
        cfg = BertConfig(vocab_size=tok.vocab_size, hidden_size=64, num_hidden_layers=2,
                         num_attention_heads=2, intermediate_size=128,
                         max_position_embeddings=256)
        enc, hidden = BertModel(cfg), 64
    else:
        from transformers import AutoModel
        enc = AutoModel.from_pretrained(name)
        hidden = enc.config.hidden_size
    return rx.RavenXModel(enc, hidden).to(device)


def set_encoder_grad(model, on):
    for p in model.encoder.parameters():
        p.requires_grad = on


def save_ckpt(model, tok, args, max_len):
    """Disconnect-safe: overwrite the checkpoint after every epoch."""
    os.makedirs(args.out, exist_ok=True)
    if not args.tiny:
        model.encoder.save_pretrained(os.path.join(args.out, "encoder"))
        tok.save_pretrained(os.path.join(args.out, "encoder"))
    torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")},
               os.path.join(args.out, "heads.pt"))
    json.dump({"backbone": args.model, "hidden": model.z.in_features, "max_len": max_len,
               "split_seed": 42}, open(os.path.join(args.out, "raven_x.json"), "w"), indent=2)


def train(args):
    import time
    from transformers import AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda" and not args.tiny and not args.no_amp     # fp16 on a real GPU
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    autocast = ((lambda: torch.autocast("cuda", dtype=torch.float16))
                if use_amp else contextlib.nullcontext)
    tok = AutoTokenizer.from_pretrained("google/muril-base-cased" if args.tiny else args.model)
    model = build_model(args.model, tok, args.tiny, device)
    hi = dl.load_hasoc_hindi(); hx = dl.load_hatexplain()
    w = dict(alpha=0.5, beta=2.0, lam_s=0.02, lam_c=0.01)
    max_len = 64 if args.tiny else 128
    gamma_target = 0.5
    if args.tiny:
        epochs, cap = 1, 3
    elif args.fast:
        epochs, cap = args.epochs, 600          # quick real pass: cap steps/epoch
    else:
        epochs, cap = args.epochs, None

    def run_epoch(opt, tag, gamma_on, seed_off):
        batches = list(stratified_batches(hi["train"], hx["train"], args.bs,
                                          args.seed + seed_off, cap))
        t0 = time.time()
        for bi, ex in enumerate(batches):
            bt = {k: v.to(device) for k, v in rx.encode_examples(ex, tok, max_len).items()}
            gamma = gamma_target * min(1.0, bi / max(1, len(batches) // 2)) if gamma_on else 0.0
            opt.zero_grad()
            with autocast():
                total, parts = rx.compute_loss(model, bt, w, gamma)
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            if bi % max(1, len(batches) // 4) == 0 or args.tiny:
                print(f"  {tag} batch {bi+1}/{len(batches)} gamma={gamma:.2f} "
                      f"loss={total.item():.4f} rat={parts['rat'].item():.3f} "
                      f"comp={parts['comp'].item():.3f}", flush=True)
        print(f"  [{tag}] {len(batches)} batches in {time.time()-t0:.0f}s", flush=True)

    print(f"[train] device={device} amp={use_amp} mode={'fast' if args.fast else 'full'} "
          f"model={'tiny' if args.tiny else args.model}")
    set_encoder_grad(model, False)
    opt0 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                             lr=1e-3 if args.tiny else 1e-4)
    print("[train] Stage 0 — freeze encoder, warm heads (gamma=0)")
    run_epoch(opt0, "stage0", gamma_on=False, seed_off=0)
    set_encoder_grad(model, True)
    opt1 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                             lr=1e-3 if args.tiny else 2e-5)
    print("[train] Stage 1 — unfreeze; faithfulness " +
          ("on FINAL epoch only (fast)" if args.fast else "with gamma ramp every epoch"))
    for ep in range(epochs):
        gamma_on = (ep == epochs - 1) if args.fast else True
        run_epoch(opt1, f"stage1.{ep+1}", gamma_on=gamma_on, seed_off=ep + 1)
        mf1, tf1 = evaluate(model, tok, hi["val"], hx["val"], max_len, device,
                            cap=64 if args.tiny else 400)
        print(f"[train] epoch {ep+1}/{epochs}: HASOC macro-F1={mf1:.3f}  "
              f"HateXplain token-F1={tf1:.3f}  (faithfulness={'on' if gamma_on else 'off'})")
        save_ckpt(model, tok, args, max_len)
        print(f"[train] checkpoint saved -> {args.out}/  (epoch {ep+1}, disconnect-safe)")
    print("[train] done.")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny encoder + 3 batches, CPU")
    ap.add_argument("--model", default="google/muril-base-cased")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="raven-codemixed/ckpt_ravenx")
    ap.add_argument("--fast", action="store_true",
                    help="quick real pass: cap steps/epoch, faithfulness on final epoch only")
    ap.add_argument("--no-amp", dest="no_amp", action="store_true",
                    help="disable fp16 mixed precision")
    a = ap.parse_args()
    a.tiny = a.smoke
    if a.smoke:
        a.epochs, a.out = 1, "/tmp/ravenx_smoke_ckpt"
    train(a)
