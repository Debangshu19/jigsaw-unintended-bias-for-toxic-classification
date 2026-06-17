#!/usr/bin/env python3
"""Generate two figure SVGs with exact values / standard formulas:
  - diagram-stage1-rnn.svg : bar chart of the six classical/RNN bias-aware AUCs (exact, measured)
  - diagram-bias-variance.svg : illustrative bias-variance trade-off curves (standard theory)
Numbers in the bar chart come straight from FACTS.md (Regime A)."""
import math
from pathlib import Path

ASSET = Path(__file__).resolve().parent.parent / "assets"

PANEL, STROKE, INK, MUTED = "#11141b", "#2c3340", "#e8eaf0", "#97a1b2"
BLUE, GREEN, AMBER, VIOLET, RED = "#5b8cff", "#3ddc97", "#ffb454", "#b48cff", "#ff6b6b"
FONT = '-apple-system, Helvetica, Arial, sans-serif'

# ---------------------------------------------------------------- bar chart
def bar_chart():
    # (label lines, value, colour)  — EXACT bias-aware AUC, FACTS section 2
    data = [
        (["TF-IDF +", "LogReg"], 0.9061, MUTED),
        (["Single", "LSTM"], 0.8927, BLUE),
        (["BiLSTM"], 0.8992, BLUE),
        (["Weighted", "BiLSTM"], 0.9056, BLUE),
        (["BiGRU +", "Attn"], 0.9181, GREEN),
        (["BiGRU-Conv1D", "+ Attn"], 0.9190, GREEN),
    ]
    W, H = 1000, 590
    L, R, T, B = 140, 960, 95, 450
    ymin, ymax = 0.88, 0.92
    def ypix(v): return B - (v - ymin) / (ymax - ymin) * (B - T)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}">']
    s.append(f'<rect x="1" y="1" width="{W-2}" height="{H-2}" rx="16" fill="{PANEL}" stroke="{STROKE}"/>')
    s.append(f'<text x="40" y="46" fill="{INK}" font-size="21" font-weight="700">Classical &amp; RNN baselines &#8212; Jigsaw bias-aware AUC</text>')
    s.append(f'<text x="40" y="70" fill="{MUTED}" font-size="12.5">Higher is better. y-axis zoomed to 0.88&#8211;0.92 so the small gaps are visible. Regime A; measured by us.</text>')
    # gridlines + y labels
    v = ymin
    while v <= ymax + 1e-9:
        y = ypix(v)
        s.append(f'<line x1="{L}" y1="{y:.1f}" x2="{R}" y2="{y:.1f}" stroke="{STROKE}" stroke-dasharray="3 4"/>')
        s.append(f'<text x="{L-12}" y="{y+4:.1f}" fill="{MUTED}" font-size="12" text-anchor="end">{v:.2f}</text>')
        v = round(v + 0.01, 2)
    n = len(data)
    slot = (R - L) / n
    bw = 86
    for i, (lines, val, col) in enumerate(data):
        cx = L + slot * (i + 0.5)
        y = ypix(val)
        s.append(f'<rect x="{cx-bw/2:.1f}" y="{y:.1f}" width="{bw}" height="{B-y:.1f}" rx="6" fill="{col}" fill-opacity="0.85" stroke="{col}"/>')
        s.append(f'<text x="{cx:.1f}" y="{y-9:.1f}" fill="{INK}" font-size="13.5" font-weight="700" text-anchor="middle">{val:.4f}</text>')
        for j, ln in enumerate(lines):
            s.append(f'<text x="{cx:.1f}" y="{B+22+j*15:.1f}" fill="{MUTED}" font-size="11.5" text-anchor="middle">{ln}</text>')
    # band annotation
    s.append(f'<text x="{R}" y="{T-12}" fill="{GREEN}" font-size="12" text-anchor="end">the whole family sits in a ~0.89&#8211;0.92 band</text>')
    # legend
    lx = 40
    for lab, col in [("classical", MUTED), ("LSTM family", BLUE), ("GRU + attention", GREEN)]:
        s.append(f'<rect x="{lx}" y="{H-30}" width="13" height="13" rx="3" fill="{col}"/>')
        s.append(f'<text x="{lx+19}" y="{H-19}" fill="{MUTED}" font-size="12">{lab}</text>')
        lx += 24 + len(lab) * 7.3 + 26
    s.append('</svg>')
    (ASSET / "diagram-stage1-rnn.svg").write_text("\n".join(s), encoding="utf-8")
    print("wrote diagram-stage1-rnn.svg")

# ------------------------------------------------------ bias-variance curves
def bias_variance():
    W, H = 1000, 590
    L, R, T, B = 110, 900, 95, 470
    def xpix(t): return L + t * (R - L)
    def ypix(e): return B - e * (B - T) / 0.85
    bias = lambda t: 0.70 * math.exp(-3 * t) + 0.03
    var = lambda t: 0.05 + 0.60 * t * t
    tot = lambda t: bias(t) + var(t)
    def path(fn, col, w=3, dash=""):
        pts = [f"{xpix(i/60):.1f},{ypix(fn(i/60)):.1f}" for i in range(61)]
        d = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" stroke-width="{w}"{d} stroke-linejoin="round"/>'
    # sweet spot (numeric min of total)
    ts = min((i/200 for i in range(201)), key=tot)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}">']
    s.append(f'<rect x="1" y="1" width="{W-2}" height="{H-2}" rx="16" fill="{PANEL}" stroke="{STROKE}"/>')
    s.append(f'<text x="40" y="46" fill="{INK}" font-size="21" font-weight="700">Why a bigger model is not automatically better</text>')
    s.append(f'<text x="40" y="70" fill="{MUTED}" font-size="12.5">Illustrative (standard bias-variance theory) &#8212; not our measurements.</text>')
    # axes
    s.append(f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" stroke="{MUTED}" stroke-width="1.5"/>')
    s.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{B}" stroke="{MUTED}" stroke-width="1.5"/>')
    s.append(f'<text x="{(L+R)/2:.0f}" y="{B+54}" fill="{INK}" font-size="13.5" text-anchor="middle">model capacity  (LSTM &#8594; DistilBERT &#8594; BERT-large)  &#8594;</text>')
    s.append(f'<text x="34" y="{(T+B)/2:.0f}" fill="{INK}" font-size="13.5" text-anchor="middle" transform="rotate(-90 34 {(T+B)/2:.0f})">expected error &#8594;</text>')
    # sweet-spot marker
    xs = xpix(ts)
    s.append(f'<line x1="{xs:.1f}" y1="{T}" x2="{xs:.1f}" y2="{B}" stroke="{GREEN}" stroke-width="1.5" stroke-dasharray="4 4"/>')
    s.append(f'<text x="{xs:.1f}" y="{T-6}" fill="{GREEN}" font-size="12" text-anchor="middle">best capacity</text>')
    # under/over labels
    s.append(f'<text x="{L+70}" y="{B-8}" fill="{MUTED}" font-size="12">underfitting</text>')
    s.append(f'<text x="{R-70}" y="{B-8}" fill="{MUTED}" font-size="12" text-anchor="end">overfitting</text>')
    # curves
    s.append(path(bias, BLUE))
    s.append(path(var, AMBER))
    s.append(path(tot, INK, 3.4))
    # curve labels
    s.append(f'<text x="{xpix(0.62):.1f}" y="{ypix(bias(0.62))-8:.1f}" fill="{BLUE}" font-size="13" font-weight="600">bias&#178; (drops as model grows)</text>')
    s.append(f'<text x="{xpix(0.80):.1f}" y="{ypix(var(0.80))+4:.1f}" fill="{AMBER}" font-size="13" font-weight="600" text-anchor="end">variance (rises)</text>')
    s.append(f'<text x="{xpix(0.06):.1f}" y="{ypix(tot(0.06))-10:.1f}" fill="{INK}" font-size="13" font-weight="700">total error</text>')
    # model markers on the total curve
    for t, name, col in [(0.30, "DistilBERT (small)", GREEN), (0.72, "BERT-large (more capacity)", VIOLET)]:
        x, y = xpix(t), ypix(tot(t))
        s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{col}" stroke="#0b0d12" stroke-width="2"/>')
        anchor = "start" if t < 0.5 else "end"
        dx = 12 if t < 0.5 else -12
        s.append(f'<text x="{x+dx:.1f}" y="{y-12:.1f}" fill="{col}" font-size="12.5" font-weight="600" text-anchor="{anchor}">{name}</text>')
    s.append(f'<text x="{(L+R)/2:.0f}" y="{H-26}" fill="{MUTED}" font-size="12" text-anchor="middle">A single larger encoder moves right: it lowers bias but adds variance &#8212; so it raises the ceiling, it does not remove single-model error.</text>')
    s.append('</svg>')
    (ASSET / "diagram-bias-variance.svg").write_text("\n".join(s), encoding="utf-8")
    print("wrote diagram-bias-variance.svg")

if __name__ == "__main__":
    bar_chart()
    bias_variance()
