#!/usr/bin/env python3
"""Assemble the Raven presentation into one self-contained HTML, injecting the inline-SVG
diagrams into their <div class="figure" id="fig-*"> placeholders.

Usage:
  assemble.py [theme] [layout] [outfile] [assetsub]
    theme   : dark (default) | light
    layout  : slides (default, one section per page) | flow (continuous, less whitespace)
    outfile : output html name (default raven-presentation.html)
    assetsub: sub-folder of assets/ to pull SVGs from ('' = assets/, 'light' = assets/light/)
"""
import re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECT = ROOT / "sections"
ASSET = ROOT / "assets"

ORDER = [
    "00-cover.html", "02-overview.html", "03-how-we-measure.html", "04-stage1-rnn.html",
    "05-stage2-distilbert.html", "06-stage3-bigger.html", "07-stage4-ensemble.html",
    "08-final-model.html", "09-ravenx.html", "10-roadmap.html", "99-references.html",
]
TOC = [
    ("01", "What this project is", "overview"),
    ("02", "Data and how we measure", "measure"),
    ("03", "Stage 1 — Classical & RNN models (LSTM family)", "stage1"),
    ("04", "Stage 2 — DistilBERT: a fine-tuned transformer", "stage2"),
    ("05", "Stage 3 — Scaling up to a larger transformer", "stage3"),
    ("06", "Stage 4 — Ensemble learning by simple averaging", "stage4"),
    ("07", "The final model — what it does and where it stops", "final"),
    ("08", "RAVEN-X — our original Hindi method", "ravenx"),
    ("09", "What we need next", "roadmap"),
    ("10", "References", "references"),
]
FIG2SVG = {
    "fig-system": "diagram-system.svg", "fig-stage1-rnn": "diagram-stage1-rnn.svg",
    "fig-distilbert": "diagram-distilbert.svg", "fig-bias-variance": "diagram-bias-variance.svg",
    "fig-ensemble": "diagram-ensemble.svg", "fig-ravenx": "diagram-ravenx.svg",
}

TOKENS = {
    "dark": {
        "bg": "#0b0d12", "panel": "#161a23", "panel2": "#1d222e", "stroke": "#2c3340",
        "ink": "#e8eaf0", "inkstrong": "#ffffff", "muted": "#97a1b2",
        "blue": "#5b8cff", "green": "#3ddc97", "amber": "#ffb454", "violet": "#b48cff",
        "thbg": "#1a2440", "think": "#dbe4ff", "rowalt": "#12161e", "softline": "#20262f",
        "codeink": "#d7e0ee", "foot": "#5b647a", "footline": "#181d26", "lede": "#cfd6e4",
        "cardh": "#dbe4ff", "mi": "#7ef0c0", "li": "#cdb4ff", "pi": "#ffce8f",
        "slidebg": "linear-gradient(180deg,#0b0d12 0%,#0c0f15 100%)",
    },
    "light": {
        "bg": "#ffffff", "panel": "#f6f8fc", "panel2": "#eef2f8", "stroke": "#dde3ec",
        "ink": "#1f242c", "inkstrong": "#0b0e14", "muted": "#5a6473",
        "blue": "#2f5fe0", "green": "#0e9266", "amber": "#a96a09", "violet": "#6a3fc0",
        "thbg": "#1f3a66", "think": "#ffffff", "rowalt": "#f4f6fb", "softline": "#e7ebf2",
        "codeink": "#2a2f3a", "foot": "#8a93a3", "footline": "#e8ebf1", "lede": "#39424f",
        "cardh": "#1f3a66", "mi": "#0a7a52", "li": "#5a36a8", "pi": "#9a5a06",
        "slidebg": "#ffffff",
    },
}


def css(theme, layout):
    t = TOKENS[theme]
    flow = layout == "flow"
    slide_geom = ("min-height:auto; break-before:auto; padding:14mm 16mm 13mm; border-bottom:1px solid var(--stroke);"
                  if flow else "min-height:297mm; break-before:page; padding:20mm 18mm 22mm;")
    extra_flow = (".slide:first-child{border-bottom:none;} .cover{min-height:250mm;border-bottom:none;}"
                  "#references{break-before:page;} h2,h3{break-after:avoid;}"
                  if flow else "")
    return f"""<style>
:root{{
  --bg:{t['bg']}; --panel:{t['panel']}; --panel2:{t['panel2']}; --stroke:{t['stroke']};
  --ink:{t['ink']}; --inkstrong:{t['inkstrong']}; --muted:{t['muted']};
  --blue:{t['blue']}; --green:{t['green']}; --amber:{t['amber']}; --violet:{t['violet']};
  --thbg:{t['thbg']}; --think:{t['think']}; --rowalt:{t['rowalt']}; --softline:{t['softline']};
  --codeink:{t['codeink']}; --foot:{t['foot']}; --footline:{t['footline']}; --lede:{t['lede']};
  --cardh:{t['cardh']}; --mi:{t['mi']}; --li:{t['li']}; --pi:{t['pi']};
}}
*{{box-sizing:border-box}}
@page{{ size:A4 portrait; margin:0; }}
html,body{{margin:0;padding:0;background:var(--bg);color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
  font-size:10.6pt; line-height:1.5; -webkit-print-color-adjust:exact; print-color-adjust:exact;}}
.slide{{ position:relative; {slide_geom} background:{t['slidebg']}; }}
.slide:first-child{{break-before:auto;}}
.slide::before{{ content:""; position:absolute; top:0; left:0; right:0; height:5px;
  background:linear-gradient(90deg,var(--blue),var(--violet) 55%,var(--green)); }}
{extra_flow}
.kicker{{ color:var(--blue); font-weight:700; letter-spacing:.16em; text-transform:uppercase;
  font-size:8.5pt; margin-bottom:6px; }}
h1{{ font-size:30pt; line-height:1.1; margin:.1em 0 .25em; letter-spacing:-.01em; color:var(--inkstrong);}}
h2{{ font-size:19pt; line-height:1.15; margin:.1em 0 .5em; letter-spacing:-.01em; color:var(--inkstrong);}}
h3{{ font-size:13.5pt; margin:1.1em 0 .3em; color:var(--inkstrong); }}
h4{{ font-size:11pt; margin:0 0 .35em; color:var(--inkstrong); }}
p{{ margin:.45em 0; }} b,strong{{color:var(--inkstrong);}}
.muted{{ color:var(--muted); }}
ul,ol{{ margin:.4em 0 .6em; padding-left:1.25em; }} li{{ margin:.22em 0; }}
a{{ color:var(--blue); text-decoration:none; }}
code{{ font-family:"SF Mono",ui-monospace,Menlo,Consolas,monospace; font-size:.92em; }}
table.data{{ width:100%; border-collapse:collapse; margin:.7em 0 1em; font-size:9.6pt; break-inside:avoid; }}
table.data th{{ background:var(--thbg); color:var(--think); text-align:left; font-weight:600;
  padding:7px 9px; border-bottom:1px solid var(--stroke); }}
table.data td{{ padding:6px 9px; border-bottom:1px solid var(--softline); }}
table.data tbody tr:nth-child(even){{ background:var(--rowalt); }}
table.data td b, table.data td strong{{ color:var(--inkstrong); }}
.keypoint,.note{{ border-radius:10px; padding:11px 14px; margin:.8em 0; break-inside:avoid; font-size:9.9pt;}}
.keypoint{{ background:rgba(91,140,255,.10); border:1px solid rgba(91,140,255,.32); border-left:4px solid var(--blue);}}
.note{{ background:rgba(255,180,84,.10); border:1px solid rgba(255,180,84,.30); border-left:4px solid var(--amber);}}
.math{{ background:var(--panel2); border:1px solid var(--stroke); border-radius:10px;
  padding:11px 14px; margin:.8em 0; font-family:"SF Mono",ui-monospace,Menlo,Consolas,monospace;
  font-size:9.3pt; line-height:1.65; color:var(--codeink); break-inside:avoid; }}
.cols{{ display:flex; gap:16px; margin:.6em 0; }} .col{{ flex:1; min-width:0; }}
.cards{{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:.7em 0; }}
.card{{ background:var(--panel); border:1px solid var(--stroke); border-radius:12px; padding:12px 14px; break-inside:avoid; }}
.card h4{{ color:var(--cardh); }} .card p{{ font-size:9.5pt; color:var(--muted); margin:.2em 0;}}
.tag{{ display:inline-block; font-size:7.6pt; font-weight:700; letter-spacing:.04em; text-transform:uppercase;
  padding:1px 7px; border-radius:999px; vertical-align:middle; white-space:nowrap; }}
.t-measured{{ background:rgba(61,220,151,.16); color:var(--mi); border:1px solid rgba(61,220,151,.45);}}
.t-lit{{ background:rgba(180,140,255,.16); color:var(--li); border:1px solid rgba(180,140,255,.45);}}
.t-proposed{{ background:rgba(255,180,84,.16); color:var(--pi); border:1px solid rgba(255,180,84,.45);}}
.figure{{ margin:1.1em auto; text-align:center; break-inside:avoid; }}
.figure svg{{ display:block; margin:0 auto; max-width:100%; max-height:232mm; height:auto; }}
.figure figcaption{{ color:var(--muted); font-size:8.8pt; font-style:italic; margin-top:8px; }}
.cover{{ display:flex; flex-direction:column; justify-content:center; }}
.cover-mark{{ position:absolute; top:16mm; right:18mm; font-weight:800; letter-spacing:.32em; color:var(--blue); opacity:.6; font-size:12pt;}}
.cover h1{{ font-size:40pt; max-width:16em; }}
.lede{{ font-size:13.5pt; color:var(--lede); max-width:34em; margin-top:.4em;}}
.thesis{{ color:var(--muted); max-width:38em; font-size:10pt; margin-top:1.1em; border-left:3px solid var(--stroke); padding-left:12px;}}
.cover-stats{{ display:flex; gap:18px; margin:2.2em 0 1.4em; flex-wrap:wrap;}}
.stat{{ background:var(--panel); border:1px solid var(--stroke); border-radius:14px; padding:14px 16px; min-width:185px; flex:1;}}
.stat-num{{ font-size:26pt; font-weight:800; color:var(--inkstrong); letter-spacing:-.02em;}}
.stat-lab{{ color:var(--muted); font-size:8.8pt; margin-top:4px; line-height:1.4;}}
.cover-foot{{ display:flex; gap:28px; color:var(--muted); font-size:10pt; margin-top:.6em;}}
.toc-list{{ margin-top:1.2em; }}
.toc-row{{ display:flex; align-items:baseline; gap:16px; padding:10px 4px; border-bottom:1px solid var(--softline); color:var(--ink);}}
.toc-n{{ color:var(--blue); font-weight:800; font-size:11pt; width:2em;}}
.toc-t{{ font-size:12pt; }}
.refs{{ font-size:9.7pt; }} .refs li{{ margin:.4em 0; color:var(--lede); }}
.runfoot{{ position:fixed; bottom:8mm; left:18mm; right:18mm; display:flex; justify-content:space-between;
  color:var(--foot); font-size:7.8pt; letter-spacing:.04em; border-top:1px solid var(--footline); padding-top:5px;}}
</style>"""


def inject_svgs(html, assetsub):
    base = (ASSET / assetsub) if assetsub else ASSET
    def repl(m):
        figid = m.group("id"); svgfile = FIG2SVG.get(figid)
        if not svgfile: return m.group(0)
        p = base / svgfile
        if not p.exists():
            sys.stderr.write(f"  [warn] missing svg for {figid}: {p}\n"); return m.group(0)
        svg = p.read_text(encoding="utf-8").strip()
        svg = re.sub(r"^<\?xml[^>]*\?>\s*", "", svg)
        return m.group(0) + "\n" + svg + "\n"
    return re.compile(r'<div\s+class="figure"\s+id="(?P<id>fig-[a-z0-9\-]+)"\s*>', re.I).subn(repl, html)[0]


def build_toc():
    rows = "\n".join(
        f'<a class="toc-row" href="#{a}"><span class="toc-n">{n}</span><span class="toc-t">{tt}</span></a>'
        for n, tt, a in TOC)
    return (f'<section class="slide toc" id="toc"><div class="kicker">Contents</div>'
            f'<h2>What this document covers</h2>'
            f'<p class="muted">We walk through the project once, then through every model we tried in order — '
            f'what it did, why it falls short (both the maths and what we actually saw), and how the next step fixes it — '
            f'ending with our own method and the full architecture diagrams.</p>'
            f'<div class="toc-list">{rows}</div></section>')


def build(theme="dark", layout="slides", out="raven-presentation.html", assetsub=""):
    parts = []
    for fn in ORDER:
        p = SECT / fn
        if not p.exists():
            sys.stderr.write(f"  [warn] missing section: {fn}\n"); continue
        parts.append(inject_svgs(p.read_text(encoding="utf-8"), assetsub))
        if fn == "00-cover.html":
            parts.append(build_toc())
    head = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<title>Raven — Toxic & Hate-Speech Detection</title>' + css(theme, layout) + '</head><body>'
            '<div class="runfoot"><span>Raven &mdash; final-year project &middot; toxic &amp; hate-speech detection</span>'
            '<span>Niladri Hazra &amp; team &middot; supervisor Dr. Arpita Dutta</span></div>\n')
    doc = head + "\n\n".join(parts) + "\n</body></html>\n"
    (ROOT / out).write_text(doc, encoding="utf-8")
    print(f"wrote {out}  (theme={theme}, layout={layout}, {len(doc):,} bytes)")


if __name__ == "__main__":
    a = sys.argv[1:]
    build(theme=a[0] if len(a) > 0 else "dark",
          layout=a[1] if len(a) > 1 else "slides",
          out=a[2] if len(a) > 2 else "raven-presentation.html",
          assetsub=a[3] if len(a) > 3 else "")
