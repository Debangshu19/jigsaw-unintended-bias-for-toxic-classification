#!/usr/bin/env python3
"""Produce light-theme copies of the six diagram SVGs (assets/light/) and rasterise
every light SVG to a 2x PNG (assets/png/) using headless Google Chrome — the PNGs are
what the editable .docx embeds."""
import re, subprocess, tempfile, os
from pathlib import Path

ASSET = Path(__file__).resolve().parent.parent / "assets"
LIGHT = ASSET / "light"; PNG = ASSET / "png"
LIGHT.mkdir(exist_ok=True); PNG.mkdir(exist_ok=True)
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

SVGS = ["diagram-system", "diagram-stage1-rnn", "diagram-distilbert",
        "diagram-bias-variance", "diagram-ensemble", "diagram-ravenx"]

# dark hex -> light hex (panels become near-white cards; text becomes dark; accents darkened for contrast on white)
MAP = {
    "#0b0d12": "#ffffff", "#11141b": "#f6f8fc", "#161a23": "#f6f8fc", "#1d222e": "#e9edf5",
    "#2c3340": "#ccd4e0", "#97a1b2": "#5a6473", "#e8eaf0": "#1b1f27",
    "#5b8cff": "#2f5fe0", "#3ddc97": "#0e9266", "#ffb454": "#a96a09",
    "#b48cff": "#6a3fc0", "#36d6c3": "#0e9b8c", "#ff6b6b": "#d23b3b",
}

def recolor(svg: str) -> str:
    for d, l in MAP.items():
        svg = svg.replace(d, l).replace(d.upper(), l)
    return svg

def dims(svg: str):
    w = re.search(r'<svg[^>]*\bwidth="(\d+)"', svg)
    h = re.search(r'<svg[^>]*\bheight="(\d+)"', svg)
    return int(w.group(1)) if w else 1000, int(h.group(1)) if h else 700

for base in SVGS:
    dark = (ASSET / f"{base}.svg").read_text(encoding="utf-8")
    light = recolor(dark)
    (LIGHT / f"{base}.svg").write_text(light, encoding="utf-8")
    w, h = dims(light)
    # wrap in a zero-margin white page so the screenshot is tight and on white
    html = f'<!doctype html><html><head><meta charset="utf-8"><style>*{{margin:0}}html,body{{background:#fff}}</style></head><body>{light}</body></html>'
    with tempfile.NamedTemporaryFile("w", suffix=".html", dir=str(PNG), delete=False) as f:
        f.write(html); tmp = f.name
    out = PNG / f"{base}.png"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    "--force-device-scale-factor=2", "--default-background-color=ffffffff",
                    f"--window-size={w},{h}", f"--screenshot={out}", tmp],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.unlink(tmp)
    print(f"  {base}: light svg + png ({w}x{h} -> {out.name})")

print("assets done")
