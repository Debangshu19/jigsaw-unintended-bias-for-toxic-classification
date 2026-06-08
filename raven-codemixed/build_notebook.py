"""Generate a SELF-CONTAINED Colab/Kaggle notebook for the real RAVEN-X run.

It embeds the exact (locally smoke-tested) modules via %%writefile cells, stages the
data, runs the real TF-IDF baseline, then trains the real MuRIL model on a free T4.
No repo access or manual upload needed — open in Colab, choose T4, Run all.

    raven-api/.venv/bin/python raven-codemixed/build_notebook.py
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))


def code(src):
    lines = src.splitlines(keepends=True)
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": lines}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def writefile(fname):
    body = open(os.path.join(HERE, fname), encoding="utf-8").read()
    return code(f"%%writefile {fname}\n" + body)


cells = [
    md("""# RAVEN-X — real run on a free GPU (Colab/Kaggle T4)

**Cross-script rationale transfer for Hindi hate speech.** This notebook is self-contained:
it writes its own code, stages the data, runs a real baseline, and trains the real MuRIL model.

### Do this first
1. **Runtime → Change runtime type → T4 GPU** (Colab) or **Settings → Accelerator → GPU T4** (Kaggle).
2. **Runtime → Run all.** Total ≈ **25–45 min** in fast mode (fp16). Everything below is real — no fabrication.

> Data (HASOC-2019, HateXplain) is **research-only**; this notebook keeps it in the session and
> never commits it. Cite Mandl et al. 2019, Mathew et al. 2021, MuRIL (Khanuja et al. 2021)."""),
    md("## 1 · Confirm the GPU is attached"),
    code("import torch\n"
         "print('CUDA available:', torch.cuda.is_available())\n"
         "print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU "
         "(set Runtime->T4 GPU!)')"),
    md("## 2 · Install dependencies\n"
       "> Colab already ships a **CUDA-matched torch + torchvision + transformers**. We do **NOT** "
       "upgrade torch — doing so breaks torchvision and makes `transformers` fail to import. "
       "We only ensure scikit-learn/scipy (used by the baseline) are present."),
    code("# Do NOT upgrade torch on Colab (breaks torchvision -> transformers import).\n"
         "!pip -q install scikit-learn scipy\n"
         "import torch, transformers\n"
         "print('torch', torch.__version__, '| transformers', transformers.__version__,\n"
         "      '| CUDA', torch.cuda.is_available(),\n"
         "      '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"),
    md("## 3 · Stage the data (research-only; stays in the session)"),
    code("import os, subprocess, urllib.request\n"
         "os.makedirs('data/hasoc2019', exist_ok=True)\n"
         "os.makedirs('data/hatexplain', exist_ok=True)\n"
         "# HASOC-2019 open mirror -> copy the Hindi tsv (labeled) and blind test\n"
         "subprocess.run(['git','clone','--depth','1',\n"
         "  'https://github.com/TharinduDR/HASOC-2019','/tmp/hasoc'], check=True)\n"
         "for f in ['hindi_dataset.tsv','hasoc2019_hi_test.tsv']:\n"
         "    subprocess.run(['cp', f'/tmp/hasoc/data/{f}', f'data/hasoc2019/{f}'], check=True)\n"
         "# HateXplain (rationales)\n"
         "base='https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/'\n"
         "for f in ['dataset.json','post_id_divisions.json']:\n"
         "    urllib.request.urlretrieve(base+f, f'data/hatexplain/{f}')\n"
         "print('staged:', os.listdir('data/hasoc2019'), os.listdir('data/hatexplain'))"),
    md("## 4 · Write the RAVEN-X code (identical to the locally smoke-tested modules)"),
    writefile("data_loaders.py"),
    writefile("raven_x.py"),
    writefile("train_raven_x.py"),
    writefile("baselines.py"),
    md("## 5 · Sanity: audit the data + verify the pipeline (fast)"),
    code("!python data_loaders.py\n"
         "!python raven_x.py --smoke"),
    md("## 6 · REAL baseline — TF-IDF + LogReg (the benchmark lower bound)"),
    code("!python baselines.py"),
    md("## 7 · REAL training — fine-tune MuRIL (the actual model, on the T4)\n"
       "**Fast mode (~25–45 min):** fp16 + capped steps/epoch + faithfulness on the final epoch. "
       "Prints HASOC macro-F1 + HateXplain token-F1 each epoch and **saves a checkpoint every "
       "epoch (disconnect-safe).**\n\n"
       "For the full paper-grade run later, drop `--fast` and use `--epochs 4` (~1.5 h)."),
    code("!python train_raven_x.py --model google/muril-base-cased --fast --epochs 3 "
         "--out ckpt_ravenx\n"
         "# full run later:  !python train_raven_x.py --model google/muril-base-cased "
         "--epochs 4 --out ckpt_ravenx"),
    md("## 8 · Save your results\n"
       "Download the checkpoint + zip it from the file browser, or push to Google Drive:\n"
       "```python\n"
       "from google.colab import drive; drive.mount('/content/drive')\n"
       "!cp -r ckpt_ravenx /content/drive/MyDrive/\n"
       "```\n"
       "Screenshot the epoch macro-F1 / token-F1 lines — **those are your real, defensible "
       "numbers** for the report and viva."),
]

nb = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
    "accelerator": "GPU", "colab": {"provenance": []}},
    "nbformat": 4, "nbformat_minor": 5}

out = os.path.join(HERE, "ravenx_colab.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out, "—", len(cells), "cells")
