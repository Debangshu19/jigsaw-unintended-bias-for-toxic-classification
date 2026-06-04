# Raven

Raven is an end-to-end hate-speech and toxic-comment detection project.

This repo now contains:

- `raven-web`: a Vite/React website and demo UI inspired by the Finny reference layout.
- `raven-mobile`: an Expo mobile app scaffold that scans pasted comments through the same API flow.
- `raven-api`: a FastAPI inference service that can load a fine-tuned local DistilBERT model.
- `raven-model`: scripts for fine-tuning and exporting a Raven DistilBERT checkpoint from CSV data.
- `extension`: a Chrome Manifest V3 prototype that highlights risky comments on supported pages.
- `docs/raven-plan.md`: model, website, mobile app, and extension plan.

## Train And Export The Model

Use the Jigsaw CSV or another labeled toxicity CSV with a text column and target column:

```bash
cd raven-model
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_distilbert.py \
  --train-csv /path/to/train.csv \
  --text-column comment_text \
  --label-column target \
  --output-dir ../models/raven-distilbert
```

## Run The Website

```bash
cd raven-web
npm install
npm run dev
```

The website demo calls `http://localhost:8000/predict-batch` first. If the API is not running, it falls back to the browser demo scorer and shows that source in the UI.

## Run The Mobile App

```bash
cd raven-mobile
npm install
npm run start
```

Set `EXPO_PUBLIC_RAVEN_API_URL` if the mobile device cannot reach `http://127.0.0.1:8000`.

## Run The API

```bash
cd raven-api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

If `RAVEN_MODEL_DIR` points to an exported Hugging Face model folder, the API uses it. Otherwise it runs a clearly marked demo fallback.

```bash
export RAVEN_MODEL_DIR=/Users/niladri/Documents/mine/clgFinalYear/models/raven-distilbert
export RAVEN_THRESHOLD=0.5
uvicorn app:app --reload --port 8000
```

Smoke test:

```bash
cd raven-api
source .venv/bin/activate
python smoke_test.py
```

## Load The Extension

1. Open Chrome or Edge extension settings.
2. Enable developer mode.
3. Load unpacked extension from `extension/`.
4. Keep `raven-api` running on port `8000`.
5. Use the Raven popup to check API status and manually scan the current page.
