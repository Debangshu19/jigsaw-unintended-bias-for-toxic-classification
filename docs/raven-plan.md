# Raven End-to-End Plan

## Product

Raven identifies risky, toxic, or hateful comments and helps a reviewer decide what needs attention. The first usable product should be:

1. A website landing page with a working classifier demo.
2. A backend inference API that loads a real local Raven model.
3. A Chrome extension that scans visible comments and highlights risky ones.
4. Later, a mobile app that uses the same backend and design language.

## Model Choice

Best practical first model: `distilbert-base-uncased` fine-tuned on the Jigsaw toxicity data already used in the notebooks.

Why this path:

- It is small enough for a college project and practical demos.
- It can be fine-tuned with the notebooks already in this repo.
- It can be exported as a Hugging Face model and served locally.
- It is more defensible than pretending an external API is a custom model.

Recommended variants:

- Baseline: TF-IDF + Logistic Regression, fast and explainable.
- Main Raven model: fine-tuned DistilBERT, about 66M parameters before quantization.
- Browser-local future path: ONNX-quantized DistilBERT with Transformers.js.

## Honest API Position

The public product can say "Raven engine" or "Raven model" if the system is running a Raven-owned fine-tuned model. If an external API is used later as a fallback, it should not be falsely described as a proprietary model. The UI does not need to expose implementation details, but documentation and reports should stay accurate.

## Model Export

The scriptable path is now `raven-model/train_distilbert.py`. Run it against the Jigsaw CSV:

```bash
cd raven-model
pip install -r requirements.txt
python train_distilbert.py \
  --train-csv /path/to/train.csv \
  --text-column comment_text \
  --label-column target \
  --output-dir ../models/raven-distilbert
```

Then run the API with:

```bash
cd raven-api
export RAVEN_MODEL_DIR=/path/to/models/raven-distilbert
export RAVEN_THRESHOLD=0.5
uvicorn app:app --reload --port 8000
```

## API Contract

`GET /health` returns the current source.

`GET /metadata` returns the threshold and endpoint list.

`POST /predict`

```json
{
  "text": "Comment text here"
}
```

Response:

```json
{
  "label": "review",
  "score": 0.82,
  "needs_review": true,
  "source": "raven-local-model"
}
```

`POST /predict-batch` accepts `{ "texts": ["..."] }`.

## Extension Flow

1. Content script finds visible comment-like nodes.
2. It sends text batches to `http://localhost:8000/predict-batch`.
3. It highlights nodes where `needs_review` is true.
4. The popup shows API status, last scan counts, can trigger a manual rescan, and can test one typed comment with Enter.
5. If the API is unavailable, it uses the demo fallback only for UI testing.

## Website Demo Flow

1. The Raven Lab quick input scans one typed comment when Enter is pressed.
2. The textarea splits each line as one candidate comment for batch review.
3. The website tries `POST http://localhost:8000/predict-batch`.
4. If `raven-api` responds, the result panel shows the model/API source.
5. If not, the result panel shows `browser-demo-fallback` so the demo remains usable without pretending a real model is running.

## Mobile App

The first Expo scaffold now lives in `raven-mobile`:

- React Native/Expo app shell.
- Type one comment with submit/Enter or paste comments, scan through `/predict-batch`, and render a review queue.
- Uses the same local fallback approach as web/extension when the API is offline.
- Next screens to add: onboarding, link scan, settings, and model confidence.

## Immediate Build Order

1. Website with Raven assets and demo scanner.
2. FastAPI service that loads exported DistilBERT.
3. Extension that calls the service.
4. Expo mobile shell with the same scan contract.
5. Export model from notebooks or `raven-model/train_distilbert.py` and test real inference.
6. Quantize or optimize only after the end-to-end demo works.
