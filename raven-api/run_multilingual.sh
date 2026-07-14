#!/bin/bash
# Start the Raven API with English (toxic-bert) + Hindi (our MuRIL) tracks.
# Devanagari comments route to MuRIL; everything else to the English model.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"

source "$DIR/.venv/bin/activate"

export RAVEN_MODEL_ID="${RAVEN_MODEL_ID:-unitary/toxic-bert}"
export RAVEN_HI_MODEL_DIR="${RAVEN_HI_MODEL_DIR:-$ROOT/models/raven-muril-hi}"
export RAVEN_THRESHOLD="${RAVEN_THRESHOLD:-0.5}"
export TOKENIZERS_PARALLELISM=false

echo "English model : $RAVEN_MODEL_ID"
echo "Hindi model   : $RAVEN_HI_MODEL_DIR"
if [ ! -f "$RAVEN_HI_MODEL_DIR/config.json" ]; then
  echo "WARNING: Hindi model not found — train it first (raven-codemixed/train_muril_hi.py)."
fi

cd "$DIR"
exec uvicorn app:app --host 0.0.0.0 --port 8000
