from pydantic import BaseModel, Field
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from model import load_model


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)


app = FastAPI(title="Raven API", version="0.1.0")
model = load_model()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "ok": True,
        "source": getattr(model, "source", "unknown"),
        "model_ref": getattr(model, "model_ref", None),
    }


@app.get("/metadata")
def metadata():
    return {
        "name": "Raven",
        "source": getattr(model, "source", "unknown"),
        "model_ref": getattr(model, "model_ref", None),
        "threshold": getattr(model, "threshold", None),
        "endpoints": ["/health", "/metadata", "/predict", "/predict-batch"],
    }


@app.post("/predict")
def predict(payload: PredictRequest):
    return model.predict_one(payload.text).__dict__


@app.post("/predict-batch")
def predict_batch(payload: BatchPredictRequest):
    predictions = model.predict_batch(payload.texts)
    return {"predictions": [prediction.__dict__ for prediction in predictions]}
