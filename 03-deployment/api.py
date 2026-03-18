"""
api.py - FastAPI serving endpoint for the Apple Segment Classifier.

Endpoints:
    GET  /health   — returns model status
    POST /predict  — returns segment prediction and personalised content

Run locally (from project root):
    uvicorn 03-deployment.api:app --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from predict import load_pipeline, predict_segment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Loading segment pipeline...")
    pipeline = load_pipeline()
    logger.info("Model ready.")
    yield


app = FastAPI(
    title="Apple Segment Classifier API",
    description="Predicts customer segment for Apple homepage personalisation.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    product_name:       str = Field(..., example="iPhone 15")
    category:           str = Field(..., example="iPhone")
    color:              str = Field(..., example="Black")
    customer_age_group: str = Field(..., example="25–34")
    region:             str = Field(..., example="North America")
    country:            str = Field(..., example="United States")
    city:               str = Field(..., example="New York")


class PredictResponse(BaseModel):
    segment:       str
    probabilities: dict
    content:       dict


@app.get("/health")
def health():
    """Health check — confirms the API is running and the model is loaded."""
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Predict the customer segment from visitor inputs."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        return predict_segment(pipeline, request.model_dump())
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
