"""
api.py - FastAPI serving endpoint for the Apple Segment Classifier.

Endpoints:
    GET  /health           — returns model status
    POST /predict          — returns segment prediction and personalised content
    POST /predict-revenue  — returns predicted revenue in USD

Run locally (from project root):
    uvicorn 03-deployment.api:app --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from predict import load_pipeline, predict_segment
from predict_revenue import load_revenue_pipeline, predict_revenue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pipeline = None
revenue_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, revenue_pipeline
    logger.info("Loading segment pipeline...")
    pipeline = load_pipeline()
    logger.info("Loading revenue pipeline...")
    revenue_pipeline = load_revenue_pipeline()
    logger.info("All models ready.")
    yield


app = FastAPI(
    title="Apple Recommender API",
    description="Predicts customer segment and expected revenue for Apple homepage personalisation.",
    version="2.0.0",
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


class RevenueRequest(BaseModel):
    product_name:       str   = Field(..., example="iPhone 15")
    category:           str   = Field(..., example="iPhone")
    color:              str   = Field(..., example="Black")
    customer_age_group: str   = Field(..., example="25–34")
    region:             str   = Field(..., example="North America")
    country:            str   = Field(..., example="United States")
    city:               str   = Field(..., example="New York")
    sales_channel:      str   = Field(..., example="Online")
    payment_method:     str   = Field(..., example="Credit Card")
    discount_pct:       float = Field(..., example=0.1)
    units_sold:         int   = Field(..., example=1)


class RevenueResponse(BaseModel):
    revenue_usd: float


@app.get("/health")
def health():
    """Health check — confirms the API is running and both models are loaded."""
    return {
        "status": "ok",
        "segment_model_loaded": pipeline is not None,
        "revenue_model_loaded": revenue_pipeline is not None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Predict the customer segment from visitor inputs."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Segment model not loaded.")
    try:
        return predict_segment(pipeline, request.model_dump())
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-revenue", response_model=RevenueResponse)
def predict_revenue_endpoint(request: RevenueRequest):
    """Predict expected revenue in USD for a given sales scenario."""
    if revenue_pipeline is None:
        raise HTTPException(status_code=503, detail="Revenue model not loaded.")
    try:
        return predict_revenue(revenue_pipeline, request.model_dump())
    except Exception as e:
        logger.error(f"Revenue prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))