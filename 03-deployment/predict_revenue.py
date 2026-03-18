"""
predict_revenue.py - Shared prediction logic for the revenue model.
Import this module wherever you need to make a revenue prediction.
"""

import logging
from pathlib import Path

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_ORDER = [
    "product_name", "category", "color", "customer_age_group",
    "region", "country", "city", "sales_channel", "payment_method",
    "discount_pct", "units_sold",
]


def load_revenue_pipeline(artifact_path="models/revenue_pipeline.pkl"):
    """Load the trained revenue pipeline from disk."""
    path = Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Revenue model not found at '{artifact_path}'. Run train_revenue.py first."
        )
    pipeline = joblib.load(path)
    logger.info(f"Revenue pipeline loaded from {artifact_path}")
    return pipeline


def predict_revenue(pipeline, input_data: dict) -> dict:
    """
    Predict expected revenue for a given sales scenario.

    Args:
        pipeline: Trained sklearn Pipeline (preprocessor + regressor).
        input_data: Dict with keys matching FEATURE_ORDER.

    Returns:
        Dict with 'revenue_usd'.
    """
    df = pd.DataFrame([input_data])[FEATURE_ORDER]
    predicted = pipeline.predict(df)[0]
    return {"revenue_usd": round(float(predicted), 2)}
