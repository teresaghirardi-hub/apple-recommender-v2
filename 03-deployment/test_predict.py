"""
test_predict.py - Unit tests for the prediction module.

Run with:
    pytest tests/
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

SAMPLE_INPUT = {
    "product_name":       "iPhone 15",
    "category":           "iPhone",
    "color":              "Black",
    "customer_age_group": "25–34",
    "region":             "North America",
    "country":            "United States",
    "city":               "New York",
}

VALID_SEGMENTS = {"Individual", "Business", "Education", "Government"}


def make_mock_pipeline(segment="Individual"):
    mock = MagicMock()
    mock.predict.return_value = [segment]
    mock.predict_proba.return_value = np.array([[0.7, 0.1, 0.1, 0.1]])
    mock.classes_ = ["Business", "Education", "Government", "Individual"]
    return mock


def test_predict_returns_valid_segment():
    from predict import predict_segment
    result = predict_segment(make_mock_pipeline(), SAMPLE_INPUT)
    assert result["segment"] in VALID_SEGMENTS


def test_predict_returns_four_probabilities():
    from predict import predict_segment
    result = predict_segment(make_mock_pipeline(), SAMPLE_INPUT)
    assert len(result["probabilities"]) == 4


def test_probabilities_sum_to_one():
    from predict import predict_segment
    result = predict_segment(make_mock_pipeline(), SAMPLE_INPUT)
    assert abs(sum(result["probabilities"].values()) - 1.0) < 0.01


def test_predict_returns_content():
    from predict import predict_segment
    result = predict_segment(make_mock_pipeline(), SAMPLE_INPUT)
    assert "headline" in result["content"]
    assert "products" in result["content"]
    assert "offer"    in result["content"]


def test_load_pipeline_raises_if_missing():
    from predict import load_pipeline
    with pytest.raises(FileNotFoundError):
        load_pipeline("models/nonexistent.pkl")


def test_segment_content_covers_all_segments():
    from predict import SEGMENT_CONTENT
    for segment in VALID_SEGMENTS:
        assert segment in SEGMENT_CONTENT