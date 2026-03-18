"""
train.py - Trains a Random Forest classifier for Apple customer segment prediction.
Logs all experiments with MLflow: baselines, grid search, confusion matrix.

Usage (from project root):
    python 02-experiment-tracking/train.py
"""

import argparse
import logging
import os

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import category_encoders as ce

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_and_prepare(cfg):
    df = pd.read_csv(cfg["data"]["raw_path"])
    logger.info(f"Loaded {len(df):,} rows")

    drop_cols = [c for c in cfg["features"]["drop_columns"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    feature_cols = (
        cfg["features"]["categorical_high_cardinality"]
        + cfg["features"]["categorical_ohe"]
        + cfg["features"]["ordinal"]
    )
    X = df[feature_cols]
    y = df[cfg["features"]["target"]]

    logger.info(f"Target distribution:\n{y.value_counts()}")
    return X, y


def build_preprocessor(cfg):
    high_card = cfg["features"]["categorical_high_cardinality"]
    ohe_cols = cfg["features"]["categorical_ohe"]
    ord_cols = cfg["features"]["ordinal"]
    ord_cats = [cfg["features"]["ordinal_categories"][c] for c in ord_cols]

    return ColumnTransformer(transformers=[
        ("target_enc", ce.TargetEncoder(cols=high_card), high_card),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ohe_cols),
        ("ordinal", OrdinalEncoder(
            categories=ord_cats,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        ), ord_cols),
    ], remainder="drop")


def log_baselines(X_train_t, y_train, X_test_t, y_test):
    """Log majority-class and logistic regression baselines to MLflow."""
    with mlflow.start_run(run_name="baseline_majority_class", nested=True):
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train_t, y_train)
        y_pred = dummy.predict(X_test_t)
        mlflow.log_metrics({
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        })
        logger.info(f"Baseline majority — Acc: {accuracy_score(y_test, y_pred):.4f}")

    with mlflow.start_run(run_name="baseline_logistic_regression", nested=True):
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train_t, y_train)
        y_pred = lr.predict(X_test_t)
        mlflow.log_metrics({
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        })
        logger.info(f"Baseline logistic — Acc: {accuracy_score(y_test, y_pred):.4f}")


def train_random_forest(X_train_t, y_train, X_test_t, y_test, cfg):
    """Grid-search Random Forest, log results and confusion matrix to MLflow."""
    params = cfg["model"]["hyperparams"]
    grid = GridSearchCV(
        RandomForestClassifier(random_state=cfg["data"]["random_state"]),
        params, cv=3, scoring="f1_weighted", n_jobs=-1, verbose=1,
    )

    with mlflow.start_run(run_name="random_forest_gridsearch", nested=True):
        grid.fit(X_train_t, y_train)
        best = grid.best_estimator_
        y_pred = best.predict(X_test_t)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics({"accuracy": acc, "f1_weighted": f1})

        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        mlflow.sklearn.log_model(best, artifact_path="random_forest")

        logger.info(f"Best RF — Acc: {acc:.4f} | F1: {f1:.4f}")
        logger.info(f"Best params: {grid.best_params_}")
        logger.info(f"\n{classification_report(y_test, y_pred)}")

    return best, acc, f1


def main(config_path="config.yaml"):
    cfg = load_config(config_path)
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    X, y = load_and_prepare(cfg)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
        stratify=y,
    )
    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    preprocessor = build_preprocessor(cfg)
    X_train_t = preprocessor.fit_transform(X_train, y_train)
    X_test_t = preprocessor.transform(X_test)

    # Save reference data for drift monitoring
    os.makedirs("data", exist_ok=True)
    pd.DataFrame(X_train_t).to_csv("data/reference.csv", index=False)
    logger.info("Reference data saved to data/reference.csv")

    with mlflow.start_run(run_name="apple_segment_classifier"):
        log_baselines(X_train_t, y_train, X_test_t, y_test)
        best_rf, acc, f1 = train_random_forest(X_train_t, y_train, X_test_t, y_test, cfg)

    pipeline = Pipeline([("preprocessor", preprocessor), ("clf", best_rf)])
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, cfg["model"]["artifact_path"])
    logger.info(f"Segment pipeline saved → {cfg['model']['artifact_path']}")
    logger.info(f"Done! Acc: {acc:.4f} | F1: {f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
