"""
train_revenue.py - Trains a Random Forest Regressor to predict revenue_usd.
Logs all experiments with MLflow: baselines, grid search, residuals plot.

Usage (from project root):
    python 02-experiment-tracking/train_revenue.py
"""

import argparse
import logging
import os

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
import category_encoders as ce

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_and_prepare(cfg):
    df = pd.read_csv(cfg["data"]["raw_path"])
    logger.info(f"Loaded {len(df):,} rows")

    rc = cfg["revenue_model"]["features"]
    feature_cols = (
        rc["categorical_high_cardinality"]
        + rc["categorical_ohe"]
        + rc["ordinal"]
        + rc["numeric"]
    )
    target = rc["target"]

    df = df.dropna(subset=[target])
    X = df[feature_cols]
    y = df[target]

    logger.info(
        f"Revenue — mean: ${y.mean():,.0f} | median: ${y.median():,.0f} | max: ${y.max():,.0f}"
    )
    return X, y


def build_preprocessor(cfg):
    rc = cfg["revenue_model"]["features"]
    high_card = rc["categorical_high_cardinality"]
    ohe_cols = rc["categorical_ohe"]
    ord_cols = rc["ordinal"]
    num_cols = rc["numeric"]
    ord_cats = [rc["ordinal_categories"][c] for c in ord_cols]

    return ColumnTransformer(transformers=[
        ("target_enc", ce.TargetEncoder(cols=high_card), high_card),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ohe_cols),
        ("ordinal", OrdinalEncoder(
            categories=ord_cats,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        ), ord_cols),
        ("numeric", StandardScaler(), num_cols),
    ], remainder="drop")


def log_baselines(X_train_t, y_train, X_test_t, y_test):
    """Log mean predictor and linear regression baselines."""
    with mlflow.start_run(run_name="baseline_mean_predictor", nested=True):
        dummy = DummyRegressor(strategy="mean")
        dummy.fit(X_train_t, y_train)
        y_pred = dummy.predict(X_test_t)
        mlflow.log_metrics({
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        })

    with mlflow.start_run(run_name="baseline_linear_regression", nested=True):
        lr = LinearRegression()
        lr.fit(X_train_t, y_train)
        y_pred = lr.predict(X_test_t)
        mlflow.log_metrics({
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        })


def train_random_forest(X_train_t, y_train, X_test_t, y_test, cfg):
    """Grid-search Random Forest Regressor, log results to MLflow."""
    params = cfg["revenue_model"]["hyperparams"]
    grid = GridSearchCV(
        RandomForestRegressor(random_state=cfg["data"]["random_state"]),
        params, cv=3, scoring="r2", n_jobs=-1, verbose=1,
    )

    with mlflow.start_run(run_name="revenue_rf_gridsearch", nested=True):
        grid.fit(X_train_t, y_train)
        best = grid.best_estimator_
        y_pred = best.predict(X_test_t)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics({"mae": mae, "r2": r2, "rmse": rmse})

        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.3, s=10)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
        ax.set_xlabel("Actual Revenue")
        ax.set_ylabel("Predicted Revenue")
        ax.set_title("Actual vs Predicted Revenue")
        mlflow.log_figure(fig, "actual_vs_predicted.png")
        plt.close(fig)

        mlflow.sklearn.log_model(best, artifact_path="revenue_random_forest")
        logger.info(f"Best RF — MAE: ${mae:,.0f} | R²: {r2:.4f} | RMSE: ${rmse:,.0f}")
        logger.info(f"Best params: {grid.best_params_}")

    return best, mae, r2, rmse


def main(config_path="config.yaml"):
    cfg = load_config(config_path)
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["revenue_model"]["mlflow_experiment"])

    X, y = load_and_prepare(cfg)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
    )
    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    preprocessor = build_preprocessor(cfg)
    X_train_t = preprocessor.fit_transform(X_train, y_train)
    X_test_t = preprocessor.transform(X_test)

    with mlflow.start_run(run_name="apple_revenue_predictor"):
        log_baselines(X_train_t, y_train, X_test_t, y_test)
        best_rf, mae, r2, rmse = train_random_forest(X_train_t, y_train, X_test_t, y_test, cfg)

    pipeline = Pipeline([("preprocessor", preprocessor), ("reg", best_rf)])
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, cfg["revenue_model"]["artifact_path"])
    logger.info(f"Revenue pipeline saved → {cfg['revenue_model']['artifact_path']}")
    logger.info(f"Done! MAE: ${mae:,.0f} | R²: {r2:.4f} | RMSE: ${rmse:,.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
