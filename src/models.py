from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_logreg(X: np.ndarray, y: np.ndarray, **params: Any) -> Pipeline:
    """Train a regularized logistic regression baseline."""
    defaults = {
        "C": 0.1,
        "max_iter": 1000,
        "solver": "lbfgs",
        "class_weight": "balanced",
    }
    defaults.update(params)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**defaults)),
        ]
    )
    model.fit(X, y)
    return model


def train_lgbm(X: np.ndarray, y: np.ndarray, **params: Any):
    """Train a LightGBM classifier."""
    import lightgbm as lgb

    defaults = {
        "objective": "binary",
        "n_estimators": 350,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.05,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }
    defaults.update(params)
    model = lgb.LGBMClassifier(**defaults)
    model.fit(X, y)
    return model


def train_xgb(X: np.ndarray, y: np.ndarray, **params: Any):
    """Train an XGBoost classifier, with a sklearn fallback if unavailable."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=42,
        )
        model.fit(X, y)
        return model

    defaults = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": 350,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 2,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }
    defaults.update(params)
    model = XGBClassifier(**defaults)
    model.fit(X, y)
    return model


def predict_positive_proba(model: Any, X: np.ndarray) -> np.ndarray:
    """Return probability for class 1 from sklearn-style classifiers."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    return np.asarray(model.predict(X), dtype=float)
