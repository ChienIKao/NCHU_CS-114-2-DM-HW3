import numpy as np
import pytest

from src.evaluate import auc_score
from src.models import predict_positive_proba, train_lgbm, train_logreg, train_xgb

RNG = np.random.default_rng(42)
X = RNG.standard_normal((160, 16)).astype(np.float32)
y = RNG.integers(0, 2, 160).astype(np.float32)
X_test = RNG.standard_normal((40, 16)).astype(np.float32)


def _assert_probability_vector(preds: np.ndarray, n: int = 40):
    assert preds.shape == (n,)
    assert np.isfinite(preds).all()
    assert 0.0 <= preds.min() <= preds.max() <= 1.0


def test_logreg_predicts_probabilities():
    model = train_logreg(X, y, max_iter=300)
    _assert_probability_vector(predict_positive_proba(model, X_test))


def test_lgbm_predicts_probabilities():
    model = train_lgbm(X, y, n_estimators=20, num_leaves=15)
    _assert_probability_vector(predict_positive_proba(model, X_test))


def test_xgb_predicts_probabilities():
    model = train_xgb(X, y, n_estimators=20, max_depth=3)
    _assert_probability_vector(predict_positive_proba(model, X_test))


def test_auc_score_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    assert auc_score(y_true, y_score) == pytest.approx(1.0)


def test_auc_score_random():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 1000)
    y_score = rng.random(1000)
    score = auc_score(y_true, y_score)
    assert 0.4 < score < 0.6
