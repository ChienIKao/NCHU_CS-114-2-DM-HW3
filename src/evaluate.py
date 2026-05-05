from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC AUC as a plain Python float."""
    return float(roc_auc_score(y_true, y_score))


def describe_auc(scores: list[float] | np.ndarray) -> str:
    values = np.asarray(scores, dtype=float)
    return f"{values.mean():.5f} +/- {values.std():.5f}"
