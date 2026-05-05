"""Submission 1: logistic regression baseline on encoded categorical features."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data import TARGET_COL, load_raw_test, load_raw_train, submission_ids
from src.evaluate import auc_score, describe_auc
from src.features import build_feature_matrix
from src.models import predict_positive_proba, train_logreg

SUBMISSIONS_DIR = Path("submissions")


def _cross_validate(train: pd.DataFrame, n_splits: int = 5) -> list[float]:
    y = train[TARGET_COL].to_numpy(np.float32)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores: list[float] = []

    for fold, (tr_idx, val_idx) in enumerate(cv.split(train, y), start=1):
        fold_train = train.iloc[tr_idx].reset_index(drop=True)
        fold_valid = train.iloc[val_idx].drop(columns=[TARGET_COL]).reset_index(drop=True)
        X_tr, X_val = build_feature_matrix(fold_train, fold_valid)
        model = train_logreg(X_tr, y[tr_idx])
        preds = predict_positive_proba(model, X_val)
        score = auc_score(y[val_idx], preds)
        scores.append(score)
        print(f"Fold {fold} AUC: {score:.5f}")

    return scores


def main() -> None:
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    train = load_raw_train()
    test = load_raw_test()

    scores = _cross_validate(train)
    print(f"LogReg CV AUC: {describe_auc(scores)}")

    y = train[TARGET_COL].to_numpy(np.float32)
    X_train, X_test = build_feature_matrix(train, test)
    model = train_logreg(X_train, y)
    preds = predict_positive_proba(model, X_test)

    out = pd.DataFrame({"Id": submission_ids(test), "Action": preds})
    out_path = SUBMISSIONS_DIR / "submission1_logreg.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
