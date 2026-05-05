"""Submission 4: high-score sparse one-hot Logistic Regression with all pair features."""
from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.data import TARGET_COL, load_raw_test, load_raw_train, submission_ids
from src.evaluate import auc_score, describe_auc
from src.sparse_features import build_sparse_onehot_matrices

SUBMISSIONS_DIR = Path("submissions")


def train_sparse_logreg(X, y, C: float = 1.0):
    model = LogisticRegression(C=C, max_iter=1500, solver="liblinear")
    model.fit(X, y)
    return model


def _cross_validate(train: pd.DataFrame, C: float, n_splits: int = 5) -> list[float]:
    y = train[TARGET_COL].to_numpy(np.float32)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores: list[float] = []

    for fold, (tr_idx, val_idx) in enumerate(cv.split(train, y), start=1):
        fold_train = train.iloc[tr_idx].reset_index(drop=True)
        fold_valid = train.iloc[val_idx].drop(columns=[TARGET_COL]).reset_index(drop=True)
        X_tr, X_val, _ = build_sparse_onehot_matrices(fold_train, fold_valid)
        model = train_sparse_logreg(X_tr, y[tr_idx], C=C)
        preds = model.predict_proba(X_val)[:, 1]
        score = auc_score(y[val_idx], preds)
        scores.append(score)
        print(f"Fold {fold} AUC: {score:.5f}")

    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength.")
    parser.add_argument(
        "--output",
        default="submission4_sparse_logreg.csv",
        help="Output CSV filename under submissions/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    train = load_raw_train()
    test = load_raw_test()
    y = train[TARGET_COL].to_numpy(np.float32)

    scores = _cross_validate(train, C=args.C)
    print(f"Sparse LogReg CV AUC: {describe_auc(scores)}")

    X_train, X_test, encoder = build_sparse_onehot_matrices(train, test)
    print(f"Sparse feature shape: train={X_train.shape}, test={X_test.shape}")
    print(f"One-hot encoded columns: {len(encoder.get_feature_names_out())}")

    model = train_sparse_logreg(X_train, y, C=args.C)
    preds = model.predict_proba(X_test)[:, 1]

    out = pd.DataFrame({"Id": submission_ids(test), "Action": preds})
    out_path = SUBMISSIONS_DIR / args.output
    out.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
