"""Submission 3: blended LightGBM + XGBoost ensemble."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data import TARGET_COL, load_raw_test, load_raw_train, submission_ids
from src.evaluate import auc_score
from src.features import build_feature_matrix
from src.models import predict_positive_proba, train_lgbm, train_xgb

SUBMISSIONS_DIR = Path("submissions")
LGBM_WEIGHT = 0.6


def _oof_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    model_fn: Callable,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(train), dtype=float)
    test_preds = np.zeros(len(test), dtype=float)

    for fold, (tr_idx, val_idx) in enumerate(cv.split(train, y), start=1):
        fold_train = train.iloc[tr_idx].reset_index(drop=True)
        fold_valid = train.iloc[val_idx].drop(columns=[TARGET_COL]).reset_index(drop=True)
        X_tr, X_val = build_feature_matrix(fold_train, fold_valid)
        _, X_test = build_feature_matrix(fold_train, test)

        model = model_fn(X_tr, y[tr_idx])
        oof[val_idx] = predict_positive_proba(model, X_val)
        test_preds += predict_positive_proba(model, X_test) / n_splits
        print(f"  Fold {fold} AUC: {auc_score(y[val_idx], oof[val_idx]):.5f}")

    return oof, test_preds


def main() -> None:
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    train = load_raw_train()
    test = load_raw_test()
    y = train[TARGET_COL].to_numpy(np.float32)

    print("Training LightGBM...")
    lgbm_oof, lgbm_test = _oof_predict(train, test, y, train_lgbm)
    print(f"LGBM OOF AUC: {auc_score(y, lgbm_oof):.5f}")

    print("Training XGBoost...")
    xgb_oof, xgb_test = _oof_predict(train, test, y, train_xgb)
    print(f"XGB  OOF AUC: {auc_score(y, xgb_oof):.5f}")

    blend_oof = LGBM_WEIGHT * lgbm_oof + (1 - LGBM_WEIGHT) * xgb_oof
    blend_test = LGBM_WEIGHT * lgbm_test + (1 - LGBM_WEIGHT) * xgb_test
    print(f"Ensemble OOF AUC: {auc_score(y, blend_oof):.5f}")

    out = pd.DataFrame({"Id": submission_ids(test), "Action": blend_test})
    out_path = SUBMISSIONS_DIR / "submission3_ensemble.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
