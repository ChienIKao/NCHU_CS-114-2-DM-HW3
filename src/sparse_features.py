from __future__ import annotations

from itertools import combinations

import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder

from src.data import FEATURE_COLS

ALL_PAIR_COMBOS = list(combinations(FEATURE_COLS, 2))


def add_all_pair_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return base categorical columns plus every pairwise categorical interaction."""
    result = df[FEATURE_COLS].copy()
    for left, right in ALL_PAIR_COMBOS:
        result[f"{left}_{right}"] = (
            result[left].astype("string") + "__" + result[right].astype("string")
        )
    return result.astype("string")


def build_sparse_onehot_matrices(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, OneHotEncoder]:
    """Fit one-hot encoding on train categories and transform train/test."""
    encoder = OneHotEncoder(handle_unknown="ignore", dtype="float32")
    X_train = encoder.fit_transform(add_all_pair_features(train))
    X_test = encoder.transform(add_all_pair_features(test))
    return X_train.tocsr(), X_test.tocsr(), encoder
