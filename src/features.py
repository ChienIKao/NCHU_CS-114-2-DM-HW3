from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from src.data import FEATURE_COLS, TARGET_COL

COMBO_PAIRS = [
    ("RESOURCE", "MGR_ID"),
    ("RESOURCE", "ROLE_CODE"),
    ("MGR_ID", "ROLE_CODE"),
    ("ROLE_DEPTNAME", "ROLE_TITLE"),
    ("ROLE_FAMILY", "ROLE_CODE"),
]


def add_combination_features(
    df: pd.DataFrame,
    pairs: Iterable[tuple[str, str]] = COMBO_PAIRS,
) -> pd.DataFrame:
    """Add categorical pair-combination features as stable string categories."""
    result = df.copy()
    for left, right in pairs:
        if left in result.columns and right in result.columns:
            result[f"{left}_{right}"] = (
                result[left].astype("string") + "__" + result[right].astype("string")
            )
    return result


def target_encode(
    df: pd.DataFrame,
    cols: list[str],
    target_col: str = TARGET_COL,
    mapping: dict[str, dict[Any, float]] | None = None,
    smoothing: float = 10.0,
) -> tuple[pd.DataFrame, dict[str, dict[Any, float]]]:
    """Smoothed target encoding fitted on train data or applied with a mapping."""
    encoded: dict[str, pd.Series] = {}
    fitted: dict[str, dict[Any, float]] = {}

    for col in cols:
        if mapping is not None:
            col_map = mapping[col]
            global_mean = col_map["__global__"]
            encoded[col] = df[col].map(col_map).fillna(global_mean).astype(float)
            continue

        global_mean = float(df[target_col].mean())
        stats = df.groupby(col, dropna=False)[target_col].agg(["mean", "count"])
        smooth = (stats["mean"] * stats["count"] + global_mean * smoothing) / (
            stats["count"] + smoothing
        )
        col_map = {key: float(value) for key, value in smooth.items()}
        col_map["__global__"] = global_mean
        fitted[col] = col_map
        encoded[col] = df[col].map(col_map).fillna(global_mean).astype(float)

    return pd.DataFrame(encoded, index=df.index), mapping if mapping is not None else fitted


def count_encode(
    df: pd.DataFrame,
    cols: list[str],
    mapping: dict[str, dict[Any, int]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[Any, int]]]:
    """Replace categories with their training frequency."""
    encoded: dict[str, pd.Series] = {}
    fitted: dict[str, dict[Any, int]] = {}

    for col in cols:
        if mapping is not None:
            col_map = mapping[col]
            encoded[col] = df[col].map(col_map).fillna(1).astype(float)
            continue

        counts = df[col].value_counts(dropna=False).to_dict()
        fitted[col] = {key: int(value) for key, value in counts.items()}
        encoded[col] = df[col].map(counts).fillna(1).astype(float)

    return pd.DataFrame(encoded, index=df.index), mapping if mapping is not None else fitted


def feature_columns() -> list[str]:
    """Return base categorical features plus engineered pair features."""
    return FEATURE_COLS + [f"{left}_{right}" for left, right in COMBO_PAIRS]


def build_feature_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create encoded train/test feature frames using train-fitted mappings."""
    train_fe = add_combination_features(train)
    test_fe = add_combination_features(test)
    cols = feature_columns()

    te_train, te_map = target_encode(train_fe, cols, target_col=target_col)
    te_test, _ = target_encode(test_fe, cols, mapping=te_map)
    te_train = te_train.add_suffix("_te")
    te_test = te_test.add_suffix("_te")

    ce_train, ce_map = count_encode(train_fe, cols)
    ce_test, _ = count_encode(test_fe, cols, mapping=ce_map)
    ce_train = np.log1p(ce_train).add_suffix("_count")
    ce_test = np.log1p(ce_test).add_suffix("_count")

    return pd.concat([te_train, ce_train], axis=1), pd.concat([te_test, ce_test], axis=1)


def build_feature_matrix(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> tuple[np.ndarray, np.ndarray]:
    """Return encoded `float32` matrices for model training and prediction."""
    train_frame, test_frame = build_feature_frames(train, test, target_col=target_col)
    return train_frame.to_numpy(np.float32), test_frame.to_numpy(np.float32)
