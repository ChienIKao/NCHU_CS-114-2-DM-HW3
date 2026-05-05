import numpy as np
import pandas as pd

from src.data import FEATURE_COLS
from src.features import (
    add_combination_features,
    build_feature_matrix,
    count_encode,
    feature_columns,
    target_encode,
)


def _make_train(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    train = pd.DataFrame({col: rng.integers(1, 12, n) for col in FEATURE_COLS})
    train["ACTION"] = rng.integers(0, 2, n)
    return train


def _make_test(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({col: rng.integers(1, 12, n) for col in FEATURE_COLS})


def test_target_encode_output_shape():
    train = _make_train()
    encoded, mapping = target_encode(train, FEATURE_COLS)
    assert encoded.shape == (200, len(FEATURE_COLS))
    assert set(mapping) == set(FEATURE_COLS)


def test_target_encode_values_in_range():
    train = _make_train()
    encoded, _ = target_encode(train, FEATURE_COLS)
    assert encoded.min().min() >= 0.0
    assert encoded.max().max() <= 1.0


def test_target_encode_applies_to_test():
    train = _make_train()
    test = _make_test()
    _, mapping = target_encode(train, FEATURE_COLS)
    encoded_test, _ = target_encode(test, FEATURE_COLS, mapping=mapping)
    assert encoded_test.shape == (50, len(FEATURE_COLS))


def test_count_encode_output_shape_and_positive_values():
    train = _make_train()
    encoded, mapping = count_encode(train, FEATURE_COLS)
    assert encoded.shape == (200, len(FEATURE_COLS))
    assert (encoded >= 1).all().all()
    assert set(mapping) == set(FEATURE_COLS)


def test_add_combination_features_adds_columns():
    train = _make_train()
    result = add_combination_features(train)
    assert "RESOURCE_MGR_ID" in result.columns
    assert "RESOURCE_ROLE_CODE" in result.columns


def test_feature_columns_contains_engineered_names():
    cols = feature_columns()
    assert "ROLE_DEPTNAME_ROLE_TITLE" in cols
    assert len(cols) > len(FEATURE_COLS)


def test_build_feature_matrix_returns_two_arrays():
    train = _make_train()
    test = _make_test()
    X_train, X_test = build_feature_matrix(train, test)
    assert X_train.shape[0] == 200
    assert X_test.shape[0] == 50
    assert X_train.shape[1] == X_test.shape[1]
    assert X_train.dtype == np.float32
