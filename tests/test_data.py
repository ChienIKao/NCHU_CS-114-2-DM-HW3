from pathlib import Path

import pytest

from src.data import FEATURE_COLS, TARGET_COL, load_raw_test, load_raw_train, submission_ids

pytestmark = pytest.mark.skipif(
    not Path("data/raw/train.csv").exists() or not Path("data/raw/test.csv").exists(),
    reason="Raw Kaggle data not downloaded yet",
)


def test_train_has_target_column():
    train = load_raw_train()
    assert TARGET_COL in train.columns


def test_test_lacks_target_column():
    test = load_raw_test()
    assert TARGET_COL not in test.columns


def test_feature_columns_present():
    train = load_raw_train()
    for col in FEATURE_COLS:
        assert col in train.columns


def test_target_is_binary():
    train = load_raw_train()
    assert set(train[TARGET_COL].unique()).issubset({0, 1})


def test_submission_ids_length_matches_test():
    test = load_raw_test()
    ids = submission_ids(test)
    assert len(ids) == len(test)
