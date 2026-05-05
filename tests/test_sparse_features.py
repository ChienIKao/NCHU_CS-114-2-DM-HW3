import pandas as pd
from scipy import sparse

from src.data import FEATURE_COLS
from src.sparse_features import ALL_PAIR_COMBOS, add_all_pair_features, build_sparse_onehot_matrices


def _make_frame(n: int = 12) -> pd.DataFrame:
    return pd.DataFrame({col: [(i % 4) + 1 for i in range(n)] for col in FEATURE_COLS})


def test_add_all_pair_features_adds_every_pair():
    result = add_all_pair_features(_make_frame())
    assert result.shape[1] == len(FEATURE_COLS) + len(ALL_PAIR_COMBOS)
    assert "RESOURCE_MGR_ID" in result.columns
    assert "ROLE_FAMILY_ROLE_CODE" in result.columns


def test_build_sparse_onehot_matrices_returns_csr():
    train = _make_frame(20)
    test = _make_frame(8)
    X_train, X_test, encoder = build_sparse_onehot_matrices(train, test)
    assert sparse.isspmatrix_csr(X_train)
    assert sparse.isspmatrix_csr(X_test)
    assert X_train.shape[0] == 20
    assert X_test.shape[0] == 8
    assert X_train.shape[1] == X_test.shape[1]
    assert len(encoder.get_feature_names_out()) == X_train.shape[1]
