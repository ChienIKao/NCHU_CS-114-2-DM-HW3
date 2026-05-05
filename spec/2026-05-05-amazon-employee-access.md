# Amazon Employee Access Challenge — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete ML pipeline that predicts employee resource access approval (AUC metric) for the Amazon Employee Access Kaggle competition, from raw data to 3+ scored submissions.

**Architecture:** All categorical features are target/count-encoded → fed into gradient boosting models (LightGBM primary, XGBoost/LogReg for ensemble). Three submission iterations: baseline → feature engineering → tuned ensemble.

**Tech Stack:** Python 3.11+, scikit-learn, lightgbm, xgboost, pandas, numpy, kagglehub, matplotlib, seaborn, pytest

---

## File Map

```
hw3/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                  # files downloaded from kaggle (gitignored)
│   └── processed/            # encoded feature matrices (gitignored)
├── src/
│   ├── __init__.py
│   ├── data.py               # download + load raw CSVs
│   ├── features.py           # all encoding / feature-engineering logic
│   ├── models.py             # train / predict wrappers
│   └── evaluate.py           # AUC helpers
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modelling.ipynb
├── submissions/              # CSV files to upload to Kaggle (gitignored)
├── tests/
│   ├── test_features.py
│   └── test_models.py
└── scripts/
    ├── run_baseline.py
    ├── run_improved.py
    └── run_ensemble.py
```

---

## Task 1: Project Scaffold (venv, requirements, gitignore, README)

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `README.md`
- Create: `src/__init__.py`
- Create all empty `__init__.py` files listed in the file map

- [ ] **Step 1: Create the virtual environment**

```powershell
# Run from hw3/ root
python -m venv venv
# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1
```

- [ ] **Step 2: Create `requirements.txt`**

```
kagglehub==0.3.9
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.5.2
lightgbm==4.5.0
xgboost==2.1.3
matplotlib==3.9.4
seaborn==0.13.2
jupyter==1.1.1
ipykernel==6.29.5
pytest==8.3.4
```

Install:

```powershell
pip install -r requirements.txt
```

Expected: All packages install without error.

- [ ] **Step 3: Create `.gitignore`**

```gitignore
# Python
venv/
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.eggs/

# Data (large files — keep out of git)
data/raw/
data/processed/
submissions/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Secrets / env
.env
kaggle.json

# OS
.DS_Store
Thumbs.db
```

- [ ] **Step 4: Create `README.md`**

```markdown
# HW3 — Amazon Employee Access Challenge

Binary classification (AUC) on the [Amazon Employee Access Challenge](https://www.kaggle.com/competitions/amazon-employee-access-challenge).

## Setup

```bash
python -m venv venv
# Windows
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data

```bash
python -c "from src.data import download_data; download_data()"
```

Files are saved to `data/raw/`.

## Run submissions

```bash
# Submission 1 — Logistic Regression baseline
python scripts/run_baseline.py

# Submission 2 — LightGBM + target encoding
python scripts/run_improved.py

# Submission 3 — Tuned ensemble
python scripts/run_ensemble.py
```

CSV files appear in `submissions/`. Upload each to Kaggle manually.

## Tests

```bash
pytest tests/ -v
```

## Project structure

| Path | Purpose |
|------|---------|
| `src/data.py` | Download + load raw CSVs |
| `src/features.py` | Target encoding, count encoding, feature combinations |
| `src/models.py` | Train / predict wrappers |
| `src/evaluate.py` | AUC utilities |
| `scripts/` | End-to-end pipeline entry-points |
| `notebooks/` | EDA and experimentation |
```

- [ ] **Step 5: Create placeholder `__init__.py` files**

```powershell
New-Item -ItemType File src/__init__.py
New-Item -ItemType Directory data/raw, data/processed, submissions, notebooks, scripts, tests -Force
New-Item -ItemType File tests/__init__.py
```

- [ ] **Step 6: Commit scaffold**

```bash
git init
git add .gitignore requirements.txt README.md src/__init__.py tests/__init__.py
git commit -m "chore: project scaffold with venv, requirements, and gitignore"
```

---

## Task 2: Data Loading (`src/data.py`)

**Files:**
- Create: `src/data.py`
- Test: `tests/test_data.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_data.py`:

```python
import os
import pandas as pd
import pytest
from src.data import load_raw_train, load_raw_test

# Skip if raw data not present (CI without Kaggle credentials)
pytestmark = pytest.mark.skipif(
    not os.path.exists("data/raw/train.csv"),
    reason="Raw data not downloaded yet"
)

def test_train_has_action_column():
    df = load_raw_train()
    assert "ACTION" in df.columns

def test_test_lacks_action_column():
    df = load_raw_test()
    assert "ACTION" not in df.columns

def test_feature_columns_present():
    expected = [
        "RESOURCE", "MGR_ID", "ROLE_ROLLUP_1", "ROLE_ROLLUP_2",
        "ROLE_DEPTNAME", "ROLE_TITLE", "ROLE_FAMILY_DESC",
        "ROLE_FAMILY", "ROLE_CODE",
    ]
    df = load_raw_train()
    for col in expected:
        assert col in df.columns, f"Missing column: {col}"

def test_action_is_binary():
    df = load_raw_train()
    assert set(df["ACTION"].unique()).issubset({0, 1})
```

- [ ] **Step 2: Run test to verify it fails (or skips)**

```powershell
pytest tests/test_data.py -v
```

Expected: SKIP (data not yet downloaded) or FAIL with ImportError.

- [ ] **Step 3: Implement `src/data.py`**

```python
from pathlib import Path
import pandas as pd
import kagglehub

RAW_DIR = Path("data/raw")
FEATURE_COLS = [
    "RESOURCE", "MGR_ID", "ROLE_ROLLUP_1", "ROLE_ROLLUP_2",
    "ROLE_DEPTNAME", "ROLE_TITLE", "ROLE_FAMILY_DESC",
    "ROLE_FAMILY", "ROLE_CODE",
]


def download_data() -> Path:
    """Download competition files to data/raw/ if not already present."""
    if (RAW_DIR / "train.csv").exists():
        print(f"Data already present at {RAW_DIR}")
        return RAW_DIR
    path = kagglehub.competition_download("amazon-employee-access-challenge")
    src = Path(path)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*.csv"):
        (RAW_DIR / f.name).write_bytes(f.read_bytes())
    print(f"Data saved to {RAW_DIR}")
    return RAW_DIR


def load_raw_train() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "train.csv")


def load_raw_test() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "test.csv")
```

- [ ] **Step 4: Download data and run tests**

```powershell
python -c "from src.data import download_data; download_data()"
pytest tests/test_data.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data.py tests/test_data.py
git commit -m "feat: data download and loading utilities"
```

---

## Task 3: Feature Engineering (`src/features.py`)

**Files:**
- Create: `src/features.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_features.py`:

```python
import pandas as pd
import numpy as np
import pytest
from src.features import (
    target_encode,
    count_encode,
    add_combination_features,
    build_feature_matrix,
)

FEATURE_COLS = [
    "RESOURCE", "MGR_ID", "ROLE_ROLLUP_1", "ROLE_ROLLUP_2",
    "ROLE_DEPTNAME", "ROLE_TITLE", "ROLE_FAMILY_DESC",
    "ROLE_FAMILY", "ROLE_CODE",
]


def _make_train():
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {col: rng.integers(1, 10, n) for col in FEATURE_COLS}
    )
    df["ACTION"] = rng.integers(0, 2, n)
    return df


def _make_test():
    rng = np.random.default_rng(0)
    n = 50
    return pd.DataFrame(
        {col: rng.integers(1, 10, n) for col in FEATURE_COLS}
    )


def test_target_encode_output_shape():
    train = _make_train()
    encoded, mapping = target_encode(train, FEATURE_COLS, target_col="ACTION")
    assert encoded.shape == (200, len(FEATURE_COLS))


def test_target_encode_values_in_range():
    train = _make_train()
    encoded, _ = target_encode(train, FEATURE_COLS, target_col="ACTION")
    assert encoded.min().min() >= 0.0
    assert encoded.max().max() <= 1.0


def test_target_encode_applies_to_test():
    train = _make_train()
    test = _make_test()
    _, mapping = target_encode(train, FEATURE_COLS, target_col="ACTION")
    encoded_test = target_encode(test, FEATURE_COLS, mapping=mapping)[0]
    assert encoded_test.shape == (50, len(FEATURE_COLS))


def test_count_encode_output_shape():
    train = _make_train()
    encoded, mapping = count_encode(train, FEATURE_COLS)
    assert encoded.shape == (200, len(FEATURE_COLS))


def test_count_encode_positive():
    train = _make_train()
    encoded, _ = count_encode(train, FEATURE_COLS)
    assert (encoded >= 1).all().all()


def test_add_combination_features_adds_columns():
    train = _make_train()
    result = add_combination_features(train)
    assert "RESOURCE_MGR_ID" in result.columns
    assert "RESOURCE_ROLE_CODE" in result.columns


def test_build_feature_matrix_returns_two_arrays():
    train = _make_train()
    test = _make_test()
    X_train, X_test = build_feature_matrix(train, test)
    assert X_train.shape[0] == 200
    assert X_test.shape[0] == 50
    assert X_train.shape[1] == X_test.shape[1]
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/test_features.py -v
```

Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `src/features.py`**

```python
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional

FEATURE_COLS = [
    "RESOURCE", "MGR_ID", "ROLE_ROLLUP_1", "ROLE_ROLLUP_2",
    "ROLE_DEPTNAME", "ROLE_TITLE", "ROLE_FAMILY_DESC",
    "ROLE_FAMILY", "ROLE_CODE",
]
COMBO_PAIRS = [
    ("RESOURCE", "MGR_ID"),
    ("RESOURCE", "ROLE_CODE"),
    ("MGR_ID", "ROLE_CODE"),
]


def target_encode(
    df: pd.DataFrame,
    cols: list[str],
    target_col: str = "ACTION",
    mapping: Optional[dict] = None,
    smoothing: float = 10.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Smoothed target encoding.
    If mapping is None, compute from df (training mode).
    If mapping is provided, apply it (inference mode).
    """
    global_mean = df[target_col].mean() if mapping is None else None
    result = {}
    out_mapping: dict = {}

    for col in cols:
        if mapping is not None:
            col_map = mapping[col]
            result[col] = df[col].map(col_map).fillna(col_map.get("__global__", 0.5))
        else:
            stats = df.groupby(col)[target_col].agg(["mean", "count"])
            smoothed = (stats["mean"] * stats["count"] + global_mean * smoothing) / (
                stats["count"] + smoothing
            )
            col_map = smoothed.to_dict()
            col_map["__global__"] = global_mean
            out_mapping[col] = col_map
            result[col] = df[col].map(col_map).fillna(global_mean)

    return pd.DataFrame(result, index=df.index), (mapping if mapping is not None else out_mapping)


def count_encode(
    df: pd.DataFrame,
    cols: list[str],
    mapping: Optional[dict] = None,
) -> tuple[pd.DataFrame, dict]:
    """Count encoding: replace each value with its frequency in training data."""
    result = {}
    out_mapping: dict = {}

    for col in cols:
        if mapping is not None:
            col_map = mapping[col]
            result[col] = df[col].map(col_map).fillna(1)
        else:
            counts = df[col].value_counts().to_dict()
            out_mapping[col] = counts
            result[col] = df[col].map(counts).fillna(1)

    return pd.DataFrame(result, index=df.index), (mapping if mapping is not None else out_mapping)


def add_combination_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add integer hash of feature pair combinations."""
    df = df.copy()
    for a, b in COMBO_PAIRS:
        if a in df.columns and b in df.columns:
            df[f"{a}_{b}"] = df[a].astype(str) + "_" + df[b].astype(str)
            df[f"{a}_{b}"] = df[f"{a}_{b}"].apply(hash).abs() % 100_000
    return df


def build_feature_matrix(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full pipeline: combination features → target encode → count encode → concat.
    Returns (X_train, X_test) as float32 numpy arrays.
    """
    train = add_combination_features(train)
    test = add_combination_features(test)

    all_feat_cols = FEATURE_COLS + [f"{a}_{b}" for a, b in COMBO_PAIRS]

    te_train, te_map = target_encode(train, all_feat_cols, target_col="ACTION")
    te_test, _ = target_encode(test, all_feat_cols, mapping=te_map)

    ce_train, ce_map = count_encode(train, all_feat_cols)
    ce_test, _ = count_encode(test, all_feat_cols, mapping=ce_map)

    X_train = pd.concat([te_train, ce_train], axis=1).values.astype(np.float32)
    X_test = pd.concat([te_test, ce_test], axis=1).values.astype(np.float32)
    return X_train, X_test
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
pytest tests/test_features.py -v
```

Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "feat: target encoding, count encoding, and combination features"
```

---

## Task 4: Model Wrappers (`src/models.py` + `src/evaluate.py`)

**Files:**
- Create: `src/models.py`
- Create: `src/evaluate.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_models.py`:

```python
import numpy as np
import pytest
from src.models import train_logreg, train_lgbm, train_xgb
from src.evaluate import auc_score

RNG = np.random.default_rng(42)
X = RNG.standard_normal((300, 20)).astype(np.float32)
y = RNG.integers(0, 2, 300).astype(np.float32)
X_te = RNG.standard_normal((100, 20)).astype(np.float32)


def test_logreg_predicts_probabilities():
    model = train_logreg(X, y)
    preds = model.predict_proba(X_te)[:, 1]
    assert preds.shape == (100,)
    assert 0.0 <= preds.min() and preds.max() <= 1.0


def test_lgbm_predicts_probabilities():
    model = train_lgbm(X, y)
    preds = model.predict(X_te)
    assert preds.shape == (100,)
    assert 0.0 <= preds.min() and preds.max() <= 1.0


def test_xgb_predicts_probabilities():
    model = train_xgb(X, y)
    preds = model.predict_proba(X_te)[:, 1]
    assert preds.shape == (100,)
    assert 0.0 <= preds.min() and preds.max() <= 1.0


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
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/test_models.py -v
```

Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `src/evaluate.py`**

```python
import numpy as np
from sklearn.metrics import roc_auc_score


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score))
```

- [ ] **Step 4: Implement `src/models.py`**

```python
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import xgboost as xgb


def train_logreg(X: np.ndarray, y: np.ndarray) -> Pipeline:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs")),
    ])
    model.fit(X, y)
    return model


def train_lgbm(X: np.ndarray, y: np.ndarray) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y, eval_set=[(X, y)], callbacks=[lgb.early_stopping(50, verbose=False)])
    return model


def train_xgb(X: np.ndarray, y: np.ndarray) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X, y, eval_set=[(X, y)], early_stopping_rounds=50, verbose=False)
    return model
```

- [ ] **Step 5: Run tests to verify they pass**

```powershell
pytest tests/test_models.py -v
```

Expected: All 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/models.py src/evaluate.py tests/test_models.py
git commit -m "feat: logistic regression, LightGBM, and XGBoost training wrappers"
```

---

## Task 5: Submission 1 — Logistic Regression Baseline

**Files:**
- Create: `scripts/run_baseline.py`

- [ ] **Step 1: Implement `scripts/run_baseline.py`**

```python
"""Submission 1: Logistic Regression on count-encoded features."""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.data import load_raw_train, load_raw_test
from src.features import build_feature_matrix
from src.models import train_logreg
from src.evaluate import auc_score

SUBMISSIONS_DIR = Path("submissions")
SUBMISSIONS_DIR.mkdir(exist_ok=True)


def main():
    train = load_raw_train()
    test = load_raw_test()

    y = train["ACTION"].values.astype(np.float32)
    X_train, X_test = build_feature_matrix(train, test)

    # CV estimate
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        train_logreg(X_train, y).__class__(C=0.1, max_iter=1000),
        X_train, y, cv=cv, scoring="roc_auc", n_jobs=-1,
    )
    print(f"LogReg CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    # Full fit + predict
    model = train_logreg(X_train, y)
    preds = model.predict_proba(X_test)[:, 1]

    out = pd.DataFrame({"id": test.index + 1, "Action": preds})
    out_path = SUBMISSIONS_DIR / "submission1_logreg.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```powershell
python scripts/run_baseline.py
```

Expected output:
```
LogReg CV AUC: 0.85xx ± 0.00xx
Saved → submissions/submission1_logreg.csv
```

- [ ] **Step 3: Upload `submissions/submission1_logreg.csv` to Kaggle**

Go to https://www.kaggle.com/competitions/amazon-employee-access-challenge/submit, upload the file, and record the public AUC score.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_baseline.py
git commit -m "feat: submission 1 — logistic regression baseline"
```

---

## Task 6: Submission 2 — LightGBM + Full Feature Engineering

**Files:**
- Create: `scripts/run_improved.py`

- [ ] **Step 1: Implement `scripts/run_improved.py`**

```python
"""Submission 2: LightGBM with target encoding + combination features."""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.data import load_raw_train, load_raw_test
from src.features import build_feature_matrix
from src.models import train_lgbm
from src.evaluate import auc_score

SUBMISSIONS_DIR = Path("submissions")
SUBMISSIONS_DIR.mkdir(exist_ok=True)


def main():
    train = load_raw_train()
    test = load_raw_test()

    y = train["ACTION"].values.astype(np.float32)
    X_train, X_test = build_feature_matrix(train, test)

    # OOF (out-of-fold) CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    test_preds = np.zeros(len(test))

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y)):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = train_lgbm(X_tr, y_tr)
        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test) / cv.n_splits

        fold_auc = auc_score(y_val, oof_preds[val_idx])
        print(f"Fold {fold + 1} AUC: {fold_auc:.4f}")

    overall_auc = auc_score(y, oof_preds)
    print(f"Overall OOF AUC: {overall_auc:.4f}")

    out = pd.DataFrame({"id": test.index + 1, "Action": test_preds})
    out_path = SUBMISSIONS_DIR / "submission2_lgbm.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```powershell
python scripts/run_improved.py
```

Expected output: 5 fold AUC lines + "Overall OOF AUC: 0.87xx+" + file saved.

- [ ] **Step 3: Upload `submissions/submission2_lgbm.csv` to Kaggle and record score.**

- [ ] **Step 4: Commit**

```bash
git add scripts/run_improved.py
git commit -m "feat: submission 2 — LightGBM with OOF cross-validation"
```

---

## Task 7: Submission 3 — Tuned Ensemble (LightGBM + XGBoost)

**Files:**
- Create: `scripts/run_ensemble.py`

- [ ] **Step 1: Implement `scripts/run_ensemble.py`**

```python
"""Submission 3: Blended LightGBM + XGBoost ensemble."""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.data import load_raw_train, load_raw_test
from src.features import build_feature_matrix
from src.models import train_lgbm, train_xgb
from src.evaluate import auc_score

SUBMISSIONS_DIR = Path("submissions")
SUBMISSIONS_DIR.mkdir(exist_ok=True)
LGBM_WEIGHT = 0.6  # blend weight for LightGBM


def _oof_predict(model_fn, X_train, y, X_test, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    test_preds = np.zeros(len(X_test))
    for tr_idx, val_idx in cv.split(X_train, y):
        m = model_fn(X_train[tr_idx], y[tr_idx])
        if hasattr(m, "predict_proba"):
            oof[val_idx] = m.predict_proba(X_train[val_idx])[:, 1]
            test_preds += m.predict_proba(X_test)[:, 1] / n_splits
        else:
            oof[val_idx] = m.predict(X_train[val_idx])
            test_preds += m.predict(X_test) / n_splits
    return oof, test_preds


def main():
    train = load_raw_train()
    test = load_raw_test()

    y = train["ACTION"].values.astype(np.float32)
    X_train, X_test = build_feature_matrix(train, test)

    print("Training LightGBM...")
    lgbm_oof, lgbm_test = _oof_predict(train_lgbm, X_train, y, X_test)
    print(f"LGBM OOF AUC: {auc_score(y, lgbm_oof):.4f}")

    print("Training XGBoost...")
    xgb_oof, xgb_test = _oof_predict(train_xgb, X_train, y, X_test)
    print(f"XGB  OOF AUC: {auc_score(y, xgb_oof):.4f}")

    blend_oof = LGBM_WEIGHT * lgbm_oof + (1 - LGBM_WEIGHT) * xgb_oof
    blend_test = LGBM_WEIGHT * lgbm_test + (1 - LGBM_WEIGHT) * xgb_test
    print(f"Ensemble OOF AUC: {auc_score(y, blend_oof):.4f}")

    out = pd.DataFrame({"id": test.index + 1, "Action": blend_test})
    out_path = SUBMISSIONS_DIR / "submission3_ensemble.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```powershell
python scripts/run_ensemble.py
```

Expected output: LGBM OOF, XGB OOF, Ensemble OOF AUC lines + file saved. Ensemble AUC should exceed both single-model scores.

- [ ] **Step 3: Upload `submissions/submission3_ensemble.csv` to Kaggle and record score.**

- [ ] **Step 4: Commit**

```bash
git add scripts/run_ensemble.py
git commit -m "feat: submission 3 — LightGBM + XGBoost ensemble"
```

---

## Task 8: Run Full Test Suite

- [ ] **Step 1: Run all tests**

```powershell
pytest tests/ -v --tb=short
```

Expected: All tests PASS (test_data tests may SKIP if data not present).

- [ ] **Step 2: Final commit**

```bash
git add -A
git commit -m "chore: final cleanup and passing test suite"
```

---

## Self-Review Checklist

- [x] venv setup instructions in Task 1
- [x] `requirements.txt` with pinned versions in Task 1
- [x] `.gitignore` covering venv, data, submissions in Task 1
- [x] `README.md` with setup and run instructions in Task 1
- [x] Project structure matches File Map
- [x] 3 Kaggle submissions: Task 5 (LogReg), Task 6 (LightGBM), Task 7 (Ensemble)
- [x] TDD: failing test → implement → pass → commit for Tasks 2–4
- [x] All code blocks are complete (no TBD / placeholders)
- [x] Type signatures consistent across features.py / models.py / scripts
- [x] `build_feature_matrix` used consistently in all 3 scripts
