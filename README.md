# HW3 - Amazon Employee Access Challenge

Binary classification for the Kaggle Amazon Employee Access Challenge. The goal is to predict `ACTION` and optimize AUC.

## Results

| Version | Model | Public AUC | Private AUC |
| --- | --- | ---: | ---: |
| 1 | Logistic Regression baseline | 0.78157 | 0.79105 |
| 2 | XGBoost | 0.84310 | 0.85785 |
| 3 | LightGBM + XGBoost ensemble | 0.84233 | 0.85698 |
| 4 | Sparse all-pairs Logistic Regression | 0.90029 | 0.89455 |

The best Kaggle result is Submission 4. Sparse one-hot encoding with all pairwise categorical interactions gives the linear model direct access to high-signal access patterns without forcing arbitrary numeric order on categorical IDs.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data

First authenticate KaggleHub. Use the `username` and `key` values from Kaggle's downloaded `kaggle.json`, then fill `.env`:

```dotenv
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

If your token starts with `KGAT_`, generate/download the classic `kaggle.json` API token from Kaggle account settings and copy the JSON `key` value instead.
You can also place `kaggle.json` in this project root. The loader will use it for Kaggle authentication and override stale Kaggle values in `.env`.

Optional auth check:

```powershell
venv\Scripts\python.exe -c "from src.data import _load_local_kaggle_credentials; _load_local_kaggle_credentials(); from kagglehub.auth import whoami; print(whoami(verbose=False))"
```

Then download the competition files:

```powershell
python -c "from src.data import download_data; download_data()"
```

Competition files are copied to `data/raw/`, which is ignored by git.
If KaggleHub reports that the user is not authenticated, add your Kaggle API token and accept the competition rules on Kaggle before running the command again.

## Run Submission Versions

```powershell
python scripts/run_baseline.py
python scripts/run_improved.py
python scripts/run_ensemble.py
python scripts/run_sparse_logreg.py
```

To reproduce the small C sweep:

```powershell
python scripts/run_sparse_logreg.py --C 0.7 --output submission4_sparse_logreg_C0p7.csv
python scripts/run_sparse_logreg.py --C 1.5 --output submission4_sparse_logreg_C1p5.csv
```

Generated CSV files appear in `submissions/`.

Optional Kaggle CLI submission:

```powershell
kaggle competitions submit -c amazon-employee-access-challenge -f submissions/submission4_sparse_logreg.csv -m "submission4 sparse all-pairs logistic regression"
```

## Git Hygiene

The repository intentionally ignores Kaggle credentials, raw data, virtual environments, and generated submissions:

- `.env`
- `kaggle.json`
- `venv/`
- `data/raw/`
- `data/processed/`
- `submissions/`

## Tests

```powershell
pytest tests/ -v --tb=short
```

Data-dependent tests skip cleanly if `data/raw/train.csv` and `data/raw/test.csv` are not present.

## Structure

| Path | Purpose |
| --- | --- |
| `src/data.py` | Kaggle download and raw CSV loading |
| `src/features.py` | target encoding, count encoding, feature combinations |
| `src/sparse_features.py` | sparse one-hot features with all pairwise interactions |
| `src/models.py` | model training wrappers |
| `src/evaluate.py` | AUC helpers |
| `scripts/` | submission iterations: encoded models, ensemble, and sparse one-hot model |
| `docs/report_draft.md` | report draft and score log template |
