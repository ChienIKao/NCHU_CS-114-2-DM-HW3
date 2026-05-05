# HW3 - Amazon Employee Access Challenge

Binary classification for the Kaggle Amazon Employee Access Challenge. The goal is to predict `ACTION` and optimize AUC.

## Results

| Version | Model | Public AUC | Private AUC |
| --- | --- | ---: | ---: |
| 1 | Logistic Regression baseline | 0.78157 | 0.79105 |
| 2 | XGBoost | 0.84310 | 0.85785 |
| 3 | LightGBM + XGBoost ensemble | 0.84233 | 0.85698 |

The best Kaggle result is Submission 2, the XGBoost model. The ensemble slightly improved local CV but was marginally lower on Kaggle, likely because the weaker LightGBM component diluted the XGBoost ranking.

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
```

Generated CSV files appear in `submissions/`.

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
| `src/models.py` | model training wrappers |
| `src/evaluate.py` | AUC helpers |
| `scripts/` | three submission iterations: Logistic Regression, XGBoost, and ensemble |
| `docs/report_draft.md` | report draft and score log template |
