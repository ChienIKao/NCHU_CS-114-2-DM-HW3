# AGENTS.md

Guidance for Codex agents working in this workspace.

## Project Overview

This is NCHU Data Mining course Homework 3. The task is binary classification for the Kaggle Amazon Employee Access Challenge, predicting `ACTION` for employee resource access requests. The competition metric is AUC.

Relevant files:
- `tasks/DM_Homework_3.pdf`: assignment PDF.
- `tasks/kagglehub.md`: KaggleHub download snippet.
- `spec/2026-05-05-amazon-employee-access.md`: implementation plan.
- `CLAUDE.md`: existing project guidance.

## Dataset

Download the Kaggle competition files with:

```python
import kagglehub

path = kagglehub.competition_download("amazon-employee-access-challenge")
```

Expected raw files are `train.csv` and `test.csv`. Keep downloaded data out of version control.

Columns:
- Target: `ACTION`
- Features: `RESOURCE`, `MGR_ID`, `ROLE_ROLLUP_1`, `ROLE_ROLLUP_2`, `ROLE_DEPTNAME`, `ROLE_TITLE`, `ROLE_FAMILY_DESC`, `ROLE_FAMILY`, `ROLE_CODE`

All features are categorical IDs.

## Development Notes

- Prefer a small, scriptable Python ML pipeline over notebook-only work.
- Use AUC-oriented evaluation and probability predictions.
- Good feature approaches for this dataset include target encoding, count encoding, and pairwise categorical combinations such as `RESOURCE_MGR_ID`.
- Avoid one-hot encoding as the main approach because the categorical space is sparse.
- Keep generated data and submission CSVs under ignored directories such as `data/` and `submissions/`.

## Planned Structure

The spec proposes this structure:

```text
src/
  data.py
  features.py
  models.py
  evaluate.py
scripts/
  run_baseline.py
  run_improved.py
  run_ensemble.py
tests/
  test_data.py
  test_features.py
  test_models.py
```

Follow the implementation plan in `spec/2026-05-05-amazon-employee-access.md` unless the user asks to change direction.

## Verification

Use focused tests while building:

```powershell
pytest tests/ -v --tb=short
```

Data-dependent tests should skip cleanly if Kaggle credentials or raw files are not present.
