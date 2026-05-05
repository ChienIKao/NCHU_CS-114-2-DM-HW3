# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NCHU Data Mining Course (114-2) — Homework 3. Binary classification task using the [Amazon Employee Access Challenge](https://www.kaggle.com/competitions/amazon-employee-access-challenge) Kaggle competition dataset. Goal: predict `ACTION` (0 or 1) for employee resource access requests, evaluated by AUC.

## Dataset

Download via kagglehub (see [tasks/kagglehub.md](tasks/kagglehub.md)):

```python
import kagglehub
path = kagglehub.competition_download('amazon-employee-access-challenge')
```

Files: `train.csv`, `test.csv`. All features are categorical IDs: `RESOURCE`, `MGR_ID`, `ROLE_ROLLUP_1`, `ROLE_ROLLUP_2`, `ROLE_DEPTNAME`, `ROLE_TITLE`, `ROLE_FAMILY_DESC`, `ROLE_FAMILY`, `ROLE_CODE`. Target: `ACTION`.

## Assignment Requirements

- Minimum 3 Kaggle submissions (screenshot evidence required)
- Feature engineering, model training, iterative improvement
- Final deliverable: PDF report + code ZIP

## Development Environment

Python with standard ML stack. Recommended environment setup:

```bash
pip install kagglehub pandas numpy scikit-learn lightgbm xgboost matplotlib seaborn
```

Run Jupyter notebooks:

```bash
jupyter notebook
# or
jupyter lab
```

## Key Technical Notes

- All features are categorical — use target encoding or count encoding (not one-hot, too sparse)
- Evaluation metric is AUC, not accuracy — tune `predict_proba` thresholds accordingly
- Combine features (e.g., `RESOURCE_MGR_ID` pairs) as additional engineered features
- The assignment PDF is at [tasks/DM_Homework_3.pdf](tasks/DM_Homework_3.pdf)
