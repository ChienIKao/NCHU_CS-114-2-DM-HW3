from __future__ import annotations

from pathlib import Path
import json
import os
import shutil
import zipfile

import pandas as pd

RAW_DIR = Path("data/raw")
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

TARGET_COL = "ACTION"
ID_COL = "id"
FEATURE_COLS = [
    "RESOURCE",
    "MGR_ID",
    "ROLE_ROLLUP_1",
    "ROLE_ROLLUP_2",
    "ROLE_DEPTNAME",
    "ROLE_TITLE",
    "ROLE_FAMILY_DESC",
    "ROLE_FAMILY",
    "ROLE_CODE",
]


def _load_local_kaggle_credentials() -> None:
    """Load Kaggle credentials from .env or local kaggle.json without printing secrets."""
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=Path(".env"))
    except ImportError:
        pass

    credentials_path = Path("kaggle.json")
    if not credentials_path.exists():
        return

    with credentials_path.open(encoding="utf-8") as file:
        credentials = json.load(file)

    username = str(credentials.get("username", "")).strip()
    key = str(credentials.get("key", "")).strip()
    if username and key:
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key


def _copy_competition_files(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)

    for csv_file in source.rglob("*.csv"):
        shutil.copy2(csv_file, destination / csv_file.name)

    for zip_file in source.rglob("*.zip"):
        with zipfile.ZipFile(zip_file) as archive:
            for member in archive.namelist():
                name = Path(member).name
                if name in {TRAIN_FILE, TEST_FILE}:
                    archive.extract(member, destination)
                    extracted = destination / member
                    if extracted != destination / name:
                        shutil.move(str(extracted), destination / name)
                        parent = extracted.parent
                        while parent != destination and parent.exists():
                            try:
                                parent.rmdir()
                            except OSError:
                                break
                            parent = parent.parent


def download_data(raw_dir: Path | str = RAW_DIR) -> Path:
    """Download Kaggle competition files and copy them into `data/raw`."""
    raw_path = Path(raw_dir)
    train_path = raw_path / TRAIN_FILE
    test_path = raw_path / TEST_FILE
    if train_path.exists() and test_path.exists():
        print(f"Data already present at {raw_path}")
        return raw_path

    _load_local_kaggle_credentials()
    import kagglehub

    try:
        downloaded = Path(kagglehub.competition_download("amazon-employee-access-challenge"))
    except Exception as exc:
        if exc.__class__.__name__ == "UnauthenticatedError":
            raise RuntimeError(
                "KaggleHub is not authenticated. Add a Kaggle API token and make sure "
                "the Amazon Employee Access Challenge rules are accepted on Kaggle."
            ) from exc
        raise
    _copy_competition_files(downloaded, raw_path)

    missing = [name for name in (TRAIN_FILE, TEST_FILE) if not (raw_path / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected Kaggle files after download: {missing}")

    print(f"Data saved to {raw_path}")
    return raw_path


def load_raw_train(raw_dir: Path | str = RAW_DIR) -> pd.DataFrame:
    """Load Kaggle training data."""
    return pd.read_csv(Path(raw_dir) / TRAIN_FILE)


def load_raw_test(raw_dir: Path | str = RAW_DIR) -> pd.DataFrame:
    """Load Kaggle test data."""
    return pd.read_csv(Path(raw_dir) / TEST_FILE)


def submission_ids(test: pd.DataFrame) -> pd.Series:
    """Return the Kaggle submission id column, falling back to one-based row ids."""
    if ID_COL in test.columns:
        return test[ID_COL]
    return pd.Series(range(1, len(test) + 1), name=ID_COL)
