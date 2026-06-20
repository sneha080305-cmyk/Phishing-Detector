"""
Dataset Loader
==============
Handles loading & preprocessing of phishing email datasets.

Supported datasets:
    - CEAS 2008  (CSV with 'body', 'label' columns)
    - Enron spam (folders: enron/ham/ and enron/spam/)
    - Custom CSV (any CSV with configurable column names)

Outputs a pandas DataFrame with columns:
    sender, subject, body_text, label (0=legit, 1=phishing)
"""

import os
import zipfile
import hashlib
from pathlib import Path

import pandas as pd
import requests
from loguru import logger
from tqdm import tqdm


DATA_DIR = Path(__file__).parent.parent / "data"


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_ceas2008(path: str | Path = None) -> pd.DataFrame:
    """
    Load CEAS 2008 dataset.

    Download from: https://monkey.org/~jose/phishing/
    Expected CSV columns: 'body', 'label' (1=spam/phish, 0=ham)
    """
    if path is None:
        path = DATA_DIR / "raw" / "ceas2008.csv"

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"CEAS 2008 dataset not found at {path}.\n"
            "Download it from: https://www.kaggle.com/datasets/rtatman/fraudulent-email-corpus\n"
            "or the CEAS 2008 proceedings dataset and place it at data/raw/ceas2008.csv"
        )

    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")

    # Normalise column names (dataset has varied formats)
    df.columns = [c.lower().strip() for c in df.columns]

    col_map = {}
    for col in df.columns:
        if col in ("body", "content", "email_body", "text"):
            col_map[col] = "body_text"
        elif col in ("label", "class", "spam", "is_phishing"):
            col_map[col] = "label"
        elif col in ("subject", "email_subject"):
            col_map[col] = "subject"
        elif col in ("from", "sender", "from_email"):
            col_map[col] = "sender"

    df = df.rename(columns=col_map)

    for col in ("body_text", "label"):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found. Columns: {list(df.columns)}")

    df = _fill_missing_cols(df)
    df["label"] = df["label"].astype(int)
    df = df.dropna(subset=["body_text"])
    df["body_text"] = df["body_text"].astype(str)

    logger.info(f"Loaded CEAS 2008: {len(df)} emails | "
                f"{df['label'].sum()} phishing | {(df['label']==0).sum()} legit")
    return df[["sender", "subject", "body_text", "label"]]


def load_enron(ham_dir: str | Path, spam_dir: str | Path) -> pd.DataFrame:
    """
    Load Enron spam/ham dataset from two directories.

    Each directory should contain raw .txt email files.
    """
    records = []

    for label, directory in [(0, ham_dir), (1, spam_dir)]:
        directory = Path(directory)
        files = list(directory.glob("*.txt")) + list(directory.glob("*.eml"))
        logger.info(f"Loading {len(files)} {'phishing' if label else 'legit'} emails from {directory}")

        for fpath in tqdm(files, desc=f"{'spam' if label else 'ham'}"):
            try:
                text = fpath.read_text(errors="replace")
                records.append({
                    "sender": "",
                    "subject": "",
                    "body_text": text,
                    "label": label
                })
            except Exception as e:
                logger.warning(f"Skipping {fpath.name}: {e}")

    df = pd.DataFrame(records)
    logger.info(f"Loaded Enron: {len(df)} emails")
    return df


def load_custom_csv(
    path: str | Path,
    text_col: str = "body",
    label_col: str = "label",
    sender_col: str = None,
    subject_col: str = None,
    phishing_label=1
) -> pd.DataFrame:
    """
    Load any CSV as a phishing dataset with configurable column mapping.

    Args:
        path: CSV file path
        text_col: column name containing email body text
        label_col: column name containing labels
        sender_col: column name for sender (optional)
        subject_col: column name for subject (optional)
        phishing_label: value in label_col that means 'phishing'
    """
    df = pd.read_csv(path, on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]

    records = pd.DataFrame({
        "sender": df[sender_col].astype(str) if sender_col and sender_col in df.columns else "",
        "subject": df[subject_col].astype(str) if subject_col and subject_col in df.columns else "",
        "body_text": df[text_col].astype(str),
        "label": (df[label_col] == phishing_label).astype(int)
    })

    records = records.dropna(subset=["body_text"])
    logger.info(f"Loaded custom CSV: {len(records)} emails | "
                f"{records['label'].sum()} phishing | {(records['label']==0).sum()} legit")
    return records


# ── Pipeline ──────────────────────────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame, extractor=None) -> tuple[pd.DataFrame, pd.Series]:
    """
    Convert a raw email DataFrame into an ML-ready feature matrix.

    Args:
        df: DataFrame from any loader above
        extractor: FeatureExtractor instance (created if not provided)

    Returns:
        X: feature DataFrame
        y: label Series
    """
    from modules.email_parser import EmailParser
    from modules.feature_extractor import FeatureExtractor

    parser = EmailParser()
    if extractor is None:
        extractor = FeatureExtractor()

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        try:
            doc = parser.parse({
                "sender": row.get("sender", ""),
                "subject": row.get("subject", ""),
                "body": row.get("body_text", "")
            })
            features = extractor.extract(doc)
            records.append(features)
        except Exception as e:
            logger.warning(f"Feature extraction error: {e}")
            records.append({})

    X = pd.DataFrame(records).fillna(0)
    y = df["label"].reset_index(drop=True)

    logger.info(f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    return X, y


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fill_missing_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("sender", "subject"):
        if col not in df.columns:
            df[col] = ""
    return df
