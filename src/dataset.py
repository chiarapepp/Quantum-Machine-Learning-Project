"""
Dataset loader for NF-UNSW-NB15 NetFlow data used in
"Network Anomaly Detection Using Quantum Neural Networks on Noisy Quantum Computers".

This module:
- loads the NF-UNSW-NB15 NetFlow CSV
- keeps the 8 features used in the paper
- uses the dataset Label column directly (0 = benign, 1 = malicious)
- removes rows with missing or non-numeric feature values
- balances the dataset by downsampling benign flows with random_state=123
- performs an 85/15 train-test split with random_state=1
"""

from __future__ import annotations

from typing import Optional, Tuple
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


FEATURE_COLUMNS = [
    "PROTOCOL",
    "TCP_FLAGS",
    "L7_PROTO",
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS",
]

LABEL_COLUMN = "Label"


def load_and_prepare_nf_unsw(
    csv_path: str,
    save_processed_csv: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load NF-UNSW-NB15 NetFlow data, keep the paper features, balance the classes,
    and return train/test numpy arrays.

    Parameters
    ----------
    csv_path:
        Path to the raw NF-UNSW-NB15 CSV file.
    save_processed_csv:
        Optional output path for the balanced processed CSV.

    Returns
    -------
    X_train, X_test, y_train, y_test
        Train/test split as numpy arrays.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(
        f"[dataset] Loaded CSV '{csv_path}' with "
        f"{len(df):,} rows and {len(df.columns)} columns."
    )

    required_columns = FEATURE_COLUMNS + [LABEL_COLUMN]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df[required_columns].copy()

    n_before = len(df)
    df = df.dropna()
    n_dropped_nan = n_before - len(df)
    if n_dropped_nan > 0:
        warnings.warn(
            f"Dropped {n_dropped_nan} rows with missing values."
        )

    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors="coerce")

    n_before = len(df)
    df = df.dropna()
    n_dropped_non_numeric = n_before - len(df)
    if n_dropped_non_numeric > 0:
        warnings.warn(
            f"Dropped {n_dropped_non_numeric} rows with non-numeric values."
        )

    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

    unique_labels = sorted(df[LABEL_COLUMN].unique().tolist())
    if unique_labels != [0, 1]:
        raise ValueError(
            f"Expected Label column to contain only [0, 1], found {unique_labels}"
        )

    n_benign = int((df[LABEL_COLUMN] == 0).sum())
    n_malicious = int((df[LABEL_COLUMN] == 1).sum())

    print(
        f"[dataset] Counts before balancing -> "
        f"benign: {n_benign:,}, malicious: {n_malicious:,}"
    )

    if n_benign == 0 or n_malicious == 0:
        raise ValueError("Both benign and malicious samples must be present.")

    benign_df = df[df[LABEL_COLUMN] == 0]
    malicious_df = df[df[LABEL_COLUMN] == 1]

    if len(benign_df) < len(malicious_df):
        raise ValueError(
            "Benign class has fewer samples than malicious class. "
            "This loader expects the original NF-UNSW-NB15 class imbalance."
        )

    benign_downsampled = resample(
        benign_df,
        replace=False,
        n_samples=len(malicious_df),
        random_state=123,
    )

    balanced_df = (
        pd.concat([malicious_df, benign_downsampled], axis=0)
        .sample(frac=1.0, random_state=123)
        .reset_index(drop=True)
    )

    print(f"[dataset] After balancing: {len(balanced_df):,} rows.")

    if len(malicious_df) == 72406 and len(balanced_df) != 144812:
        warnings.warn(
            "Expected 144,812 rows after balancing when malicious samples are 72,406, "
            f"but found {len(balanced_df):,}."
        )

    X = balanced_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = balanced_df[LABEL_COLUMN].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=1,
        shuffle=True,
        stratify=y,
    )

    print(
        f"[dataset] Split -> train: {len(X_train):,} rows, "
        f"test: {len(X_test):,} rows."
    )

    if save_processed_csv is not None:
        out_dir = os.path.dirname(save_processed_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        processed_df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
        processed_df[LABEL_COLUMN] = y
        processed_df.to_csv(save_processed_csv, index=False)
        print(f"[dataset] Saved balanced processed CSV to: {save_processed_csv}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    sample_path = os.path.join("data", "raw", "NF-UNSW-NB15-v2.csv")

    if os.path.exists(sample_path):
        X_train, X_test, y_train, y_test = load_and_prepare_nf_unsw(
            sample_path,
            save_processed_csv="data/processed/nf_unsw_balanced.csv",
        )
        print("Shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    else:
        print(
            "No CSV found at data/raw/NF-UNSW-NB15-v2.csv. "
            "Place the raw file there and run again."
        )