# src/data_utils.py
"""
Data loading and quantum encoding utilities.

Separates data preparation from training logic so it can be reused by
train.py, noise_eval.py, and run_experiment.py without duplication.
"""

from __future__ import annotations
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import encoding as enc_module


# Column names in the processed CSV (output of dataset.py)
_RAW_COLS = [
    "PROTOCOL",
    "TCP_FLAGS",
    "L7_PROTO",
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS",
]
_ENC_COLS = enc_module.FEATURES   # canonical encoder order (same 8 features, renamed)

_RENAME = dict(zip(_RAW_COLS, _ENC_COLS))


def load_encoded_splits(
    processed_csv: str,
    test_size: float = 0.15,
    random_state: int = 1,
    n_bins: int = 100,
) -> Dict[str, np.ndarray]:
    """
    Load the balanced processed CSV, rename columns to encoder names,
    split 85/15, fit the QuantumEncoder on training data only,
    and return angle-encoded arrays.

    Parameters
    ----------
    processed_csv : path to the output of dataset.load_and_prepare_nf_unsw()
    test_size     : fraction for test split (paper: 0.15)
    random_state  : paper uses 1
    n_bins        : percentile bins for continuous features (paper: 100)

    Returns
    -------
    dict with X_train, y_train, X_test, y_test  (all numpy arrays)
    """
    df = pd.read_csv(processed_csv)

    # Accept both raw column names and already-renamed names
    if "PROTOCOL" in df.columns:
        df = df.rename(columns=_RENAME)

    # Validate
    missing = [c for c in _ENC_COLS + ["label_binary"] if c not in df.columns]
    if missing:
        raise ValueError(f"Processed CSV missing columns: {missing}. Present: {list(df.columns)}")

    X_all = df[_ENC_COLS].copy()
    y_all = df["label_binary"].to_numpy(dtype=int)

    # Stratified split — matches paper (random_state=1)
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=random_state, shuffle=True, stratify=y_all,
    )

    train_df = X_all.iloc[train_idx].reset_index(drop=True)
    test_df  = X_all.iloc[test_idx].reset_index(drop=True)
    y_train  = y_all[train_idx]
    y_test   = y_all[test_idx]

    # Fit encoder on TRAIN only (prevents data leakage)
    encoder = enc_module.QuantumEncoder(train_df, n_bins=n_bins)

    X_train = encoder.encode_dataset(train_df)
    X_test  = encoder.encode_dataset(test_df)

    print(
        f"[data_utils] train={X_train.shape}  test={X_test.shape}  "
        f"label_balance train={np.bincount(y_train)}  test={np.bincount(y_test)}"
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test":  X_test,
        "y_test":  y_test,
    }