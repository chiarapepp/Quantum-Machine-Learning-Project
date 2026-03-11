"""
Data loading and quantum encoding utilities.

- load the balanced processed CSV
- validate required columns
- perform the stratified 85/15 split
- fit the QuantumEncoder on the training split only
- transform train and test splits into angle-encoded arrays
"""

from __future__ import annotations
from typing import Any, Dict
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import encoding as enc_module


FEATURE_COLUMNS = list(enc_module.FEATURE_COLUMNS)
LABEL_COLUMN = "Label"

def split_processed_dataframe(
    df: pd.DataFrame,
    test_size: float = 0.15,
    random_state: int = 1,
) -> Dict[str, pd.DataFrame | np.ndarray]:
    """
    Split the processed DataFrame into train/test parts using stratified sampling.

    The paper uses:
    - test_size = 0.15
    - random_state = 1
    """
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be between 0 and 1.")

    X_all = df[FEATURE_COLUMNS].copy()
    y_all = df[LABEL_COLUMN].to_numpy(dtype=int)

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_all,
        y_all,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=y_all,
    )

    X_train_df = X_train_df.reset_index(drop=True)
    X_test_df = X_test_df.reset_index(drop=True)
    y_train = np.asarray(y_train, dtype=int)
    y_test = np.asarray(y_test, dtype=int)

    return {
        "X_train_df": X_train_df,
        "X_test_df": X_test_df,
        "y_train": y_train,
        "y_test": y_test,
    }


def load_encoded_splits(
    processed_csv: str,
    test_size: float = 0.15,
    random_state: int = 1,
    n_bins: int = 100,
) -> Dict[str, Any]:
    """
    Load the balanced processed CSV, split it, fit the QuantumEncoder on train only,
    and return encoded train/test arrays.
    """
    if not os.path.exists(processed_csv):
        raise FileNotFoundError(f"Processed CSV not found: {processed_csv}")

    df = pd.read_csv(processed_csv)

    split_pack = split_processed_dataframe(
        df,
        test_size=test_size,
        random_state=random_state,
    )

    X_train_df = split_pack["X_train_df"]
    X_test_df = split_pack["X_test_df"]
    y_train = split_pack["y_train"]
    y_test = split_pack["y_test"]

    encoder = enc_module.QuantumEncoder(n_bins=n_bins)
    X_train = encoder.fit_transform(X_train_df)
    X_test = encoder.transform(X_test_df)

    print(
        f"[data_utils] Encoded splits -> "
        f"train: {X_train.shape}, test: {X_test.shape}"
    )
    print(
        f"[data_utils] Label balance -> "
        f"train: {np.bincount(y_train)}, test: {np.bincount(y_test)}"
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_df": X_train_df,
        "X_test_df": X_test_df,
        "encoder": encoder,
    }


if __name__ == "__main__":
    sample_path = os.path.join("data", "processed", "nf_unsw_balanced.csv")

    if os.path.exists(sample_path):
        pack = load_encoded_splits(sample_path)
        print(
            "[data_utils] Loaded successfully:",
            pack["X_train"].shape,
            pack["X_test"].shape,
        )
    else:
        print(
            "No processed CSV found at data/processed/nf_unsw_balanced.csv. "
            "Generate it first with dataset.py."
        )