from __future__ import annotations
from typing import Optional
import os
import warnings
import pandas as pd
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


def build_processed_nf_unsw(
    csv_path: str,
    save_processed_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load the raw NF-UNSW-NB15 CSV, keep the 8 paper features, clean invalid rows,
    balance the classes by downsampling benign traffic, and return the balanced
    processed DataFrame.

    This function does NOT perform any train/test split.
    Splitting is delegated to data_utils.py.

    Parameters
    ----------
    csv_path:
        Path to the raw NF-UNSW-NB15 CSV file.
    save_processed_csv:
        Optional output path where the balanced processed CSV will be saved.

    Returns
    -------
    pd.DataFrame
        Balanced processed DataFrame with columns:
        FEATURE_COLUMNS + [LABEL_COLUMN]
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
        warnings.warn(f"Dropped {n_dropped_nan} rows with missing values.")

    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Non-numeric labels become NaN and will be dropped below.
    df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors="coerce")

    n_before = len(df)
    df = df.dropna()
    n_dropped_non_numeric = n_before - len(df)
    if n_dropped_non_numeric > 0:
        warnings.warn(f"Dropped {n_dropped_non_numeric} rows with non-numeric values.")

    # enforce numeric dtypes explicitly
    for col in FEATURE_COLUMNS:
        df[col] = df[col].astype(float)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

    unique_labels = sorted(df[LABEL_COLUMN].unique().tolist())
    if unique_labels != [0, 1]:
        raise ValueError(
            f"Expected Label column to contain only [0, 1], found {unique_labels}."
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
            "This loader expects the original NF-UNSW-NB15 imbalance."
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

    if save_processed_csv is not None:
        out_dir = os.path.dirname(save_processed_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        balanced_df.to_csv(save_processed_csv, index=False)
        print(f"[dataset] Saved balanced processed CSV to: {save_processed_csv}")

    return balanced_df


if __name__ == "__main__":
    sample_path = os.path.join("data", "raw", "NF-UNSW-NB15-v2.csv")
    output_path = os.path.join("data", "processed", "nf_unsw_balanced.csv")

    if os.path.exists(sample_path):
        df_processed = build_processed_nf_unsw(
            sample_path,
            save_processed_csv=output_path,
        )
        print(f"[dataset] Final processed shape: {df_processed.shape}")
    else:
        print(
            "No CSV found at data/raw/NF-UNSW-NB15-v2.csv. "
            "Place the raw file there and run again."
        )