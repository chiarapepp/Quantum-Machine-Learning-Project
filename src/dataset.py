"""
Dataset loader & preprocessing for NF-UNSW-NB15 (NetFlow) as used in "Network Anomaly Detection Using 
Quantum Neural Networks on Noisy Quantum Computers".

Implements the data preparation steps as described in the paper:
 - use NF-UNSW-NB15 NetFlow CSV
 - keep eight features: IP protocol, tcp flags, layer 7 protocol,
   in byte count, out byte count, in packet count, out packet count, flow duration
 - drop IP and port features
 - cast layer7 protocol to float when needed
 - ensure no missing values (drop rows with NaNs)
 - resample benign flows to balance the dataset using sklearn.utils.resample
   with random_state=123 (downsample benign to match malicious count)
 - final balanced dataset has 144,812 samples (2 * 72,406)
 - perform a 15% test / 85% train split with random_state=1
"""

from typing import Tuple, List, Optional
import os
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = [
    "PROTOCOL",                 # ip protocol
    "TCP_FLAGS",                # tcp flags
    "L7_PROTO",                 # layer 7 protocol
    "IN_BYTES",                 # inbound bytes
    "OUT_BYTES",                # outbound bytes
    "IN_PKTS",                  # inbound packets
    "OUT_PKTS",                 # outbound packets
    "FLOW_DURATION_MILLISECONDS"  # flow duration
]

LABEL_COLUMN = "Label"

def load_and_prepare_nf_unsw(csv_path: str,
                             save_processed_csv: Optional[str] = None) \
                             -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the NF-UNSW-NB15 CSV, extract the 8 features used in the paper,
    balance the dataset by downsampling benign flows (random_state=123),
    and return train/test split (15% test / 85% train with random_state=1).

    Returns:
        X_train, X_test, y_train, y_test (all numpy arrays)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    original_count = len(df)
    print(f"[dataset] Loaded CSV '{csv_path}' with {original_count:,} rows and {len(df.columns)} columns.")

    # Ensure required columns exist
    missing = [c for c in FEATURE_COLUMNS + [LABEL_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    feature_cols = FEATURE_COLUMNS
    label_col = LABEL_COLUMN

    df_selected = df[feature_cols + [label_col]].copy()

    # Drop rows with any NaN (paper said dataset had no missing values; still be robust)
    n_before = len(df_selected)
    df_selected = df_selected.dropna()
    n_after_dropna = len(df_selected)
    if n_after_dropna != n_before:
        warnings.warn(f"Dropped {n_before - n_after_dropna} rows due to NaNs. Paper reported no missing values.")

    # Convert the layer 7 protocol column to float if possible (paper says they classified it as float)
    layer7_col = feature_cols[2]  # according to TARGET_FEATURE_KEYS order: third entry is layer 7
    # Some values may be strings like "<num>.<num>" — try to extract a numeric part
    def try_cast_layer7(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (int, float, np.number)):
            return float(v)
        s = str(v).strip()
        # try to extract a float from the string
        m = re.search(r"([-+]?\d*\.\d+|\d+)", s)
        if m:
            return float(m.group(0))
        # fallback: NaN
        return np.nan

    df_selected[layer7_col] = df_selected[layer7_col].apply(try_cast_layer7)
    if df_selected[layer7_col].isnull().any():
        # If any failed to parse, drop those rows (should be rare)
        n_before = len(df_selected)
        df_selected = df_selected.dropna(subset=[layer7_col])
        warnings.warn(f"Dropped {n_before - len(df_selected)} rows because layer7 protocol couldn't be parsed to float.")

    # Convert all features to numeric dtype
    for c in feature_cols:
        df_selected[c] = pd.to_numeric(df_selected[c], errors='coerce')
    # drop any rows that couldn't convert
    df_selected = df_selected.dropna()
    print(f"[dataset] After numeric conversion & cleaning: {len(df_selected):,} rows remain.")

    # Recode labels to binary: benign vs malicious
    # The paper says there are malicious samples (4.4%); we treat label != 0 as malicious.
    # But we must inspect distinct label values and map accordingly.
    unique_labels = sorted(df_selected[label_col].unique())
    print(f"[dataset] Unique label values in CSV: {unique_labels[:20]} (showing first 20)")

    # Heuristic: assume '0' or 'normal' indicates benign. Otherwise treat the smallest value as benign if unclear.
    def is_benign_label(v):
        if isinstance(v, str):
            vl = v.lower()
            if vl in ("normal", "benign", "0", "none"):
                return True
            # sometimes 'BENIGN' or 'Normal'
            if "normal" in vl or "benign" in vl:
                return True
            return False
        else:
            # numeric
            try:
                return int(v) == 0
            except Exception:
                return False

    # If there are textual labels like 'Normal' detect them:
    # Build binary label column: 0 = benign, 1 = malicious
    df_selected["_is_benign"] = df_selected[label_col].apply(lambda vv: is_benign_label(vv))
    # If no benign detected via heuristic, try another approach: assume smallest value is benign
    if df_selected["_is_benign"].sum() == 0:
        # assume numeric labels and benign is the smallest label
        min_label = min(unique_labels)
        df_selected["_binary_label"] = (df_selected[label_col] != min_label).astype(int)
        print(f"[dataset] No textual 'normal' labels found; assumed label {min_label} is benign.")
    else:
        df_selected["_binary_label"] = (~df_selected["_is_benign"]).astype(int)

    # Now count benign/malicious
    n_benign = int((df_selected["_binary_label"] == 0).sum())
    n_malicious = int((df_selected["_binary_label"] == 1).sum())
    print(f"[dataset] Counts before balancing -> benign: {n_benign:,}, malicious: {n_malicious:,}")

    # According to the paper: total original had 1,623,118 samples with 72,406 malicious (4.4%),
    # they resampled the benign NetFlows to produce a balanced dataset with 144,812 samples
    # i.e., final = 2 * 72,406 (downsample benign to match malicious count).
    # We'll implement exactly that: downsample benign to n_malicious samples (random_state=123).
    if n_malicious == 0:
        raise ValueError("No malicious samples found in the provided CSV. Check label column mapping.")
    # If benign count is larger, downsample benign to equal malicious; if smaller, upsample benign (rare).
    benign_df = df_selected[df_selected["_binary_label"] == 0]
    malicious_df = df_selected[df_selected["_binary_label"] == 1]

    if len(benign_df) >= n_malicious:
        benign_downsampled = resample(benign_df,
                                      replace=False,
                                      n_samples=n_malicious,
                                      random_state=123)
    else:
        # improbable for NF-UNSW, but handle by upsampling benign to match malicious (replace=True)
        benign_downsampled = resample(benign_df,
                                      replace=True,
                                      n_samples=n_malicious,
                                      random_state=123)

    balanced_df = pd.concat([malicious_df, benign_downsampled], axis=0).sample(frac=1.0, random_state=123).reset_index(drop=True)
    print(f"[dataset] After balancing: {len(balanced_df):,} rows (expected 144,812 if malicious=72,406).")

    # Verify size matches paper's claim when possible
    if len(balanced_df) != 144812 and n_malicious == 72406:
        warnings.warn(f"Paper reports final balanced dataset size 144,812 (2*72,406). "
                      f"We produced {len(balanced_df)} rows. If your CSV differs, this is expected.")

    # Extract features and binary labels
    X = balanced_df[feature_cols].to_numpy(dtype=float)
    y = balanced_df["_binary_label"].to_numpy(dtype=int)

    # Perform 15% test / 85% train split (paper uses random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=1, shuffle=True, stratify=y
    )

    print(f"[dataset] Split -> train: {len(X_train):,} rows, test: {len(X_test):,} rows.")

    # Optionally save processed CSV
    if save_processed_csv:
        out_df = pd.DataFrame(X, columns=feature_cols)
        out_df["label_binary"] = y
        out_df.to_csv(save_processed_csv, index=False)
        print(f"[dataset] Saved balanced processed CSV to: {save_processed_csv}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # quick local test: assumes data/raw/NF-UNSW-NB15-v2.csv relative to project root
    sample_path = os.path.join("data", "raw", "NF-UNSW-NB15-v2.csv")
    if os.path.exists(sample_path):
        Xtr, Xte, ytr, yte = load_and_prepare_nf_unsw(sample_path, save_processed_csv="data/processed/nf_unsw_balanced.csv")
        print("Shapes:", Xtr.shape, Xte.shape, ytr.shape, yte.shape)
    else:
        print("No CSV at", sample_path, " — place NF-UNSW-NB15-v2.csv at data/raw/ and run again.")
