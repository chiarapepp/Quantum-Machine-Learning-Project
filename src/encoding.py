"""
Feature preprocessing and angle encoding for quantum inputs.

This module defines the paper-aligned feature set, computes percentile-based
binning thresholds from training data, maps classical features to rotation
angles, and applies RX encoding on quantum wires.
"""

import numpy as np
import pandas as pd
import pennylane as qml

# every angle is a multiple of this, per the paper's encoding scheme
ANGLE_QUANTUM = np.deg2rad(0.25)

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

# Features with a limited number of unique values → categorical encoding
CATEGORICAL_FEATURES = ["PROTOCOL", "TCP_FLAGS"]

# Features with many unique values → percentile-based binning
CONTINUOUS_FEATURES = [
    "L7_PROTO",
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS",
]

N_QUBITS = len(FEATURE_COLUMNS)  # one qubit per feature → 8 qubits


class QuantumEncoder:
    """
    Encode the 8 selected NF-UNSW features into RX rotation angles.

    Encoding rule:
    - features with few unique values are mapped categorically to [0, pi]
    - features with many unique values are mapped by percentile binning
    - all resulting angles are quantized to 0.25-degree steps

    The output is an array of shape (n_samples, 8), with one angle per feature
    and therefore one qubit per feature.
    """
    def __init__(self, n_bins: int = 100):
        self.n_bins = n_bins
        self.cat_maps: dict = {}          # col → {value: bin_index}
        self.percentile_edges: dict = {}  # col → np.ndarray of bin edges

    def fit(self, df: pd.DataFrame):
        df = df[FEATURE_COLUMNS].copy()

        for col in CATEGORICAL_FEATURES:
            unique_values = sorted(df[col].astype(float).unique())
            self.cat_maps[col] = {v: i for i, v in enumerate(unique_values)}

        percentiles = np.linspace(0, 100, self.n_bins + 1)
        for col in CONTINUOUS_FEATURES:
            self.percentile_edges[col] = np.percentile(
                df[col].astype(float), percentiles
            )

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        df = df[FEATURE_COLUMNS].copy()
        n = len(df)

        # Initialize angles matrix (n_samples, n_features)
        angles = np.zeros((n, N_QUBITS), dtype=float)

        for j, col in enumerate(FEATURE_COLUMNS):
            col_vals = df[col].astype(float).values

            if col in CATEGORICAL_FEATURES:
                mapping = self.cat_maps[col]
                n_cats = len(mapping)

                idx_arr = np.array([mapping[v] for v in col_vals])

                theta_arr = (
                    np.zeros(n) if n_cats == 1
                    else (idx_arr / (n_cats - 1)) * np.pi
                )

            else:
                edges = self.percentile_edges[col]

                bin_indices = np.searchsorted(edges, col_vals, side="right") - 1
                bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)

                theta_arr = (bin_indices * np.pi) / self.n_bins

            # Quantize to nearest 0.25° step
            theta_arr = np.round(theta_arr / ANGLE_QUANTUM) * ANGLE_QUANTUM

            # clipping to [0, π] (to handle any potential numerical issues)
            angles[:, j] = np.clip(theta_arr, 0.0, np.pi)

        return angles

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)


# x is an array of shape (n_features,) containing the angles for each feature/qubit 
# and is the output of the QuantumEncoder.transform() method
def apply_rx_encoding(x: np.ndarray) -> None:
    qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation="X")