from __future__ import annotations
from typing import Dict, Iterable, List, Sequence
import numpy as np
import pandas as pd
import pennylane as qml


ANGLE_QUANTUM = np.deg2rad(0.25)  # every angle is a multiple of this, per the paper's encoding scheme

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

CATEGORICAL_FEATURES = [
    "PROTOCOL",
    "TCP_FLAGS",
    "L7_PROTO",
]

CONTINUOUS_FEATURES = [
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS",
]


class QuantumEncoder:
    """
    Classical-to-quantum feature encoder for NF-UNSW-NB15.

    Encoding rules follow the paper:
    - one qubit per feature
    - one RX rotation per feature
    - projected angle in [0, pi]
    - angle quantization with 0.25 degree granularity
    - low-cardinality features are mapped categorically
    - high-cardinality features are mapped by percentile binning

    Fit on the training set only, then reuse the fitted encoder on validation
    and test sets.
    """

    def __init__(self, n_bins: int = 100):
        if n_bins < 2:
            raise ValueError("n_bins must be at least 2.")

        self.n_bins = int(n_bins)
        self.cat_maps: Dict[str, Dict[float, int]] = {}
        self.percentile_edges: Dict[str, np.ndarray] = {}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "QuantumEncoder":

        self._validate_dataframe(df)

        for col in CATEGORICAL_FEATURES:
            values = pd.to_numeric(df[col], errors="raise").astype(float).to_numpy()
            unique_values = np.unique(values)
            self.cat_maps[col] = {
                float(value): idx for idx, value in enumerate(sorted(unique_values))
            }

        percentiles = np.linspace(0.0, 100.0, self.n_bins + 1)

        for col in CONTINUOUS_FEATURES:
            values = pd.to_numeric(df[col], errors="raise").astype(float).to_numpy()
            edges = np.percentile(values, percentiles)
            self.percentile_edges[col] = np.asarray(edges, dtype=float)

        self.is_fitted = True
        return self

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit the encoder on df and return the encoded angles."""
        return self.fit(df).transform(df)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Encode a DataFrame into an (N, 8) array of RX angles.
        """
        self._check_is_fitted()
        self._validate_dataframe(df)

        encoded_rows = [self.encode_sample(row) for _, row in df.iterrows()]
        return np.vstack(encoded_rows).astype(float)

    def encode_sample(self, row: pd.Series | Dict[str, float]) -> np.ndarray:
        """
        Encode one sample into eight angles.
        """
        self._check_is_fitted()

        angles: List[float] = []
        for col in FEATURE_COLUMNS:
            value = float(row[col])

            if col in CATEGORICAL_FEATURES:
                theta = self._encode_categorical(col, value)
            else:
                theta = self._encode_continuous(col, value)

            angles.append(theta)

        return np.asarray(angles, dtype=float)

    def _encode_categorical(self, col: str, value: float) -> float:
        """
        Map a low-cardinality feature categorically into [0, pi].
        """
        mapping = self.cat_maps[col]

        if value not in mapping:
            raise ValueError(
                f"Unseen categorical value {value} for column '{col}'. "
                "Fit the encoder on training data that contains this value "
                "or explicitly handle unseen values in your preprocessing."
            )

        index = mapping[value]
        n_unique = len(mapping)

        if n_unique == 1:
            theta = 0.0
        else:
            theta = (index / (n_unique - 1)) * np.pi

        return self._quantize(theta)

    def _encode_continuous(self, col: str, value: float) -> float:
        """
        Map a continuous feature by percentile binning.

        If [0, pi] is split into B bins and the value falls in bin k,
        then theta = k * pi / B, matching the paper's example.
        """
        edges = self.percentile_edges[col]

        bin_index = np.searchsorted(edges, value, side="right") - 1
        bin_index = int(np.clip(bin_index, 0, self.n_bins - 1))

        theta = (bin_index * np.pi) / self.n_bins
        return self._quantize(theta)

    def _quantize(self, theta: float) -> float:
        """
        Quantize an angle to the paper's 0.25 degree granularity.
        """
        quantized = np.round(theta / ANGLE_QUANTUM) * ANGLE_QUANTUM
        return float(np.clip(quantized, 0.0, np.pi))

def apply_rx_encoding(x: Sequence[float]) -> None:
    """
    Apply the paper's feature encoding circuit: one RX rotation per qubit.
    """
    for wire, theta in enumerate(x):
        qml.RX(theta, wires=wire)