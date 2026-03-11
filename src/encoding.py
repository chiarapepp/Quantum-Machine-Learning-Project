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

CATEGORICAL_FEATURES = ["PROTOCOL", "TCP_FLAGS"]

CONTINUOUS_FEATURES = [
    "L7_PROTO",
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS",
]


class QuantumEncoder:
    def __init__(self, n_bins: int = 100):
        self.n_bins = n_bins
        self.cat_maps = {}
        self.percentile_edges = {}

    def fit(self, df: pd.DataFrame):
        df = df[FEATURE_COLUMNS].copy()

        for col in CATEGORICAL_FEATURES:
            unique_values = sorted(df[col].astype(float).unique())
            self.cat_maps[col] = {v: i for i, v in enumerate(unique_values)}

        percentiles = np.linspace(0, 100, self.n_bins + 1)
        for col in CONTINUOUS_FEATURES:
            self.percentile_edges[col] = np.percentile(df[col].astype(float), percentiles)

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        df = df[FEATURE_COLUMNS].copy()
        encoded = []

        for _, row in df.iterrows():
            angles = []
            for col in FEATURE_COLUMNS:
                value = float(row[col])

                if col in CATEGORICAL_FEATURES:
                    mapping = self.cat_maps[col]
                    idx = mapping[value]
                    n = len(mapping)
                    theta = 0.0 if n == 1 else (idx / (n - 1)) * np.pi
                else:
                    edges = self.percentile_edges[col]
                    bin_index = np.searchsorted(edges, value, side="right") - 1
                    bin_index = int(np.clip(bin_index, 0, self.n_bins - 1))
                    theta = (bin_index * np.pi) / self.n_bins

                theta = np.round(theta / ANGLE_QUANTUM) * ANGLE_QUANTUM
                theta = float(np.clip(theta, 0.0, np.pi))
                angles.append(theta)

            encoded.append(angles)

        return np.asarray(encoded, dtype=float)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)


def apply_rx_encoding(x):
    for wire, theta in enumerate(x):
        qml.RX(theta, wires=wire)