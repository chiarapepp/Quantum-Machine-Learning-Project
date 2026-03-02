import numpy as np
import pennylane as qml

# 0.25 degree in radians (rotation granularity)
ANGLE_QUANTUM = np.deg2rad(0.25)  # ≈ 0.004363323

# 8 features (names expected in the DataFrame passed to the encoder)
FEATURES = [
    "ip_protocol",
    "tcp_flags",
    "layer7_protocol",
    "in_bytes",
    "out_bytes",
    "in_packets",
    "out_packets",
    "flow_duration",
]

# Features that are categorical-ish (few unique values)
CATEGORICAL_FEATURES = [
    "ip_protocol",
    "tcp_flags",
    "layer7_protocol",
]

# Continuous numeric features → percentile binning
CONTINUOUS_FEATURES = [
    "in_bytes",
    "out_bytes",
    "in_packets",
    "out_packets",
    "flow_duration",
]


class QuantumEncoder:
    """Paper-faithful feature→angle encoder.

    Implements the encoding procedure described in the paper:
      - one qubit per feature
      - RX rotations only
      - angle in [0, π]
      - categorical features → evenly spaced angles
      - continuous features → percentile binning
      - quantization to granularity = 0.25° (ANGLE_QUANTUM)

    Notes:
      - Fit the encoder on TRAIN data only, then use encode_dataset(test_df)
        to avoid data leakage.
      - Unseen categorical values at transform time are mapped to the first
        known category (index 0) to avoid KeyErrors.
    """

    def __init__(self, df, n_bins: int = 100):
        """Fit encoder statistics from a pandas DataFrame containing FEATURES."""
        self.n_bins = int(n_bins)
        if self.n_bins < 2:
            raise ValueError("n_bins must be >= 2")

        self.bin_edges = {}  # col -> np.ndarray of edges length (n_bins+1)
        self.cat_maps = {}   # col -> dict(value -> index)
        self._fit(df)

    def _fit(self, df):
        # Prepare encodings using the provided df (ideally: training set only)
        self._prepare_categorical(df)
        self._prepare_continuous(df)

    # ---- Categorical preparation
    def _prepare_categorical(self, df):
        for col in CATEGORICAL_FEATURES:
            values = sorted(df[col].unique())
            self.cat_maps[col] = {v: i for i, v in enumerate(values)}

    # ---- Continuous preparation
    def _prepare_continuous(self, df):
        for col in CONTINUOUS_FEATURES:
            percentiles = np.linspace(0, 100, self.n_bins + 1)
            self.bin_edges[col] = np.percentile(df[col], percentiles)

    # ---- Quantization helper
    def _quantize(self, theta: float) -> float:
        # Paper: quantize to multiples of 0.25°
        qt = np.round(theta / ANGLE_QUANTUM) * ANGLE_QUANTUM
        # Keep within [0, π] after rounding
        return float(np.clip(qt, 0.0, np.pi))

    # ---- Value → θ mapping
    def _encode_categorical(self, col, value):
        mapping = self.cat_maps[col]
        n_unique = len(mapping)

        # Robustness: unseen categories in test → map to index 0
        index = mapping.get(value, 0)

        # Evenly spaced in [0, π]
        if n_unique > 1:
            theta = (index / (n_unique - 1)) * np.pi
        else:
            theta = 0.0
        return self._quantize(theta)

    def _encode_continuous(self, col, value):
        edges = self.bin_edges[col]
        # find which bin value belongs to
        bin_index = np.searchsorted(edges, value, side="right") - 1
        bin_index = int(np.clip(bin_index, 0, self.n_bins - 1))

        # Map bin index to [0, π] (inclusive)
        denom = (self.n_bins - 1)
        theta = (bin_index / denom) * np.pi if denom > 0 else 0.0
        return self._quantize(theta)

    # ---- Public API
    def encode_sample(self, row):
        """Return an array of 8 angles for one sample (row can be Series or dict-like)."""
        angles = []
        for col in FEATURES:
            val = row[col]
            if col in CATEGORICAL_FEATURES:
                angles.append(self._encode_categorical(col, val))
            else:
                angles.append(self._encode_continuous(col, val))
        return np.array(angles, dtype=float)

    def encode_dataset(self, df):
        """Encode a DataFrame -> (N, 8) numpy array of angles."""
        # iterrows is fine for clarity; can be optimized later if needed
        return np.vstack([self.encode_sample(row) for _, row in df.iterrows()])


# Quantum circuit encoding
def apply_rx_encoding(x):
    """Apply RX(θ) on each qubit according to encoded angles x."""
    for i, theta in enumerate(x):
        qml.RX(theta, wires=i)
