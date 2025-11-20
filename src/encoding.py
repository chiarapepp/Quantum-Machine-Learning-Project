import numpy as np
import pennylane as qml


# 0.25 degree in radians (rotation granularity)
ANGLE_QUANTUM = np.deg2rad(0.25)  # ≈ 0.004363323

# 8 features 
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
    """
    Implements the encoding procedure described in the paper:
    - one qubit per feature
    - RX rotations only
    - angle in [0, π]
    - categorical features → evenly spaced values
    - continuous features → percentile binning
    - quantization to granularity = 0.25° (0.004363323 rad)
    """

    def __init__(self, df, n_bins=100):
        """
        df: pandas DataFrame containing the 8 features.
        n_bins: number of percentile bins for continuous features.
        """
        self.df = df
        self.n_bins = n_bins
        self.bin_edges = {}
        self.cat_maps = {}

        # Prepare encodings
        self._prepare_categorical()
        self._prepare_continuous()

    #  Categorical preparation
    def _prepare_categorical(self):
        for col in CATEGORICAL_FEATURES:
            values = sorted(self.df[col].unique())
            self.cat_maps[col] = {v: i for i, v in enumerate(values)}

    #  Continuous preparation
    def _prepare_continuous(self):
        for col in CONTINUOUS_FEATURES:
            # compute percentile bins (e.g., 0..100)
            percentiles = np.linspace(0, 100, self.n_bins + 1)
            self.bin_edges[col] = np.percentile(self.df[col], percentiles)

    #  Value → θ mapping
    def _encode_categorical(self, col, value):
        mapping = self.cat_maps[col]
        n_unique = len(mapping)
        index = mapping[value]
        # Mapping: evenly spaced in [0, π]
        if n_unique > 1:
            theta = (index / (n_unique - 1)) * np.pi
        else:
            theta = 0.0
        return self._quantize(theta)

    def _encode_continuous(self, col, value):
        edges = self.bin_edges[col]
        # find which bin value belongs to
        bin_index = np.searchsorted(edges, value, side="right") - 1
        bin_index = np.clip(bin_index, 0, self.n_bins - 1)
        # map percent → angle
        percentile = (bin_index / self.n_bins)
        theta = percentile * np.pi
        return self._quantize(theta)

    def _quantize(self, theta):
        # paper: quantize to multiples of 0.25°
        return np.round(theta / ANGLE_QUANTUM) * ANGLE_QUANTUM


    def encode_sample(self, row):
        """Return an array of 8 angles for one sample."""
        angles = []
        for col in FEATURES:
            val = row[col]
            if col in CATEGORICAL_FEATURES:
                angles.append(self._encode_categorical(col, val))
            else:
                angles.append(self._encode_continuous(col, val))
        return np.array(angles)

    def encode_dataset(self):
        """Encode entire DataFrame -> Nx8 numpy array"""
        arr = np.vstack([self.encode_sample(row) for _, row in self.df.iterrows()])
        return arr


# Quantum circuit encoding
def apply_rx_encoding(x):
    """
    Apply RX(θ) on each qubit according to encoded angles x.
    """
    for i, theta in enumerate(x):
        qml.RX(theta, wires=i)
