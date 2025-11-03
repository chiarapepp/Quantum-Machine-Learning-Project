"""nf_dataset_interface.py

High-level dataset interface implementing the NF-UNSW-NB15 preprocessing
and encoding described in the paper "Network Anomaly Detection Using
Quantum Neural Networks on Noisy Quantum Computers".

Features:
- Wraps the existing `DataLoader` (in `dataset.py`) to produce angle-encoded
  feature matrices in [0, π] with 0.25° granularity.
- Provides helpers to save/load pre-encoded arrays (`.npz`) so experiments
  can be reproduced or run without reprocessing.
- Optional helper to convert angle matrices into Cirq circuits and TFQ tensors
  if `cirq` and `tensorflow_quantum` are available.

The implementation follows the paper's description for
encoding and label mapping (labels → {1, -1} for hinge loss).

Usage:
    from nf_dataset_interface import NFDataInterface
    d = NFDataInterface('path/to/NF-UNSW-NB15.csv')
    X_train_a, X_test_a, y_train, y_test = d.prepare_data()
    d.save_npz('encoded_angles.npz')

Notes:
- This module intentionally avoids importing TFQ at top-level so it can be
  used in environments without TFQ; the TFQ-related helper will raise a
  clear ImportError if TFQ is not installed.
"""

from typing import Tuple, Optional
import numpy as np
import os

from dataset import DataLoader


class NFDataInterface:
    """High-level interface to prepare and persist the NF-UNSW-NB15 data
    according to the paper's preprocessing and encoding.
    """

    def __init__(self, csv_path: str, random_state: int = 123):
        self.csv_path = csv_path
        self.random_state = random_state
        self.loader = DataLoader(data_path=csv_path, random_state=random_state)

        # Will be filled after prepare_data()
        self.X_train_angles = None
        self.X_test_angles = None
        self.y_train = None
        self.y_test = None

    def prepare_data(self, test_size: float = 0.15, sample_limit: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load, balance, split, encode and return angle-encoded arrays.

        Args:
            test_size: fraction of balanced dataset to use as test set (paper used 15%).
            sample_limit: optional limit on number of balanced samples to process
                (useful for debugging / quick runs).

        Returns:
            X_train_angles, X_test_angles, y_train, y_test
        """
        X_train_a, X_test_a, y_train, y_test = self.loader.load_and_preprocess(test_size=test_size)

        # Optionally limit samples (keep same proportion)
        if sample_limit is not None:
            # limit both train and test proportionally
            n_train = min(len(X_train_a), max(1, int(sample_limit * (1 - test_size))))
            n_test = min(len(X_test_a), max(1, int(sample_limit * test_size)))
            X_train_a = X_train_a[:n_train]
            y_train = y_train[:n_train]
            X_test_a = X_test_a[:n_test]
            y_test = y_test[:n_test]

        # store
        self.X_train_angles = X_train_a
        self.X_test_angles = X_test_a
        self.y_train = y_train
        self.y_test = y_test

        return X_train_a, X_test_a, y_train, y_test

    def save_npz(self, path: str):
        """Save the encoded arrays and encoding tables to a compressed npz file.

        The file will contain:
            X_train_angles, X_test_angles, y_train, y_test, encoding_tables

        """
        if self.X_train_angles is None:
            raise RuntimeError("No data prepared. Call prepare_data() first.")

        encoding_tables = self.loader.encoding_tables
        # NumPy can't save dicts of dicts cleanly in a compact way; save as object
        np.savez_compressed(path,
                            X_train_angles=self.X_train_angles,
                            X_test_angles=self.X_test_angles,
                            y_train=self.y_train,
                            y_test=self.y_test,
                            encoding_tables=encoding_tables)

    def load_npz(self, path: str):
        """Load previously saved encoded arrays (npz from save_npz).

        Populates internal attributes and returns the arrays.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        with np.load(path, allow_pickle=True) as data:
            self.X_train_angles = data['X_train_angles']
            self.X_test_angles = data['X_test_angles']
            self.y_train = data['y_train']
            self.y_test = data['y_test']
            encoding_tables = data['encoding_tables'].item()
            self.loader.encoding_tables = encoding_tables

        return self.X_train_angles, self.X_test_angles, self.y_train, self.y_test

    def get_tfq_circuit_tensor(self, which: str = 'train'):
        """Convert stored angle arrays into a TFQ circuit tensor.

        This helper requires `cirq` and `tensorflow_quantum` to be installed
        and will raise an informative ImportError otherwise.

        Args:
            which: 'train' or 'test' to choose which set to convert.

        Returns:
            (circuit_tensor, y_array)
        """
        try:
            import cirq
            import tensorflow_quantum as tfq
        except Exception as e:
            raise ImportError("cirq and tensorflow_quantum are required to convert to TFQ tensors. Install them to use this helper.") from e

        # Lazy import of the qnn skeleton helper to build circuits
        try:
            from qnn_nids_skeleton import circuits_from_angle_matrix
        except Exception as e:
            raise ImportError("Could not import circuits_from_angle_matrix from qnn_nids_skeleton. Ensure that file exists and is importable.") from e

        if which == 'train':
            if self.X_train_angles is None:
                raise RuntimeError("No training data. Call prepare_data() first.")
            angles = self.X_train_angles
            y = self.y_train
        elif which == 'test':
            if self.X_test_angles is None:
                raise RuntimeError("No test data. Call prepare_data() first.")
            angles = self.X_test_angles
            y = self.y_test
        else:
            raise ValueError("which must be 'train' or 'test'")

        circuits = circuits_from_angle_matrix(angles)
        tensor = tfq.convert_to_tensor(circuits)
        return tensor, y

    def certainty_from_probabilities(self, probabilities: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute certainty factor using loader's utility (probabilities -> C).

        Probabilities are expected to be P(|0>) as in some measurement conventions.
        The loader implements the paper's formulation that maps y_true {1,-1}.
        """
        return self.loader.calculate_certainty_factor(probabilities, y_true)


if __name__ == '__main__':
    # Quick CLI example: prepare data from CSV and save to npz
    import argparse

    p = argparse.ArgumentParser(description='Prepare NF-UNSW-NB15 encoded angles (paper-style)')
    p.add_argument('csv', help='Path to NF-UNSW-NB15 CSV')
    p.add_argument('--out', default='nf_encoded.npz', help='Output .npz file')
    p.add_argument('--limit', type=int, default=None, help='Optional sample limit for quick runs')
    args = p.parse_args()

    iface = NFDataInterface(args.csv)
    print('Preparing data... this may take a while for full dataset')
    iface.prepare_data(sample_limit=args.limit)
    print('Saving encoded arrays to', args.out)
    iface.save_npz(args.out)
    print('Done.')
