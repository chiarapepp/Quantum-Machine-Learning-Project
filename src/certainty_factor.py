from __future__ import annotations

from typing import Iterable, Literal
import numpy as np


MeasurementBasis = Literal["X", "Z"]


def measurement_basis_for_architecture(architecture: str) -> MeasurementBasis:
    """
    Return the measurement basis associated with the chosen architecture.

    Conventions:
    - simple -> X basis
    - ttn, mera, qcnn -> Z basis
    """
    arch = architecture.strip().lower()

    if arch == "simple":
        return "X"
    if arch in {"ttn", "mera", "qcnn"}:
        return "Z"

    raise ValueError(
        f"Unknown architecture '{architecture}'. "
        "Supported values: simple, ttn, mera, qcnn."
    )


def certainty_from_expval(expval: float) -> float:
    """
    Convert an expectation value into a certainty factor in [-1, 1].

    In noiseless simulation, the QNN output is already an expectation value
    of the measured Pauli observable, so the certainty factor is simply that
    value, clipped for numerical safety.
    """
    return float(np.clip(expval, -1.0, 1.0))


def certainty_from_samples(samples: Iterable[float]) -> float:
    """
    Estimate the certainty factor from shot-based Pauli outcomes in {-1, +1}.

    This is useful for noisy simulation and hardware-style inference, where
    the prediction is based on repeated measurements.
    """
    arr = np.asarray(list(samples), dtype=float)

    if arr.size == 0:
        raise ValueError("samples must not be empty.")

    unique_values = set(np.unique(arr).tolist())
    if not unique_values.issubset({-1.0, 1.0}):
        raise ValueError("samples must contain only Pauli outcomes in {-1, +1}.")

    return float(np.mean(arr))


def predict_from_certainty(
    certainty: float,
    threshold: float = 0.0,
    benign_label: int = 0,
    malicious_label: int = 1,
) -> int:
    """
    Convert one certainty factor into a class prediction.

    Decision rule:
    - certainty >= threshold -> benign_label
    - certainty <  threshold -> malicious_label
    """
    return benign_label if float(certainty) >= threshold else malicious_label


def predict_many_from_certainty(
    certainties: Iterable[float],
    threshold: float = 0.0,
    benign_label: int = 0,
    malicious_label: int = 1,
) -> np.ndarray:
    """
    Vectorized prediction from certainty factors.
    """
    arr = np.asarray(certainties, dtype=float)

    return np.where(
        arr >= threshold,
        benign_label,
        malicious_label,
    ).astype(int)


def confidence_from_certainty(certainty: float) -> float:
    """
    Unsigned confidence derived from certainty.

    Values near 0 mean low confidence.
    Values near 1 mean high confidence.
    """
    return float(abs(certainty))


def confidences_from_certainties(certainties: Iterable[float]) -> np.ndarray:
    """
    Vectorized confidence computation.
    """
    return np.abs(np.asarray(certainties, dtype=float))