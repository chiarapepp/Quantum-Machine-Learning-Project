from __future__ import annotations
from typing import Iterable, Literal, Union
import numpy as np

MeasurementBasis = Literal["X", "Z"]


def measurement_basis_for_architecture(architecture: str) -> MeasurementBasis:
    arch = architecture.strip().lower()
    if arch == "simple":
        return "X"
    if arch in {"ttn", "mera", "qcnn"}:
        return "Z"
    raise ValueError(f"Unknown architecture '{architecture}'.")


def certainty_from_expval(expval: Union[float, int]) -> float:
    """Map an expectation value to the certainty-factor range [-1, 1]."""
    return float(np.clip(float(expval), -1.0, 1.0))


def certainty_from_samples(samples: Iterable[Union[float, int]]) -> float:
    """Estimate certainty from Pauli measurement outcomes in {-1, +1}."""
    s = np.asarray(list(samples), dtype=float)
    if s.size == 0:
        raise ValueError("Empty sample array.")
    unique_vals = set(np.unique(s).tolist())
    if not unique_vals.issubset({-1.0, 1.0}):
        raise ValueError("Samples must be Pauli eigenvalues ±1.")
    return float(np.mean(s))


def predict_from_certainty(
    certainty: Union[float, int],
    positive_class_label: int = 0,
    negative_class_label: int = 1,
    threshold: float = 0.0,
) -> int:
    """
    Decision rule:
    certainty >= threshold -> positive_class_label
    certainty <  threshold -> negative_class_label
    """
    c = float(certainty)
    return positive_class_label if c >= float(threshold) else negative_class_label


def predict_many_from_certainty(
    certainties: Iterable[Union[float, int]],
    positive_class_label: int = 0,
    negative_class_label: int = 1,
    threshold: float = 0.0,
) -> np.ndarray:
    cs = np.asarray(list(certainties), dtype=float)
    return np.where(
        cs >= float(threshold),
        positive_class_label,
        negative_class_label,
    ).astype(int)


def confidence_from_certainty(certainty: Union[float, int]) -> float:
    """Unsigned confidence magnitude."""
    return float(abs(float(certainty)))


def confidences_from_certainties(certainties: Iterable[Union[float, int]]) -> np.ndarray:
    cs = np.asarray(list(certainties), dtype=float)
    return np.abs(cs)