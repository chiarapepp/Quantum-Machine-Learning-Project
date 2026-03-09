from __future__ import annotations

from typing import Iterable, Literal, Union

import numpy as np


MeasurementBasis = Literal["X", "Z"]
ArchitectureName = Literal["simple", "ttn", "mera", "qcnn"]


def measurement_basis_for_architecture(architecture: str) -> MeasurementBasis:
    """
    Return the readout basis used by the given architecture.

    Paper-consistent convention:
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
        "Supported values: 'simple', 'ttn', 'mera', 'qcnn'."
    )


def certainty_from_expval(expval: Union[float, int]) -> float:
    """
    Convert an expectation value into the certainty factor C.

    In this project, the certainty factor is the expectation value of the
    final measured Pauli observable:
    - Simple: C = <X>
    - TTN / MERA / QCNN: C = <Z>

    The result is clipped to [-1, 1] to absorb small numerical noise.
    """
    c = float(expval)
    return float(np.clip(c, -1.0, 1.0))


def certainty_from_expval_with_architecture(
    expval: Union[float, int],
    architecture: str,
) -> float:
    """
    Same as certainty_from_expval(), but validates the architecture name.
    """
    _ = measurement_basis_for_architecture(architecture)
    return certainty_from_expval(expval)


def certainty_from_samples(samples: Iterable[Union[float, int]]) -> float:
    """
    Estimate the certainty factor from sampled Pauli eigenvalues.

    Expected input:
    - samples from qml.sample(qml.PauliX(...)) for Simple
    - samples from qml.sample(qml.PauliZ(...)) for TTN / MERA / QCNN

    PennyLane returns eigenvalues in {-1, +1}, and the certainty factor is
    their mean.
    """
    s = np.asarray(list(samples), dtype=float)

    if s.size == 0:
        raise ValueError("certainty_from_samples received an empty sample array.")

    unique_vals = set(np.unique(s).tolist())
    if not unique_vals.issubset({-1.0, 1.0}):
        raise ValueError("Samples must contain only Pauli eigenvalues ±1.")

    return float(np.mean(s))


def certainty_from_samples_with_architecture(
    samples: Iterable[Union[float, int]],
    architecture: str,
) -> float:
    """
    Same as certainty_from_samples(), but validates the architecture name.
    """
    _ = measurement_basis_for_architecture(architecture)
    return certainty_from_samples(samples)


def predicted_label_from_certainty(
    certainty: Union[float, int],
    zero_class_label: int = 0,
    one_class_label: int = 1,
    threshold: float = 0.0,
) -> int:
    """
    Convert a certainty factor into a predicted label.

    Default paper-consistent rule:
    - C >= 0 -> label 0
    - C <  0 -> label 1
    """
    c = float(certainty)
    return zero_class_label if c >= float(threshold) else one_class_label


def predicted_labels_from_certainties(
    certainties: Iterable[Union[float, int]],
    zero_class_label: int = 0,
    one_class_label: int = 1,
    threshold: float = 0.0,
) -> np.ndarray:
    """
    Vectorized helper that converts a sequence of certainty factors into
    a numpy array of predicted labels.
    """
    cs = np.asarray(list(certainties), dtype=float)
    return np.where(cs >= float(threshold), zero_class_label, one_class_label).astype(int)


def confidence_from_certainty(certainty: Union[float, int]) -> float:
    """
    Return the confidence magnitude associated with the certainty factor.

    This is |C| in [0, 1]:
    - 0 -> maximally uncertain
    - 1 -> maximally confident
    """
    return float(abs(float(certainty)))


def confidences_from_certainties(certainties: Iterable[Union[float, int]]) -> np.ndarray:
    """
    Vectorized helper that returns |C| for each certainty factor.
    """
    cs = np.asarray(list(certainties), dtype=float)
    return np.abs(cs)


def prediction_and_confidence_from_expval(
    expval: Union[float, int],
    zero_class_label: int = 0,
    one_class_label: int = 1,
    threshold: float = 0.0,
) -> tuple[int, float, float]:
    """
    Convenience helper returning:
    (predicted_label, certainty_factor, confidence)
    """
    c = certainty_from_expval(expval)
    pred = predicted_label_from_certainty(
        c,
        zero_class_label=zero_class_label,
        one_class_label=one_class_label,
        threshold=threshold,
    )
    conf = confidence_from_certainty(c)
    return pred, c, conf


def prediction_and_confidence_from_samples(
    samples: Iterable[Union[float, int]],
    zero_class_label: int = 0,
    one_class_label: int = 1,
    threshold: float = 0.0,
) -> tuple[int, float, float]:
    """
    Convenience helper returning:
    (predicted_label, certainty_factor, confidence)
    from sampled Pauli eigenvalues.
    """
    c = certainty_from_samples(samples)
    pred = predicted_label_from_certainty(
        c,
        zero_class_label=zero_class_label,
        one_class_label=one_class_label,
        threshold=threshold,
    )
    conf = confidence_from_certainty(c)
    return pred, c, conf