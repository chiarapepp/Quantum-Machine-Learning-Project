# src/certainty_factor.py

from typing import Iterable, Literal, Optional, Union

import numpy as np


MeasurementBasis = Literal["Z", "X"]
ArchitectureName = Literal["ttn", "mera", "qcnn", "simple"]


def measurement_basis_for_architecture(architecture: str) -> MeasurementBasis:
    """
    Return the correct readout basis for each architecture.

    Paper-consistent convention:
    - TTN, MERA, QCNN -> Z basis
    - Simple          -> X basis
    """
    arch = architecture.strip().lower()

    if arch in {"ttn", "mera", "qcnn"}:
        return "Z"
    if arch == "simple":
        return "X"

    raise ValueError(
        f"Unknown architecture '{architecture}'. "
        "Supported values: 'ttn', 'mera', 'qcnn', 'simple'."
    )


def certainty_from_expval(expval: Union[float, int]) -> float:
    """
    Convert a measured expectation value into the certainty factor C.

    For the architectures in this project, the certainty factor is simply the
    expectation value of the final measured observable:
    - TTN / MERA / QCNN: C = <Z>
    - Simple:            C = <X>

    The returned value should lie in [-1, 1], up to small numerical error.
    """
    c = float(expval)

    # Optional small numerical stabilization
    if c > 1.0 and c < 1.0 + 1e-8:
        c = 1.0
    elif c < -1.0 and c > -1.0 - 1e-8:
        c = -1.0

    return c


def certainty_from_expval_with_architecture(
    expval: Union[float, int],
    architecture: str,
) -> float:
    """
    Same as certainty_from_expval(), but validates that the architecture is known.

    This is useful when you want the call site to make the architectural choice
    explicit, even though the computation is just C = expectation value.
    """
    _ = measurement_basis_for_architecture(architecture)
    return certainty_from_expval(expval)


def certainty_from_samples(samples: Iterable[Union[float, int]]) -> float:
    """
    Convert sampled eigenvalues of the final observable into the certainty factor.

    Expected samples:
    - from qml.sample(qml.PauliZ(...)) for TTN / MERA / QCNN
    - from qml.sample(qml.PauliX(...)) for Simple

    In both cases, PennyLane returns eigenvalues in {+1, -1}, and the certainty
    factor is their mean:
        C = mean(samples)

    This estimates the corresponding expectation value.
    """
    s = np.asarray(list(samples), dtype=float)

    if s.size == 0:
        raise ValueError("certainty_from_samples received an empty sample array.")

    unique_vals = np.unique(s)
    allowed = {-1.0, 1.0}

    if not set(unique_vals).issubset(allowed):
        raise ValueError(
            "Samples must be eigenvalues of a Pauli observable, i.e. only ±1."
        )

    return float(np.mean(s))


def certainty_from_samples_with_architecture(
    samples: Iterable[Union[float, int]],
    architecture: str,
) -> float:
    """
    Same as certainty_from_samples(), but validates that the architecture is known.

    The architecture determines which observable should have been sampled:
    - TTN / MERA / QCNN -> PauliZ
    - Simple            -> PauliX
    """
    _ = measurement_basis_for_architecture(architecture)
    return certainty_from_samples(samples)


def predicted_label_from_certainty(
    C: float,
    zero_class_label: int = 0,
    one_class_label: int = 1,
    threshold: float = 0.0,
) -> int:
    """
    Convert certainty factor into a predicted class label.

    Interpretation:
    - C >= threshold -> prediction is aligned with the '0-like' outcome
    - C <  threshold -> prediction is aligned with the '1-like' outcome

    By default:
    - outcome |0> (or positive-side outcome in the chosen basis) -> label 0
    - outcome |1> (or negative-side outcome in the chosen basis) -> label 1

    Parameters
    ----------
    C:
        Certainty factor in [-1, 1].
    zero_class_label:
        Dataset label associated with the positive side of the readout.
    one_class_label:
        Dataset label associated with the negative side of the readout.
    threshold:
        Decision threshold, default 0.0 as in the paper's sign-based decision rule.
    """
    return zero_class_label if float(C) >= float(threshold) else one_class_label


def confidence_from_certainty(C: float) -> float:
    """
    Convert certainty factor into a confidence magnitude.

    Returns |C| in [0, 1]:
    - 0   -> maximally uncertain / flat prediction
    - 1   -> maximally sharp / confident prediction
    """
    return abs(float(C))


def prediction_and_confidence_from_expval(
    expval: Union[float, int],
    zero_class_label: int = 0,
    one_class_label: int = 1,
    threshold: float = 0.0,
) -> tuple[int, float, float]:
    """
    Convenience helper returning:
    (predicted_label, certainty_factor, confidence)

    This is architecture-agnostic once the correct observable expectation value
    has already been computed by the QNode.
    """
    C = certainty_from_expval(expval)
    pred = predicted_label_from_certainty(
        C,
        zero_class_label=zero_class_label,
        one_class_label=one_class_label,
        threshold=threshold,
    )
    conf = confidence_from_certainty(C)
    return pred, C, conf


def prediction_and_confidence_from_samples(
    samples: Iterable[Union[float, int]],
    zero_class_label: int = 0,
    one_class_label: int = 1,
    threshold: float = 0.0,
) -> tuple[int, float, float]:
    """
    Convenience helper returning:
    (predicted_label, certainty_factor, confidence)

    Works from sampled ±1 eigenvalues of the final measured observable.
    """
    C = certainty_from_samples(samples)
    pred = predicted_label_from_certainty(
        C,
        zero_class_label=zero_class_label,
        one_class_label=one_class_label,
        threshold=threshold,
    )
    conf = confidence_from_certainty(C)
    return pred, C, conf