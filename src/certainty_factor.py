# src/certainty_factor.py
import numpy as np
import torch

def certainty_from_expval_z(expval_z: float) -> float:
    """
    For analytic runs (shots=None), qnode returns expectation value of PauliZ.
    This is exactly C = |alpha0|^2 - |alpha1|^2.
    """
    return float(expval_z)


def certainty_from_samples(samples) -> float:
    """
    samples: array of +1/-1 values from qml.sample(qml.PauliZ)
    Certainty = mean of samples = p0 - p1
    """
    s = np.array(samples).astype(float)
    return float(s.mean())


def predicted_label_from_certainty(C: float, true_label_expected_zero=True) -> int:
    """
    Convert certainty factor into predicted class.
    true_label_expected_zero=True means:
      class 0 = malicious (True), class 1 = benign (False).
    If false, flip roles.
    """

    if true_label_expected_zero:
        # Positive C => class 0, negative C => class 1
        return 0 if C >= 0 else 1
    else:
        # Reverse interpretation
        return 1 if C >= 0 else 0
