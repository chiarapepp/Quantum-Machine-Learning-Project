from __future__ import annotations

from typing import Any, Dict, Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from certainty_factor import (
    certainty_from_expval,
    predict_many_from_certainty,
    confidences_from_certainties,
)


def predict_outputs(
    qnode: Callable,
    X: np.ndarray,
    params: np.ndarray,
) -> np.ndarray:
    """
    Run the QNN on all samples and return raw scalar outputs.

    Assumption:
    - qnode(x, params) returns one scalar per sample, typically the expectation
      value of the measured Pauli observable.
    """
    X = np.asarray(X, dtype=float)

    outputs = [qnode(x, params) for x in X]
    return np.asarray(outputs, dtype=float).reshape(-1)


def evaluate_qnode(
    qnode: Callable,
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    threshold: float = 0.0,
    benign_label: int = 0,
    malicious_label: int = 1,
) -> Dict[str, Any]:
    """
    Evaluate a PennyLane QNode on a dataset.

    Assumptions
    -----------
    - qnode(x, params) returns one scalar per sample.
    - In noiseless mode this scalar is the expectation value of the measured
      Pauli observable, and is converted into the certainty factor C in [-1, 1].

    Decision rule
    -------------
    - C >= threshold -> benign_label
    - C <  threshold -> malicious_label

    Default convention
    ------------------
    - benign_label = 0
    - malicious_label = 1

    Metrics
    -------
    Precision, recall, and F1 are computed with malicious_label as the
    positive class.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    if len(X) == 0:
        raise ValueError("Empty dataset passed to evaluate_qnode().")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length.")

    raw_outputs = predict_outputs(qnode=qnode, X=X, params=params)

    Cs = np.array([certainty_from_expval(v) for v in raw_outputs], dtype=float)

    preds = predict_many_from_certainty(
        Cs,
        threshold=threshold,
        benign_label=benign_label,
        malicious_label=malicious_label,
    )
    confs = confidences_from_certainties(Cs)

    f1 = f1_score(y, preds, pos_label=malicious_label, zero_division=0)
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, pos_label=malicious_label, zero_division=0)
    rec = recall_score(y, preds, pos_label=malicious_label, zero_division=0)

    try:
        # Larger maliciousness must correspond to larger score for class 1.
        # Since malicious is predicted for smaller certainty values,
        # we flip certainty into a malicious-class score.
        malicious_score = (-Cs + 1.0) / 2.0
        roc = roc_auc_score(y, malicious_score)
    except Exception:
        roc = float("nan")

    return {
        "f1": float(f1),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "roc_auc": float(roc),
        "cert_mean": float(np.mean(Cs)),
        "cert_std": float(np.std(Cs)),
        "conf_mean": float(np.mean(confs)),
        "conf_std": float(np.std(confs)),
        "raw_outputs": raw_outputs,
        "Cs": Cs,
        "confidence": confs,
        "y": y,
        "preds": preds,
    }


def certainty_stats(
    certainties: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute certainty-factor statistics split by correct and incorrect predictions.

    Useful for understanding whether correct predictions tend to be farther from
    0, while uncertain predictions concentrate near 0.
    """
    Cs = np.asarray(certainties, dtype=float)
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)

    if not (len(Cs) == len(yt) == len(yp)):
        raise ValueError("certainties, y_true, and y_pred must have the same length.")

    correct_mask = yt == yp
    incorrect_mask = ~correct_mask
    confs = np.abs(Cs)

    return {
        "all_mean": float(np.mean(Cs)),
        "all_std": float(np.std(Cs)),
        "all_conf_mean": float(np.mean(confs)),
        "all_conf_std": float(np.std(confs)),
        "correct_mean": float(np.mean(Cs[correct_mask])) if correct_mask.any() else float("nan"),
        "correct_std": float(np.std(Cs[correct_mask])) if correct_mask.any() else float("nan"),
        "incorrect_mean": float(np.mean(Cs[incorrect_mask])) if incorrect_mask.any() else float("nan"),
        "incorrect_std": float(np.std(Cs[incorrect_mask])) if incorrect_mask.any() else float("nan"),
        "correct_conf_mean": float(np.mean(confs[correct_mask])) if correct_mask.any() else float("nan"),
        "correct_conf_std": float(np.std(confs[correct_mask])) if correct_mask.any() else float("nan"),
        "incorrect_conf_mean": float(np.mean(confs[incorrect_mask])) if incorrect_mask.any() else float("nan"),
        "incorrect_conf_std": float(np.std(confs[incorrect_mask])) if incorrect_mask.any() else float("nan"),
        "frac_low_conf_0.2": float(np.mean(confs < 0.2)),
        "frac_low_conf_0.5": float(np.mean(confs < 0.5)),
    }


def evaluate_with_stats(
    qnode: Callable,
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    threshold: float = 0.0,
    benign_label: int = 0,
    malicious_label: int = 1,
) -> Dict[str, Any]:
    """
    Convenience wrapper:
    - runs evaluation
    - adds certainty-factor statistics
    """
    out = evaluate_qnode(
        qnode=qnode,
        X=X,
        y=y,
        params=params,
        threshold=threshold,
        benign_label=benign_label,
        malicious_label=malicious_label,
    )

    out["certainty_stats"] = certainty_stats(
        certainties=out["Cs"],
        y_true=out["y"],
        y_pred=out["preds"],
    )
    return out