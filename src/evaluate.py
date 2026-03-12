from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def predict_outputs(qnode: Callable, X: np.ndarray, params: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return np.asarray([qnode(x, params) for x in X], dtype=float).reshape(-1)


def certainty_from_output(output: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(output, dtype=float), -1.0, 1.0)


def confidence_from_certainty(certainty: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(certainty, dtype=float))


def predict_labels(raw_outputs: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    # coherent with training:
    # output >= 0 -> +1 -> malicious -> 1
    # output <  0 -> -1 -> benign    -> 0
    raw_outputs = np.asarray(raw_outputs, dtype=float)
    return (raw_outputs >= threshold).astype(int)


def evaluate_qnode(
    qnode: Callable,
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    threshold: float = 0.0,
) -> Dict[str, Any]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    if len(X) == 0:
        raise ValueError("Empty dataset passed to evaluate_qnode().")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length.")

    raw_outputs = predict_outputs(qnode, X, params)
    certainties = certainty_from_output(raw_outputs)
    confidences = confidence_from_certainty(certainties)
    preds = predict_labels(raw_outputs, threshold=threshold)

    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, pos_label=1, zero_division=0)
    rec = recall_score(y, preds, pos_label=1, zero_division=0)
    f1 = f1_score(y, preds, pos_label=1, zero_division=0)

    try:
        # high score = more malicious
        malicious_score = (certainties + 1.0) / 2.0
        roc = roc_auc_score(y, malicious_score)
    except Exception:
        roc = float("nan")

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc),
        "raw_outputs": raw_outputs,
        "certainties": certainties,
        "confidences": confidences,
        "y_true": y,
        "preds": preds,
        "cert_mean": float(np.mean(certainties)),
        "cert_std": float(np.std(certainties)),
        "conf_mean": float(np.mean(confidences)),
        "conf_std": float(np.std(confidences)),
    }


def certainty_stats(certainties: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    certainties = np.asarray(certainties, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if not (len(certainties) == len(y_true) == len(y_pred)):
        raise ValueError("certainties, y_true, and y_pred must have the same length.")

    confs = np.abs(certainties)
    correct = y_true == y_pred
    incorrect = ~correct

    return {
        "all_mean": float(np.mean(certainties)),
        "all_std": float(np.std(certainties)),
        "all_conf_mean": float(np.mean(confs)),
        "all_conf_std": float(np.std(confs)),
        "correct_mean": float(np.mean(certainties[correct])) if correct.any() else float("nan"),
        "incorrect_mean": float(np.mean(certainties[incorrect])) if incorrect.any() else float("nan"),
        "correct_conf_mean": float(np.mean(confs[correct])) if correct.any() else float("nan"),
        "incorrect_conf_mean": float(np.mean(confs[incorrect])) if incorrect.any() else float("nan"),
        "frac_low_conf_0.2": float(np.mean(confs < 0.2)),
        "frac_low_conf_0.5": float(np.mean(confs < 0.5)),
    }


def evaluate_with_stats(
    qnode: Callable,
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    threshold: float = 0.0,
) -> Dict[str, Any]:
    out = evaluate_qnode(qnode=qnode, X=X, y=y, params=params, threshold=threshold)
    out["certainty_stats"] = certainty_stats(out["certainties"], out["y_true"], out["preds"])
    return out


def evaluate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    device: str = "cpu",
    desc: str = "",
    threshold: float = 0.0,
) -> Dict[str, Any]:
    """
    Evaluate a PyTorch model (e.g. NoisySimpleQNNModel) on a dataset.

    The model's forward() must accept a float32 tensor of shape (batch, features)
    and return a 1-D tensor of certainty-like scalars in [-1, 1].
    """
    import torch

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    if len(X) == 0:
        raise ValueError("Empty dataset passed to evaluate_model().")

    all_outputs: list = []
    for start in range(0, len(X), batch_size):
        xb = torch.tensor(X[start : start + batch_size], dtype=torch.float32).to(device)
        with torch.no_grad():
            out = model(xb)
        all_outputs.append(out.detach().cpu().numpy().reshape(-1))

    raw_outputs = np.concatenate(all_outputs)
    certainties = certainty_from_output(raw_outputs)
    confidences = confidence_from_certainty(certainties)
    preds = predict_labels(raw_outputs, threshold=threshold)

    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, pos_label=1, zero_division=0)
    rec = recall_score(y, preds, pos_label=1, zero_division=0)
    f1 = f1_score(y, preds, pos_label=1, zero_division=0)

    try:
        malicious_score = (certainties + 1.0) / 2.0
        roc = roc_auc_score(y, malicious_score)
    except Exception:
        roc = float("nan")

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc),
        "raw_outputs": raw_outputs,
        "certainties": certainties,
        "confidences": confidences,
        "y_true": y,
        "preds": preds,
        "cert_mean": float(np.mean(certainties)),
        "cert_std": float(np.std(certainties)),
        "conf_mean": float(np.mean(confidences)),
        "conf_std": float(np.std(confidences)),
    }