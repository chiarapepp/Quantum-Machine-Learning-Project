from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from certainty_factor import (
    certainty_from_expval,
    predict_many_from_certainty,
    confidences_from_certainties,
)


def evaluate_model(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    device: str = "cpu",
    desc: str = "eval",
    threshold: float = 0.0,
    positive_class_label: int = 0,
    negative_class_label: int = 1,
) -> Dict[str, Any]:
    """
    Run inference on a dataset and return standard classification metrics.

    Assumptions
    -----------
    model.forward(x_batch) returns one scalar per sample representing the
    final expectation value of the measured observable, which is also the
    certainty factor C in [-1, 1].

    Default decision rule
    ---------------------
    - C >= threshold -> positive_class_label
    - C <  threshold -> negative_class_label

    With the current project convention:
    - positive_class_label = 0  -> benign
    - negative_class_label = 1  -> malicious

    Returns
    -------
    dict with:
        f1, accuracy, precision, recall, roc_auc,
        cert_mean, cert_std,
        conf_mean, conf_std,
        Cs, confidence, y, preds
    """
    model.eval()

    X = np.asarray(X)
    y = np.asarray(y, dtype=int)

    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_c: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    with torch.no_grad():
        for xb, yb in tqdm(loader, desc=desc, leave=False):
            xb = xb.to(device)
            out = model(xb)

            if isinstance(out, torch.Tensor):
                out_np = out.detach().cpu().numpy()
            else:
                out_np = np.asarray(out, dtype=float)

            all_c.append(np.asarray(out_np, dtype=float).reshape(-1))
            all_y.append(yb.detach().cpu().numpy().reshape(-1))

    if not all_c:
        raise ValueError("Empty dataset passed to evaluate_model().")

    Cs_raw = np.concatenate(all_c, axis=0)
    ys = np.concatenate(all_y, axis=0)

    Cs = np.array([certainty_from_expval(v) for v in Cs_raw], dtype=float)

    preds = predict_many_from_certainty(
        Cs,
        positive_class_label=positive_class_label,
        negative_class_label=negative_class_label,
        threshold=threshold,
    )
    confs = confidences_from_certainties(Cs)

    f1 = f1_score(ys, preds, zero_division=0)
    acc = accuracy_score(ys, preds)
    prec = precision_score(ys, preds, zero_division=0)
    rec = recall_score(ys, preds, zero_division=0)

    try:
        # Build a score for class 1 (malicious) so roc_auc_score is consistent.
        #
        # Under the default rule:
        #   C >= 0 -> class 0 (benign)
        #   C <  0 -> class 1 (malicious)
        #
        # So larger maliciousness should correspond to smaller C.
        malicious_score = (-Cs + 1.0) / 2.0
        roc = roc_auc_score(ys, malicious_score)
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
        "Cs": Cs,
        "confidence": confs,
        "y": ys,
        "preds": preds,
    }


def certainty_stats(
    certainties: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute certainty-factor statistics split by correct and incorrect predictions.

    Useful for analysing whether correct predictions tend to lie farther from 0,
    while uncertain predictions concentrate near 0.
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