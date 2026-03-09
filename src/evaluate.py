# src/evaluate.py
"""
Evaluation utilities for the QNN intrusion detection system.

Separated from train.py so evaluation logic can be reused by:
  - train.py  (val loop)
  - noise_eval.py  (noisy inference)
  - run_experiment.py  (grid-search summary)

Key design choices that match the paper:
  - Certainty factor C = expectation value of the measured observable.
  - For Simple (X-basis): C = <X>  in [-1, +1]
  - For TTN/MERA/QCNN (Z-basis): C = <Z>  in [-1, +1]
  - Prediction: C >= 0  ->  benign (label 0);  C < 0  ->  malicious (label 1).
    (malicious maps to the negative side; see paper Section III-C and Listing 1.)
  - Metrics: F1 (primary), accuracy, precision, recall, ROC-AUC.
"""

from __future__ import annotations
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, roc_auc_score,
)
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    device: str = "cpu",
    desc: str = "eval",
) -> Dict[str, Any]:
    """
    Run inference on (X, y) and return a metrics dict.

    model.forward(x_batch) must return a 1-D tensor of certainty factors C in [-1, 1].

    Prediction rule (paper-consistent):
        C >= 0  ->  0 (benign)
        C <  0  ->  1 (malicious)

    Returns
    -------
    dict with keys:
        f1, accuracy, precision, recall, roc_auc,
        cert_mean, cert_std,
        Cs  (raw certainty factor array),
        y   (ground truth array),
        preds (predicted label array)
    """
    model.eval()
    ds     = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_C: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    with torch.no_grad():
        for xb, yb in tqdm(loader, desc=desc, leave=False):
            xb  = xb.to(device)
            out = model(xb).cpu().numpy()
            all_C.append(out)
            all_y.append(yb.numpy())

    Cs    = np.concatenate(all_C)
    ys    = np.concatenate(all_y)
    preds = (Cs < 0).astype(int)    # C < 0 -> malicious (label 1)

    f1   = f1_score(ys, preds, zero_division=0)
    acc  = accuracy_score(ys, preds)
    prec = precision_score(ys, preds, zero_division=0)
    rec  = recall_score(ys, preds, zero_division=0)
    try:
        # Map C -> probability-like score: high C (benign) -> low prob of class 1
        roc = roc_auc_score(ys, (-Cs + 1) / 2.0)
    except Exception:
        roc = float("nan")

    return {
        "f1":        float(f1),
        "accuracy":  float(acc),
        "precision": float(prec),
        "recall":    float(rec),
        "roc_auc":   float(roc),
        "cert_mean": float(np.mean(Cs)),
        "cert_std":  float(np.std(Cs)),
        "Cs":        Cs,
        "y":         ys,
        "preds":     preds,
    }


# ---------------------------------------------------------------------------
# Certainty-factor analysis (paper Section III-C, Fig. 7 & 8)
# ---------------------------------------------------------------------------

def certainty_factor_from_expval(expval: float) -> float:
    """
    C = expectation value of the measured Pauli observable.

    For a sample with true label 0 (benign -> |0> expected):
        C = |alpha_0|^2 - |alpha_1|^2 = <Z>  (or <X> for Simple).
    C = +1: maximally confident correct prediction.
    C = -1: maximally confident wrong prediction.
    C =  0: maximally uncertain.
    """
    c = float(expval)
    return float(np.clip(c, -1.0, 1.0))


def certainty_stats(Cs: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Compute certainty-factor statistics split by correct / incorrect predictions.

    Useful for reproducing Fig. 7 (violin plots) and Fig. 8 (histogram) from the paper.
    """
    correct   = y_true == y_pred
    incorrect = ~correct

    return {
        "all_mean":       float(np.mean(Cs)),
        "all_std":        float(np.std(Cs)),
        "correct_mean":   float(np.mean(Cs[correct]))   if correct.any()   else float("nan"),
        "correct_std":    float(np.std(Cs[correct]))    if correct.any()   else float("nan"),
        "incorrect_mean": float(np.mean(Cs[incorrect])) if incorrect.any() else float("nan"),
        "incorrect_std":  float(np.std(Cs[incorrect]))  if incorrect.any() else float("nan"),
        # fraction of low-confidence predictions (susceptible to noise flips)
        "frac_low_conf_0.2": float(np.mean(np.abs(Cs) < 0.2)),
        "frac_low_conf_0.5": float(np.mean(np.abs(Cs) < 0.5)),
    }