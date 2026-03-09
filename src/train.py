# src/train.py
"""
Training loop for QNN intrusion detection.

Paper-exact choices:
  - Loss: Hinge loss with ±1 targets  (malicious -> +1, benign -> -1)
    matching TF-Quantum tutorial / paper Listing 1.
  - Best hyperparams (paper Table 2 / Table 4):
        arch=Simple, 6 layers, XY (XXYY), SGD, lr=0.02, decay=0.001, batch=32.
  - "decay" in the paper means learning-rate exponential decay, not L2 weight decay.
    We implement it as torch.optim.lr_scheduler.ExponentialLR(gamma = 1 - decay).
  - Evaluation uses evaluate.py which applies the paper-correct prediction threshold.
"""

from __future__ import annotations
import os
import json
import time
import argparse
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False

import src.model as model_module
import src.evaluate as eval_module
import src.data_utils as data_utils


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def hinge_loss(C: torch.Tensor, y_binary: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """
    Hinge loss with ±1 targets.

    Paper convention (Listing 1, citing TF-Quantum MNIST tutorial):
        malicious (label 1) -> target +1
        benign    (label 0) -> target -1

    loss = mean( max(0, margin - C * target) )
    """
    target = torch.where(
        y_binary == 1,
        torch.ones_like(C),
        -torch.ones_like(C),
    )
    return torch.clamp(margin - C * target, min=0.0).mean()


# ---------------------------------------------------------------------------
# One training trial
# ---------------------------------------------------------------------------

def run_trial(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    cfg:     Dict[str, Any],
) -> Dict[str, Any]:
    """
    Train one QNN configuration and return best validation metrics + checkpoint path.

    cfg keys
    --------
    arch          : str   - "simple" | "ttn" | "mera" | "qcnn"
    n_feature_qubits : int
    n_layers      : int
    layer_type    : str   - e.g. "XXYY"
    batch_size    : int
    lr            : float
    optimizer     : str   - "sgd" | "adam"
    momentum      : float - only for SGD
    lr_decay      : float - exponential LR decay per epoch  (paper: 0.001)
    hinge_margin  : float - default 1.0
    epochs        : int
    output_dir    : str
    run_name      : str
    use_wandb     : bool
    """
    run_name = cfg.get("run_name", "trial")

    if _WANDB and cfg.get("use_wandb", False):
        wandb.init(
            project=cfg.get("wandb_project", "naduqnn"),
            config=cfg,
            name=run_name,
        )

    device_str = "cuda" if (torch.cuda.is_available() and cfg.get("use_cuda", False)) else "cpu"

    # Build device + model
    dev = model_module.make_device(
        n_feature_qubits=cfg["n_feature_qubits"],
        arch=cfg["arch"],
        shots=None,
        noisy=False,
    )
    qnn = model_module.QNNModel(
        arch=cfg["arch"],
        n_feature_qubits=cfg["n_feature_qubits"],
        n_layers=cfg["n_layers"],
        dev=dev,
        layer_type=cfg.get("layer_type", "XXYY"),
    ).to(device_str)

    # Optimizer
    opt_name = cfg.get("optimizer", "sgd").lower()
    if opt_name == "sgd":
        optimizer = optim.SGD(
            qnn.parameters(),
            lr=cfg["lr"],
            momentum=cfg.get("momentum", 0.0),
            # NOTE: weight_decay here is L2 reg, NOT the paper's LR decay.
            # Paper's "decay" is applied via ExponentialLR below.
        )
    elif opt_name == "adam":
        optimizer = optim.Adam(qnn.parameters(), lr=cfg["lr"])
    else:
        raise ValueError(f"Unknown optimizer '{opt_name}'.")

    # LR scheduler: ExponentialLR with gamma = 1 - lr_decay  (paper: decay=0.001)
    lr_decay = cfg.get("lr_decay", 0.0)
    scheduler = ExponentialLR(optimizer, gamma=max(0.0, 1.0 - lr_decay))

    # Data loaders
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)

    margin = cfg.get("hinge_margin", 1.0)
    epochs = cfg["epochs"]

    best_val_f1  = -1.0
    best_ckpt    = None
    best_epoch   = -1

    for epoch in range(epochs):
        qnn.train()
        epoch_losses: list[float] = []
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"[{run_name}] epoch {epoch+1}/{epochs}", leave=False)
        for xb, yb in pbar:
            xb = xb.to(device_str)
            yb = yb.to(device_str)
            optimizer.zero_grad()
            out  = qnn(xb)
            loss = hinge_loss(out, yb, margin=margin)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
            pbar.set_postfix(loss=f"{np.mean(epoch_losses):.4f}")

        scheduler.step()

        # Validation
        val_stats = eval_module.evaluate_model(
            qnn, X_val, y_val,
            batch_size=cfg["batch_size"],
            device=device_str,
            desc="val",
        )

        elapsed = time.time() - t0
        print(
            f"[{run_name}] epoch {epoch+1}/{epochs}  "
            f"loss={np.mean(epoch_losses):.4f}  "
            f"val_f1={val_stats['f1']:.4f}  "
            f"val_acc={val_stats['accuracy']:.4f}  "
            f"time={elapsed:.1f}s"
        )

        if _WANDB and cfg.get("use_wandb", False):
            wandb.log({
                "epoch":        epoch,
                "train/loss":   float(np.mean(epoch_losses)),
                "val/f1":       val_stats["f1"],
                "val/accuracy": val_stats["accuracy"],
                "val/precision":val_stats["precision"],
                "val/recall":   val_stats["recall"],
                "val/roc_auc":  val_stats["roc_auc"],
                "lr":           scheduler.get_last_lr()[0],
            })

        if val_stats["f1"] > best_val_f1:
            best_val_f1  = val_stats["f1"]
            best_epoch   = epoch
            os.makedirs(cfg["output_dir"], exist_ok=True)
            best_ckpt = os.path.join(
                cfg["output_dir"],
                f"{run_name}_best.pt",
            )
            torch.save(
                {
                    "model_state": qnn.state_dict(),
                    "config":      cfg,
                    "epoch":       epoch,
                    "val_f1":      best_val_f1,
                },
                best_ckpt,
            )

    if _WANDB and cfg.get("use_wandb", False):
        wandb.log({"best_val_f1": best_val_f1})
        wandb.finish()

    print(f"[{run_name}] Done. Best val F1={best_val_f1:.4f} at epoch {best_epoch+1}.")
    return {"best_val_f1": best_val_f1, "best_ckpt": best_ckpt, "best_epoch": best_epoch}


# ---------------------------------------------------------------------------
# Defaults / CLI
# ---------------------------------------------------------------------------

DEFAULT_CFG: Dict[str, Any] = {
    # data
    "data_csv":          "data/processed/nf_unsw_balanced.csv",
    "raw_csv":           "data/raw/NF-UNSW-NB15-v2.csv",
    # model (paper best)
    "arch":              "simple",
    "n_feature_qubits":  8,
    "n_layers":          6,
    "layer_type":        "XXYY",   # paper "XY"
    # optimiser (paper best: SGD, lr=0.02, decay=0.001, batch=32)
    "optimizer":         "sgd",
    "lr":                0.02,
    "momentum":          0.0,
    "lr_decay":          0.001,    # ExponentialLR gamma = 1 - 0.001
    "batch_size":        32,
    "hinge_margin":      1.0,
    "epochs":            30,
    # misc
    "output_dir":        "results/checkpoints",
    "use_cuda":          False,
    "use_wandb":         False,
    "wandb_project":     "naduqnn_repro",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch",       default=DEFAULT_CFG["arch"])
    parser.add_argument("--n_layers",   type=int,   default=DEFAULT_CFG["n_layers"])
    parser.add_argument("--layer_type", default=DEFAULT_CFG["layer_type"])
    parser.add_argument("--optimizer",  default=DEFAULT_CFG["optimizer"])
    parser.add_argument("--lr",         type=float, default=DEFAULT_CFG["lr"])
    parser.add_argument("--momentum",   type=float, default=DEFAULT_CFG["momentum"])
    parser.add_argument("--lr_decay",   type=float, default=DEFAULT_CFG["lr_decay"])
    parser.add_argument("--batch_size", type=int,   default=DEFAULT_CFG["batch_size"])
    parser.add_argument("--epochs",     type=int,   default=DEFAULT_CFG["epochs"])
    parser.add_argument("--data_csv",   default=DEFAULT_CFG["data_csv"])
    parser.add_argument("--raw_csv",    default=DEFAULT_CFG["raw_csv"])
    parser.add_argument("--output_dir", default=DEFAULT_CFG["output_dir"])
    parser.add_argument("--use_wandb",  action="store_true")
    parser.add_argument("--run_name",   default="paper_best")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    cfg = DEFAULT_CFG.copy()
    cfg.update(vars(args))
    cfg["data_csv"]   = os.path.join(root, cfg["data_csv"])
    cfg["raw_csv"]    = os.path.join(root, cfg["raw_csv"])
    cfg["output_dir"] = os.path.join(root, cfg["output_dir"])

    # Prepare / load data
    if not os.path.exists(cfg["data_csv"]):
        print("[train] Generating processed CSV ...")
        import dataset as ds_module
        os.makedirs(os.path.dirname(cfg["data_csv"]), exist_ok=True)
        ds_module.load_and_prepare_nf_unsw(cfg["raw_csv"], save_processed_csv=cfg["data_csv"])

    pack = data_utils.load_encoded_splits(cfg["data_csv"])
    X_train, y_train = pack["X_train"], pack["y_train"]
    X_test,  y_test  = pack["X_test"],  pack["y_test"]
    print(f"[train] train={X_train.shape}  test={X_test.shape}")

    result = run_trial(X_train, y_train, X_test, y_test, cfg)
    print("[train] Result:", result)