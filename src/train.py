"""
Training loop for QNN intrusion detection.

This module:
- ensures the balanced processed CSV exists
- loads encoded train/test splits from data_utils.py
- trains one QNN model with hinge loss
- evaluates after each epoch
- saves the best checkpoint by validation/test F1

Paper-aligned defaults
----------------------
- best reported model: Simple architecture
- 8 feature qubits
- 6 layers
- XY interaction pattern (implemented here as "XXYY")
- SGD, lr=0.02, decay=0.001, batch_size=32
"""

from __future__ import annotations

import os
import json
import time
import argparse
from typing import Any, Dict

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False

import model as model_module
import evaluate as eval_module
import data_utils
import dataset as ds_module


def hinge_loss(
    certainty: torch.Tensor,
    y_binary: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """
    Hinge loss with ±1 targets.

    Label convention:
    - malicious (1) -> target +1
    - benign    (0) -> target -1

    loss = mean(max(0, margin - certainty * target))
    """
    target = torch.where(
        y_binary == 1,
        torch.ones_like(certainty),
        -torch.ones_like(certainty),
    )
    return torch.clamp(margin - certainty * target, min=0.0).mean()


def _build_optimizer(
    model: torch.nn.Module,
    cfg: Dict[str, Any],
) -> torch.optim.Optimizer:
    """
    Build optimizer from config.
    """
    opt_name = str(cfg.get("optimizer", "sgd")).lower()

    if opt_name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=float(cfg["lr"]),
            momentum=float(cfg.get("momentum", 0.0)),
        )

    if opt_name == "adam":
        return optim.Adam(
            model.parameters(),
            lr=float(cfg["lr"]),
        )

    raise ValueError(f"Unknown optimizer '{opt_name}'.")


def _save_checkpoint(
    path: str,
    model: torch.nn.Module,
    cfg: Dict[str, Any],
    epoch: int,
    metrics: Dict[str, Any],
) -> None:
    """
    Save checkpoint in a format that noise_eval.py can later reload.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    payload = {
        "epoch": int(epoch),
        "config": dict(cfg),
        "model_state": model.state_dict(),
        "metrics": {
            k: float(v) for k, v in metrics.items()
            if isinstance(v, (int, float, np.floating))
        },
    }
    torch.save(payload, path)


def run_trial(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Train one QNN configuration and return best validation metrics.

    Parameters
    ----------
    X_train, y_train:
        Encoded training split.
    X_val, y_val:
        Encoded validation/test split used for model selection.
        For now this is the 15% split returned by data_utils.py.
    cfg:
        Training configuration dictionary.

    Returns
    -------
    dict
        Summary with best epoch, checkpoint path, and best metrics.
    """
    run_name = str(cfg.get("run_name", "trial"))
    use_cuda = bool(cfg.get("use_cuda", False))
    device_str = "cuda" if (torch.cuda.is_available() and use_cuda) else "cpu"

    if _WANDB and bool(cfg.get("use_wandb", False)):
        wandb.init(
            project=cfg.get("wandb_project", "naduqnn_repro"),
            name=run_name,
            config=cfg,
        )

    dev = model_module.make_device(
        n_feature_qubits=int(cfg["n_feature_qubits"]),
        arch=str(cfg["arch"]),
        shots=None,
        noisy=False,
    )

    qnn = model_module.QNNModel(
        arch=str(cfg["arch"]),
        n_feature_qubits=int(cfg["n_feature_qubits"]),
        n_layers=int(cfg["n_layers"]),
        dev=dev,
        layer_type=str(cfg.get("layer_type", "XXYY")),
    ).to(device_str)

    optimizer = _build_optimizer(qnn, cfg)

    lr_decay = float(cfg.get("lr_decay", 0.0))
    gamma = max(0.0, 1.0 - lr_decay)
    scheduler = ExponentialLR(optimizer, gamma=gamma)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
    )

    epochs = int(cfg["epochs"])
    margin = float(cfg.get("hinge_margin", 1.0))
    output_dir = str(cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    best_val_f1 = -1.0
    best_epoch = -1
    best_ckpt = None
    best_metrics: Dict[str, Any] = {}

    history: list[Dict[str, float]] = []

    for epoch in range(epochs):
        qnn.train()
        epoch_losses: list[float] = []
        t0 = time.time()

        progress = tqdm(
            train_loader,
            desc=f"{run_name} | epoch {epoch + 1}/{epochs}",
            leave=False,
        )

        for xb, yb in progress:
            xb = xb.to(device_str)
            yb = yb.to(device_str)

            optimizer.zero_grad()
            certainty = qnn(xb)
            loss = hinge_loss(certainty, yb, margin=margin)
            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().cpu().item())
            epoch_losses.append(loss_value)
            progress.set_postfix(loss=f"{loss_value:.4f}")

        scheduler.step()

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

        val_stats = eval_module.evaluate_model(
            qnn,
            X_val,
            y_val,
            batch_size=int(cfg["batch_size"]),
            device=device_str,
            desc=f"{run_name}-eval",
        )

        epoch_summary = {
            "epoch": float(epoch + 1),
            "train_loss": train_loss,
            "val_f1": float(val_stats["f1"]),
            "val_accuracy": float(val_stats["accuracy"]),
            "val_precision": float(val_stats["precision"]),
            "val_recall": float(val_stats["recall"]),
            "val_roc_auc": float(val_stats["roc_auc"]),
            "cert_mean": float(val_stats["cert_mean"]),
            "cert_std": float(val_stats["cert_std"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_time_sec": float(time.time() - t0),
        }
        history.append(epoch_summary)

        print(
            f"[{run_name}] epoch {epoch + 1:03d}/{epochs:03d} | "
            f"loss={train_loss:.4f} | "
            f"f1={val_stats['f1']:.4f} | "
            f"acc={val_stats['accuracy']:.4f} | "
            f"prec={val_stats['precision']:.4f} | "
            f"rec={val_stats['recall']:.4f} | "
            f"auc={val_stats['roc_auc']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        if _WANDB and bool(cfg.get("use_wandb", False)):
            wandb.log(epoch_summary)

        if float(val_stats["f1"]) > best_val_f1:
            best_val_f1 = float(val_stats["f1"])
            best_epoch = epoch
            best_metrics = val_stats

            ckpt_path = os.path.join(output_dir, f"{run_name}.pt")
            _save_checkpoint(
                ckpt_path,
                model=qnn,
                cfg=cfg,
                epoch=epoch + 1,
                metrics=val_stats,
            )
            best_ckpt = ckpt_path

    history_path = os.path.join(output_dir, f"{run_name}_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    result = {
        "best_val_f1": float(best_val_f1),
        "best_epoch": int(best_epoch + 1),
        "best_ckpt": best_ckpt,
        "history_json": history_path,
        "best_accuracy": float(best_metrics.get("accuracy", float("nan"))),
        "best_precision": float(best_metrics.get("precision", float("nan"))),
        "best_recall": float(best_metrics.get("recall", float("nan"))),
        "best_roc_auc": float(best_metrics.get("roc_auc", float("nan"))),
        "best_cert_mean": float(best_metrics.get("cert_mean", float("nan"))),
        "best_cert_std": float(best_metrics.get("cert_std", float("nan"))),
    }

    if _WANDB and bool(cfg.get("use_wandb", False)):
        wandb.log({"best_val_f1": best_val_f1, "best_epoch": best_epoch + 1})
        wandb.finish()

    print(
        f"[{run_name}] done | "
        f"best_f1={result['best_val_f1']:.4f} | "
        f"best_epoch={result['best_epoch']} | "
        f"ckpt={result['best_ckpt']}"
    )
    return result


DEFAULT_CFG: Dict[str, Any] = {
    # data
    "data_csv": "data/processed/nf_unsw_balanced.csv",
    "raw_csv": "data/raw/NF-UNSW-NB15-v2.csv",
    "n_bins": 100,
    "test_size": 0.15,
    "split_random_state": 1,

    # model
    "arch": "simple",
    "n_feature_qubits": 8,
    "n_layers": 6,
    "layer_type": "XXYY",

    # optimization
    "optimizer": "sgd",
    "lr": 0.02,
    "momentum": 0.0,
    "lr_decay": 0.001,
    "batch_size": 32,
    "hinge_margin": 1.0,
    "epochs": 30,

    # misc
    "output_dir": "results/checkpoints",
    "use_cuda": False,
    "use_wandb": False,
    "wandb_project": "naduqnn_repro",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--arch", default=DEFAULT_CFG["arch"])
    parser.add_argument("--n_feature_qubits", type=int, default=DEFAULT_CFG["n_feature_qubits"])
    parser.add_argument("--n_layers", type=int, default=DEFAULT_CFG["n_layers"])
    parser.add_argument("--layer_type", default=DEFAULT_CFG["layer_type"])

    parser.add_argument("--optimizer", default=DEFAULT_CFG["optimizer"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CFG["lr"])
    parser.add_argument("--momentum", type=float, default=DEFAULT_CFG["momentum"])
    parser.add_argument("--lr_decay", type=float, default=DEFAULT_CFG["lr_decay"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CFG["batch_size"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CFG["epochs"])
    parser.add_argument("--hinge_margin", type=float, default=DEFAULT_CFG["hinge_margin"])

    parser.add_argument("--n_bins", type=int, default=DEFAULT_CFG["n_bins"])
    parser.add_argument("--test_size", type=float, default=DEFAULT_CFG["test_size"])
    parser.add_argument("--split_random_state", type=int, default=DEFAULT_CFG["split_random_state"])

    parser.add_argument("--data_csv", default=DEFAULT_CFG["data_csv"])
    parser.add_argument("--raw_csv", default=DEFAULT_CFG["raw_csv"])
    parser.add_argument("--output_dir", default=DEFAULT_CFG["output_dir"])
    parser.add_argument("--run_name", default="paper_best")

    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")

    args = parser.parse_args()

    cfg = DEFAULT_CFG.copy()
    cfg.update(vars(args))

    if not os.path.exists(cfg["data_csv"]):
        print("[train] processed CSV not found, generating it from raw CSV...")
        os.makedirs(os.path.dirname(cfg["data_csv"]) or ".", exist_ok=True)
        ds_module.build_processed_nf_unsw(
            csv_path=cfg["raw_csv"],
            save_processed_csv=cfg["data_csv"],
        )

    pack = data_utils.load_encoded_splits(
        processed_csv=cfg["data_csv"],
        test_size=float(cfg["test_size"]),
        random_state=int(cfg["split_random_state"]),
        n_bins=int(cfg["n_bins"]),
    )

    X_train = pack["X_train"]
    y_train = pack["y_train"]
    X_test = pack["X_test"]
    y_test = pack["y_test"]

    print(
        f"[train] encoded data loaded | "
        f"train={X_train.shape}, test={X_test.shape}"
    )

    result = run_trial(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        cfg=cfg,
    )

    print("[train] result:")
    print(json.dumps(result, indent=2))