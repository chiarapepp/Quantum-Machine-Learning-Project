"""
Training loop for QNN intrusion detection.

- ensures the balanced processed CSV exists
- loads encoded train/test splits from data_utils.py
- trains one QNN model with hinge loss
- evaluates after each epoch
- saves the best checkpoint by validation/test F1

Usage
-----
python train.py --config configs/simple_paper.json
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
    - benign    (0) -> target +1
    - malicious (1) -> target -1

    This matches the project decision rule:
    - certainty >= 0 -> benign
    - certainty <  0 -> malicious
    """
    target = 1.0 - 2.0 * y_binary.float()
    return torch.clamp(margin - certainty * target, min=0.0).mean()


def save_checkpoint(
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
            k: float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float, np.floating))
        },
    }
    torch.save(payload, path)


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Train one QNN configuration and return best validation metrics.
    """
    run_name = str(cfg.get("run_name", "trial"))
    use_cuda = bool(cfg.get("use_cuda", False))
    device_str = "cuda" if (torch.cuda.is_available() and use_cuda) else "cpu"

    use_wandb = _WANDB and bool(cfg.get("use_wandb", False))
    if use_wandb:
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

    opt_name = str(cfg.get("optimizer", "sgd")).lower()
    if opt_name == "sgd":
        optimizer = optim.SGD(
            qnn.parameters(),
            lr=float(cfg["lr"]),
            momentum=float(cfg.get("momentum", 0.0)),
        )
    elif opt_name == "adam":
        optimizer = optim.Adam(
            qnn.parameters(),
            lr=float(cfg["lr"]),
        )
    else:
        raise ValueError(f"Unknown optimizer '{opt_name}'.")

    lr_decay = float(cfg.get("lr_decay", 0.0))
    scheduler = ExponentialLR(
        optimizer,
        gamma=max(0.0, 1.0 - lr_decay),
    )

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
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

        if use_wandb:
            wandb.log(epoch_summary)

        if float(val_stats["f1"]) > best_val_f1:
            best_val_f1 = float(val_stats["f1"])
            best_epoch = epoch + 1
            best_metrics = val_stats

            best_ckpt = os.path.join(output_dir, f"{run_name}.pt")
            save_checkpoint(
                best_ckpt,
                model=qnn,
                cfg=cfg,
                epoch=best_epoch,
                metrics=val_stats,
            )

    history_path = os.path.join(output_dir, f"{run_name}_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    result = {
        "best_val_f1": float(best_val_f1),
        "best_epoch": int(best_epoch),
        "best_ckpt": best_ckpt,
        "history_json": history_path,
        "best_accuracy": float(best_metrics.get("accuracy", float("nan"))),
        "best_precision": float(best_metrics.get("precision", float("nan"))),
        "best_recall": float(best_metrics.get("recall", float("nan"))),
        "best_roc_auc": float(best_metrics.get("roc_auc", float("nan"))),
        "best_cert_mean": float(best_metrics.get("cert_mean", float("nan"))),
        "best_cert_std": float(best_metrics.get("cert_std", float("nan"))),
    }

    if use_wandb:
        wandb.log({"best_val_f1": best_val_f1, "best_epoch": best_epoch})
        wandb.finish()

    print(
        f"[{run_name}] done | "
        f"best_f1={result['best_val_f1']:.4f} | "
        f"best_epoch={result['best_epoch']} | "
        f"ckpt={result['best_ckpt']}"
    )

    return result


def load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    required_keys = [
        "data_csv",
        "raw_csv",
        "n_bins",
        "test_size",
        "split_random_state",
        "arch",
        "n_feature_qubits",
        "n_layers",
        "layer_type",
        "optimizer",
        "lr",
        "momentum",
        "lr_decay",
        "batch_size",
        "hinge_margin",
        "epochs",
        "output_dir",
        "use_cuda",
        "use_wandb",
    ]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise ValueError(f"Config file is missing required keys: {missing}")

    if "run_name" not in cfg:
        cfg["run_name"] = os.path.splitext(os.path.basename(path))[0]

    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if not os.path.exists(cfg["data_csv"]):
        print("[train] processed CSV not found, generating it from raw CSV...")
        os.makedirs(os.path.dirname(cfg["data_csv"]) or ".", exist_ok=True)
        ds_module.build_processed_nf_unsw(
            csv_path=cfg["raw_csv"],
            save_processed_csv=cfg["data_csv"],
        )

    pack = data_utils.load_encoded_splits(
        processed_csv=str(cfg["data_csv"]),
        test_size=float(cfg["test_size"]),
        random_state=int(cfg["split_random_state"]),
        n_bins=int(cfg["n_bins"]),
    )

    print(
        f"[train] encoded data loaded | "
        f"train={pack['X_train'].shape}, test={pack['X_test'].shape}"
    )

    result = train(
        X_train=pack["X_train"],
        y_train=pack["y_train"],
        X_val=pack["X_test"],
        y_val=pack["y_test"],
        cfg=cfg,
    )

    print("[train] result:")
    print(json.dumps(result, indent=2))