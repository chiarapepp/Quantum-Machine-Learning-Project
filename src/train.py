# src/train.py
"""
Training harness that uses your actual project files:
 - src/dataset.py         -> load_and_prepare_nf_unsw(csv_path, save_processed_csv=None)
 - src/encoding.py        -> QuantumEncoder
 - src/preprocessing.py   -> (not required, included for completeness)
 - src/architectures.py   -> build_simple_qnn, build_ttn_qnn, build_mera_qnn, build_qcnn_qnn
 - src/certainty_factor.py -> helpers (optional)

This script:
 - creates/reads balanced processed CSV via your dataset loader
 - builds a QuantumEncoder from the full processed DF (so percentile bins are global)
 - encodes dataset to angles (Nx8)
 - builds QNN qnode using your architectures.* builders and wraps in a torch.nn.Module
 - trains with hinge loss on ±1 labels (paper), logs to W&B
 - supports a reduced grid search or a single trial via --just_one
"""

import os
import math
import time
import json
import argparse
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

import wandb

# Your modules (exact filenames you provided)
from src import dataset as dataset_module
from src import encoding as encoding_module
from src import architectures as archs_module
from src import certainty_factor as cf_module

# -------------------------
# Column name mapping
# -------------------------
# dataset.py uses uppercase NetFlow names (FEATURE_COLUMNS). encoding expects lowercase names.
COLUMN_RENAME_MAP = {
    "PROTOCOL": "ip_protocol",
    "TCP_FLAGS": "tcp_flags",
    "L7_PROTO": "layer7_protocol",
    "IN_BYTES": "in_bytes",
    "OUT_BYTES": "out_bytes",
    "IN_PKTS": "in_packets",
    "OUT_PKTS": "out_packets",
    "FLOW_DURATION_MILLISECONDS": "flow_duration",
}

ENCODER_FEATURE_ORDER = [
    "ip_protocol",
    "tcp_flags",
    "layer7_protocol",
    "in_bytes",
    "out_bytes",
    "in_packets",
    "out_packets",
    "flow_duration",
]

# -------------------------
# Helpers
# -------------------------
def map_params_count_for_arch(arch_name: str, n_qubits: int, n_layers: int) -> int:
    """Return expected number of scalar params (6 per two-qubit block) using the same conventions as architectures.py."""
    two_q_params = 6
    if arch_name == "simple":
        pairs = max(1, n_qubits // 2)
        return n_layers * pairs * two_q_params
    elif arch_name == "ttn":
        # sum pairs per layer: floor(n/2) + floor(n/4) + ...
        total_pairs = 0
        active = n_qubits
        while active > 1:
            total_pairs += active // 2
            active = math.ceil(active / 2)
        return total_pairs * two_q_params
    elif arch_name == "mera":
        return n_layers * n_qubits * two_q_params
    elif arch_name == "qcnn":
        total_pairs = 0
        active = n_qubits
        while active > 1:
            total_pairs += active // 2
            active = math.ceil(active / 2)
        # QCNN implementation used two param-blocks per pair in that code
        return total_pairs * two_q_params * 2
    else:
        raise ValueError("Unknown arch: " + arch_name)


class QNNTorchWrapper(nn.Module):
    """
    Wrap a qnode builder (from your architectures.py) into an nn.Module.
    Stores a single flattened parameter vector as nn.Parameter and calls the qnode per-sample.
    Assumes qnode signature like: qnode(x, params) where params is a flat 1D vector.
    """
    def __init__(self, arch_name: str, n_qubits: int, n_layers: int, qml_device, shots: int = None):
        super().__init__()
        self.arch = arch_name
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.qml_dev = qml_device
        self.shots = shots

        # choose builder
        if arch_name == "simple":
            self.qnode = archs_module.build_simple_qnn(n_qubits, n_layers, self.qml_dev, result_wire=n_qubits-1, shots=shots)
        elif arch_name == "ttn":
            self.qnode = archs_module.build_ttn_qnn(n_qubits, self.qml_dev, shots=shots)
        elif arch_name == "mera":
            self.qnode = archs_module.build_mera_qnn(n_qubits, n_layers, self.qml_dev, shots=shots)
        elif arch_name == "qcnn":
            self.qnode = archs_module.build_qcnn_qnn(n_qubits, self.qml_dev, shots=shots)
        else:
            raise ValueError("Unknown arch: " + arch_name)

        # parameter vector
        n_params = map_params_count_for_arch(arch_name, n_qubits, n_layers)
        init = 0.01 * torch.randn(n_params, requires_grad=True)
        self.qparams = nn.Parameter(init)

    def forward(self, x_batch: torch.Tensor) -> torch.Tensor:
        # x_batch: (batch, n_qubits) angles (radians)
        outs = []
        for i in range(x_batch.shape[0]):
            xi = x_batch[i]
            # call qnode; qnode interface is 'torch' so xi and self.qparams are ok
            val = self.qnode(xi, self.qparams)
            # Ensure tensor
            if not isinstance(val, torch.Tensor):
                val = torch.tensor(float(val), dtype=torch.float32)
            outs.append(val)
        return torch.stack(outs).view(-1)  # shape (batch,)


# Hinge loss implemented on ±1 labels
def hinge_loss(C: torch.Tensor, y_binary: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """
    C: model outputs (certainty) in [-1,1] (torch tensor, shape (batch,))
    y_binary: {0,1} (torch tensor)
    Paper: convert True/False to +1/-1 before hinge. Here we map malicious (1) -> +1, benign (0) -> -1.
    Hinge: max(0, margin - C * t)
    """
    t = torch.where(y_binary == 1, torch.tensor(1.0, device=y_binary.device), torch.tensor(-1.0, device=y_binary.device))
    zeros = torch.zeros_like(t)
    loss = torch.max(zeros, margin - C * t)
    return loss.mean()


def accuracy_from_certainty_array(Cs: np.ndarray, y: np.ndarray) -> float:
    # predict 1 if C>=0 else 0
    preds = (Cs >= 0).astype(int)
    return float(accuracy_score(y, preds))

def hinge_accuracy(y_true, y_pred):
    y_true = torch.squeeze(y_true)
    y_pred = torch.squeeze(y_pred)

    result_bool  = torch.eq(y_true, y_pred)
    result = result_bool.float() 
    return torch.mean(result)

# -------------------------
# Training / evaluation
# -------------------------
def evaluate_model(model: QNNTorchWrapper, X: np.ndarray, y: np.ndarray, batch_size: int = 64, device='cpu') -> Dict[str, Any]:
    model.eval()
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    all_C = []
    all_y = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb).cpu().numpy()
            all_C.append(out)
            all_y.append(yb.numpy())
    Cs = np.concatenate(all_C, axis=0)
    ys = np.concatenate(all_y, axis=0)
    preds = (Cs >= 0).astype(int)
    f1 = f1_score(ys, preds, zero_division=0)
    acc = accuracy_score(ys, preds)
    prec = precision_score(ys, preds, zero_division=0)
    rec = recall_score(ys, preds, zero_division=0)
    try:
        roc = roc_auc_score(ys, (Cs + 1) / 2.0)
    except Exception:
        roc = float('nan')
    return {
        "f1": float(f1),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "roc_auc": float(roc),
        "cert_mean": float(np.mean(Cs)),
        "cert_std": float(np.std(Cs)),
        "Cs": Cs,
        "y": ys
    }


def run_one_trial(X_train, y_train, X_val, y_val, cfg: Dict[str, Any]):
    # wandb init
    wandb.init(project=cfg["wandb_project"], config=cfg, reinit=True, name=cfg.get("run_name"))
    config = wandb.config

    # device
    device = torch.device("cuda" if (torch.cuda.is_available() and config.use_cuda) else "cpu")

    # create qml device (analytic)
    import pennylane as qml
    qml_dev = qml.device("default.qubit", wires=config.n_qubits, shots=None if not config.use_shots else config.shots)

    # build model wrapper
    model = QNNTorchWrapper(config.arch, config.n_qubits, config.n_layers, qml_dev, shots=(config.shots if config.use_shots else None))
    model.to(device)

    # optimizer
    if config.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

    # dataloaders
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

    best_val_f1 = -1.0
    best_ckpt = None
    history = []

    for epoch in range(config.epochs):
        model.train()
        losses = []
        start = time.time()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)  # certainty C
            loss = hinge_loss(out, yb, margin=config.hinge_margin)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        elapsed = time.time() - start

        # eval
        eval_stats = evaluate_model(model, X_val, y_val, batch_size=config.batch_size, device=device)
        wandb.log({
            "epoch": epoch,
            "train/loss": float(np.mean(losses)),
            "val/f1": eval_stats["f1"],
            "val/accuracy": eval_stats["accuracy"],
            "val/precision": eval_stats["precision"],
            "val/recall": eval_stats["recall"],
            "val/roc_auc": eval_stats["roc_auc"],
            "val/cert_mean": eval_stats["cert_mean"],
            "val/cert_std": eval_stats["cert_std"]
        }, step=epoch)

        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **eval_stats})

        if eval_stats["f1"] > best_val_f1:
            best_val_f1 = eval_stats["f1"]
            ckpt_dir = cfg["output_dir"]
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"best_{config.arch}_L{config.n_layers}_lr{config.lr}_bs{config.batch_size}.pt")
            torch.save({
                "model_state": model.state_dict(),
                "config": dict(config),
                "epoch": epoch
            }, ckpt_path)
            best_ckpt = ckpt_path

        print(f"[{config.arch}] epoch {epoch+1}/{config.epochs} train_loss {np.mean(losses):.4f} val_f1 {eval_stats['f1']:.4f} time {elapsed:.1f}s")

    wandb.log({"best_val_f1": best_val_f1})
    wandb.finish()

    return {"best_val_f1": best_val_f1, "best_ckpt": best_ckpt, "history": history}


# -------------------------
# CLI / main
# -------------------------
DEFAULT_CONFIG = {
    "data_csv": "data/processed/nf_unsw_balanced.csv",
    "raw_csv": "data/raw/NF-UNSW-NB15-v2.csv",
    "n_qubits": 8,
    "wandb_project": "naduqnn_repro",
    "output_dir": "results/checkpoints",
    "use_cuda": False,
    "use_shots": False,
    "shots": 200,
    "epochs": 8,
    "hinge_margin": 1.0,
    # small grid defaults (paper used large grid; reduce here for practicality)
    "grid_batch_sizes": [16, 32],
    "grid_lrs": [0.01, 0.005, 0.001],
    "grid_optimizers": ["adam", "sgd"],
    "grid_momenta": [0.0, 0.2],
    "grid_weight_decays": [0.0, 0.001],
    "grid_architectures": ["ttn", "mera", "qcnn", "simple"],
    "grid_simple_layers": [1, 2, 4, 6],
    "default_layers": 2,
    "batch_size": 32
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--just_one", action="store_true", help="Run one default training trial (fast).")
    parser.add_argument("--data_csv", type=str, default=DEFAULT_CONFIG["data_csv"])
    parser.add_argument("--raw_csv", type=str, default=DEFAULT_CONFIG["raw_csv"])
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    cfg["data_csv"] = args.data_csv
    cfg["raw_csv"] = args.raw_csv

    # 1) create balanced processed CSV via your dataset loader (it will save when asked)
    # We'll call loader with save_processed_csv to produce the CSV we expect for encoding
    print("Preparing dataset via dataset.load_and_prepare_nf_unsw(...)")
    Xtr, Xte, ytr, yte = dataset_module.load_and_prepare_nf_unsw(cfg["raw_csv"], save_processed_csv=cfg["data_csv"])
    # NOTE: the loader already returns numpy arrays but we want the DataFrame for encoder (global percentiles)
    proc_df = pd.read_csv(cfg["data_csv"])

    # 2) rename columns to match encoder expected names
    missing_rename = [c for c in COLUMN_RENAME_MAP if c not in proc_df.columns]
    if missing_rename:
        raise RuntimeError(f"Processed CSV is missing expected columns for renaming: {missing_rename}. Columns present: {list(proc_df.columns)}")
    proc_df = proc_df.rename(columns=COLUMN_RENAME_MAP)
    # Keep only the encoder features in the expected order
    proc_df = proc_df[ENCODER_FEATURE_ORDER + ["label_binary"]]

    # 3) Build encoder (global) and encode whole dataset to angles
    print("Building QuantumEncoder (global percentiles) and encoding dataset to angles...")
    encoder = encoding_module.QuantumEncoder(proc_df, n_bins=100)
    angles_all = encoder.encode_dataset()  # Nx8

    # split back into train/test using sizes from dataset loader outputs
    n_train = Xtr.shape[0]
    X_train_angles = angles_all[:n_train, :]
    X_test_angles = angles_all[n_train:, :]
    y_train = ytr
    y_test = yte

    print(f"Encoded dataset shapes -> train: {X_train_angles.shape}, test: {X_test_angles.shape}")

    # 4) Run either single trial or small grid
    if args.just_one:
        # single trial: simple architecture, 2 layers, Adam, lr=0.01, bs=32
        trial_cfg = {
            "arch": "simple",
            "n_layers": 2,
            "n_qubits": cfg["n_qubits"],
            "batch_size": 32,
            "lr": 0.01,
            "optimizer": "adam",
            "momentum": 0.0,
            "weight_decay": 0.0,
            "epochs": cfg["epochs"],
            "hinge_margin": cfg["hinge_margin"],
            "output_dir": cfg["output_dir"],
            "wandb_project": cfg["wandb_project"],
            "use_cuda": cfg["use_cuda"],
            "use_shots": cfg["use_shots"],
            "shots": cfg["shots"]
        }
        run_res = run_one_trial(X_train_angles, y_train, X_test_angles, y_test, trial_cfg)
        print("Single-run finished:", run_res)
    else:
        results = []
        trial_id = 0
        for arch in cfg["grid_architectures"]:
            layers_list = cfg["grid_simple_layers"] if arch == "simple" else [cfg["default_layers"]]
            for n_layers in layers_list:
                for bs in cfg["grid_batch_sizes"]:
                    for lr in cfg["grid_lrs"]:
                        for opt in cfg["grid_optimizers"]:
                            for wd in cfg["grid_weight_decays"]:
                                momenta = cfg["grid_momenta"] if opt == "sgd" else [0.0]
                                for mom in momenta:
                                    trial_id += 1
                                    trial_cfg = {
                                        "arch": arch,
                                        "n_layers": n_layers,
                                        "n_qubits": cfg["n_qubits"],
                                        "batch_size": bs,
                                        "lr": lr,
                                        "optimizer": opt,
                                        "momentum": mom,
                                        "weight_decay": wd,
                                        "epochs": cfg["epochs"],
                                        "hinge_margin": cfg["hinge_margin"],
                                        "output_dir": cfg["output_dir"],
                                        "wandb_project": cfg["wandb_project"],
                                        "use_cuda": cfg["use_cuda"],
                                        "use_shots": cfg["use_shots"],
                                        "shots": cfg["shots"],
                                        "run_name": f"trial{trial_id}_{arch}_L{n_layers}_bs{bs}_opt{opt}_lr{lr}_wd{wd}_mom{mom}"
                                    }
                                    print("Starting trial:", trial_cfg["run_name"])
                                    res = run_one_trial(X_train_angles, y_train, X_test_angles, y_test, trial_cfg)
                                    res_entry = {"trial_name": trial_cfg["run_name"], "config": trial_cfg, **res}
                                    results.append(res_entry)
                                    # save after each trial
                                    os.makedirs("results", exist_ok=True)
                                    with open("results/grid_summary.json", "w") as f:
                                        json.dump(results, f, indent=2)
        print("Grid finished. Summary saved to results/grid_summary.json")
