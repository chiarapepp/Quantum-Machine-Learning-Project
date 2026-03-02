# src/train.py
import os
import math
import time
import json
import argparse
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import wandb
import pennylane as qml

import dataset as dataset_module
import encoding as encoding_module
import architectures as archs_module
import noise as noise_module


# -------------------------
# Paths (robust)
# -------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root


# -------------------------
# Column mapping
# -------------------------
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
# Param counting
# -------------------------
def _simple_params_per_pair(layer_type: str) -> int:
    lt = layer_type.upper()
    if lt == "ZZXXYY":
        return 3
    if lt in ("ZZXX", "XXYY", "ZZYY"):
        return 2
    raise ValueError(f"Unknown layer_type '{layer_type}'. Use: ZZXXYY, ZZXX, XXYY, ZZYY.")


def map_params_count_for_arch(arch_name: str, n_qubits: int, n_layers: int, layer_type: str = "XXYY") -> int:
    """
    Return expected number of scalar params.
    - Simple uses paper layer types (2 or 3 params per pair)
    - TTN/MERA/QCNN use Rot+Rot+CNOT blocks (6 params per pair) as in your code
    """
    two_q_params = 6

    if arch_name == "simple":
        pairs = max(1, n_qubits // 2)
        p_per_pair = _simple_params_per_pair(layer_type)
        return n_layers * pairs * p_per_pair

    elif arch_name == "ttn":
        total_pairs = 0
        active = n_qubits
        while active > 1:
            total_pairs += active // 2
            active = math.ceil(active / 2)
        return total_pairs * two_q_params

    elif arch_name == "mera":
        # your MERA code uses 2*n_layers*(n_qubits-1)/? blocks; we keep your previous convention:
        # n_layers * n_qubits blocks (approx). It's ok as long as consistent with architectures implementation.
        return n_layers * n_qubits * two_q_params

    elif arch_name == "qcnn":
        total_pairs = 0
        active = n_qubits
        while active > 1:
            total_pairs += active // 2
            active = math.ceil(active / 2)
        # two blocks per pair
        return total_pairs * two_q_params * 2

    else:
        raise ValueError("Unknown arch: " + arch_name)


# -------------------------
# Model wrapper
# -------------------------
class QNNTorchWrapper(nn.Module):
    """Torch wrapper around a PennyLane QNode builder."""

    def __init__(
        self,
        arch_name: str,
        n_qubits: int,
        n_layers: int,
        qml_device,
        shots: Optional[int] = None,
        noise_model=None,
        layer_type: str = "XXYY",
    ):
        super().__init__()
        self.arch = arch_name
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.qml_dev = qml_device
        self.shots = shots
        self.noise_model = noise_model
        self.layer_type = layer_type

        if arch_name == "simple":
            self.qnode = archs_module.build_simple_qnn(
                n_qubits=n_qubits,
                n_layers=n_layers,
                dev=self.qml_dev,
                result_wire=n_qubits - 1,
                shots=shots,
                noise_model=noise_model,
                layer_type=layer_type,   # IMPORTANT
            )
        elif arch_name == "ttn":
            self.qnode = archs_module.build_ttn_qnn(n_qubits, self.qml_dev, shots=shots, noise_model=noise_model)
        elif arch_name == "mera":
            self.qnode = archs_module.build_mera_qnn(n_qubits, n_layers, self.qml_dev, shots=shots, noise_model=noise_model)
        elif arch_name == "qcnn":
            self.qnode = archs_module.build_qcnn_qnn(n_qubits, self.qml_dev, shots=shots, noise_model=noise_model)
        else:
            raise ValueError("Unknown arch: " + arch_name)

        n_params = map_params_count_for_arch(arch_name, n_qubits, n_layers, layer_type=layer_type)
        init = 0.01 * torch.randn(n_params)
        self.qparams = nn.Parameter(init)

    def forward(self, x_batch: torch.Tensor) -> torch.Tensor:
        outs = []
        for i in range(x_batch.shape[0]):
            xi = x_batch[i]
            val = self.qnode(xi, self.qparams)
            if not isinstance(val, torch.Tensor):
                val = torch.tensor(float(val), dtype=torch.float32)
            outs.append(val)
        return torch.stack(outs).view(-1)


# -------------------------
# Loss + metrics
# -------------------------
def hinge_loss(C: torch.Tensor, y_binary: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """Hinge loss on ±1 targets (malicious=+1, benign=-1)."""
    t = torch.where(
        y_binary == 1,
        torch.tensor(1.0, device=y_binary.device),
        torch.tensor(-1.0, device=y_binary.device),
    )
    loss = torch.clamp(margin - C * t, min=0.0)
    return loss.mean()


def evaluate_model(
    model: QNNTorchWrapper,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    device: str = "cpu",
) -> Dict[str, Any]:
    model.eval()
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_C, all_y = [], []
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="eval", leave=False):
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
        roc = float("nan")

    return {
        "f1": float(f1),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "roc_auc": float(roc),
        "cert_mean": float(np.mean(Cs)),
        "cert_std": float(np.std(Cs)),
        "Cs": Cs,
        "y": ys,
    }


# -------------------------
# Data: split + encode
# -------------------------
def prepare_angles_from_processed_csv(processed_csv_path: str) -> Dict[str, Any]:
    proc_df = pd.read_csv(processed_csv_path)

    missing_rename = [c for c in COLUMN_RENAME_MAP if c not in proc_df.columns]
    if missing_rename:
        raise RuntimeError(f"Processed CSV missing expected columns: {missing_rename}. Present: {list(proc_df.columns)}")

    proc_df = proc_df.rename(columns=COLUMN_RENAME_MAP)
    if "label_binary" not in proc_df.columns:
        raise RuntimeError("Processed CSV must contain 'label_binary'.")

    proc_df = proc_df[ENCODER_FEATURE_ORDER + ["label_binary"]].copy()

    y_all = proc_df["label_binary"].to_numpy(dtype=int)
    idx_all = proc_df.index.to_numpy()

    train_idx, test_idx = train_test_split(
        idx_all, test_size=0.15, random_state=1, shuffle=True, stratify=y_all
    )

    train_df = proc_df.loc[train_idx].reset_index(drop=True)
    test_df = proc_df.loc[test_idx].reset_index(drop=True)

    # Fit encoder on TRAIN ONLY
    encoder = encoding_module.QuantumEncoder(train_df, n_bins=100)

    X_train_angles = encoder.encode_dataset(train_df)
    X_test_angles = encoder.encode_dataset(test_df)

    y_train = train_df["label_binary"].to_numpy(dtype=int)
    y_test = test_df["label_binary"].to_numpy(dtype=int)

    return {
        "X_train": X_train_angles,
        "y_train": y_train,
        "X_test": X_test_angles,
        "y_test": y_test,
    }


# -------------------------
# Training
# -------------------------
def run_one_trial(X_train, y_train, X_val, y_val, cfg: Dict[str, Any]) -> Dict[str, Any]:
    wandb.init(project=cfg["wandb_project"], config=cfg, name=cfg.get("run_name"))
    config = wandb.config

    device = torch.device("cuda" if (torch.cuda.is_available() and config.use_cuda) else "cpu")

    # Noiseless training device
    qml_dev = qml.device("default.qubit", wires=config.n_qubits, shots=None)

    model = QNNTorchWrapper(
        arch_name=config.arch,
        n_qubits=config.n_qubits,
        n_layers=config.n_layers,
        qml_device=qml_dev,
        shots=None,
        noise_model=None,
        layer_type=getattr(config, "layer_type", "XXYY"),
    ).to(device)

    if config.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

    best_val_f1 = -1.0
    best_ckpt = None

    for epoch in range(config.epochs):
        model.train()
        losses = []
        start = time.time()

        pbar = tqdm(train_loader, desc=f"train epoch {epoch+1}/{config.epochs}", leave=False)
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            out = model(xb)
            loss = hinge_loss(out, yb, margin=config.hinge_margin)
            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))
            pbar.set_postfix(loss=float(np.mean(losses)))

        elapsed = time.time() - start

        eval_stats = evaluate_model(model, X_val, y_val, batch_size=config.batch_size, device=device)
        wandb.log({
            "epoch": epoch,
            "train/loss": float(np.mean(losses)),
            "val/f1": eval_stats["f1"],
            "val/accuracy": eval_stats["accuracy"],
            "val/precision": eval_stats["precision"],
            "val/recall": eval_stats["recall"],
            "val/roc_auc": eval_stats["roc_auc"],
        }, step=epoch)

        if eval_stats["f1"] > best_val_f1:
            best_val_f1 = eval_stats["f1"]
            ckpt_dir = cfg["output_dir"]
            os.makedirs(ckpt_dir, exist_ok=True)
            lt = getattr(config, "layer_type", "na")
            ckpt_path = os.path.join(ckpt_dir, f"best_{config.arch}_L{config.n_layers}_{lt}.pt")
            torch.save({"model_state": model.state_dict(), "config": dict(config), "epoch": epoch}, ckpt_path)
            best_ckpt = ckpt_path

        print(f"[{config.arch}] epoch {epoch+1}/{config.epochs} "
              f"train_loss {np.mean(losses):.4f} val_f1 {eval_stats['f1']:.4f} time {elapsed:.1f}s")

    wandb.log({"best_val_f1": best_val_f1})
    wandb.finish()
    return {"best_val_f1": best_val_f1, "best_ckpt": best_ckpt}


# -------------------------
# Noise evaluation
# -------------------------
def evaluate_under_noise(
    trained_state_dict,
    cfg: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    noise_level: str,
    shots: int = 200
) -> Dict[str, Any]:
    """
    Rebuild the SAME architecture but on a noisy device, load trained params, evaluate.
    This is inference-only (no backprop).
    """
    noise_model = noise_module.get_paper_noise_model(noise_level)
    dev_noisy = noise_module.create_noisy_device(cfg["n_qubits"], noise_model, shots=shots)

    model_noisy = QNNTorchWrapper(
        arch_name=cfg["arch"],
        n_qubits=cfg["n_qubits"],
        n_layers=cfg["n_layers"],
        qml_device=dev_noisy,
        shots=shots,
        noise_model=noise_model,
        layer_type=cfg.get("layer_type", "XXYY"),
    )
    model_noisy.load_state_dict(trained_state_dict, strict=True)

    stats = evaluate_model(model_noisy, X_test, y_test, batch_size=cfg["batch_size"], device="cpu")
    return stats


# -------------------------
# Defaults
# -------------------------
DEFAULT_CONFIG = {
    "data_csv": "data/processed/nf_unsw_balanced.csv",
    "raw_csv": "data/raw/NF-UNSW-NB15-v2.csv",
    "n_qubits": 8,
    "wandb_project": "naduqnn_repro",
    "output_dir": "results/checkpoints",
    "use_cuda": False,
    "epochs": 8,
    "hinge_margin": 1.0,
    "batch_size": 32,
}


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--just_one", action="store_true", help="Run one default training trial.")
    parser.add_argument("--data_csv", type=str, default=DEFAULT_CONFIG["data_csv"])
    parser.add_argument("--raw_csv", type=str, default=DEFAULT_CONFIG["raw_csv"])
    parser.add_argument("--force_preprocess", action="store_true", help="Force regenerate processed CSV even if it exists.")
    parser.add_argument("--no_noise_sweep", action="store_true", help="Disable noise sweep evaluation after training.")
    parser.add_argument("--noise_shots", type=int, default=200, help="Shots for noisy evaluation (paper often uses 200).")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    cfg["data_csv"] = os.path.join(PROJECT_ROOT, args.data_csv)
    cfg["raw_csv"] = os.path.join(PROJECT_ROOT, args.raw_csv)

    # 1) processed CSV: reuse if present
    if os.path.exists(cfg["data_csv"]) and not args.force_preprocess:
        print(f"[train] Using existing processed CSV: {cfg['data_csv']}")
    else:
        print("[train] Generating processed CSV via dataset.load_and_prepare_nf_unsw(...)")
        os.makedirs(os.path.dirname(cfg["data_csv"]), exist_ok=True)
        dataset_module.load_and_prepare_nf_unsw(cfg["raw_csv"], save_processed_csv=cfg["data_csv"])

    # 2) Split + encode
    print("[train] Loading processed CSV, splitting (85/15 stratified), fitting encoder on train, encoding...")
    pack = prepare_angles_from_processed_csv(cfg["data_csv"])
    X_train_angles, y_train = pack["X_train"], pack["y_train"]
    X_test_angles, y_test = pack["X_test"], pack["y_test"]
    print(f"[train] Encoded shapes -> train: {X_train_angles.shape}, test: {X_test_angles.shape}")
    print(f"[train] Label balance -> train: {np.bincount(y_train)}, test: {np.bincount(y_test)}")

    # 3) Run
    if args.just_one:
        trial_cfg = {
            # Paper best: Simple, 6 layers, XY -> map to layer_type="XXYY"
            "arch": "simple",
            "n_layers": 6,
            "layer_type": "XXYY",          # <--- PAPER "XY"
            "n_qubits": cfg["n_qubits"],

            "batch_size": cfg["batch_size"],   # paper best uses 32
            "lr": 0.02,                        # paper best
            "optimizer": "sgd",                # paper uses SGD for best
            "momentum": 0.0,
            "weight_decay": 0.001,             # paper "decay 0.001"

            "epochs": cfg["epochs"],
            "hinge_margin": cfg["hinge_margin"],
            "output_dir": os.path.join(PROJECT_ROOT, cfg["output_dir"]),
            "wandb_project": cfg["wandb_project"],
            "use_cuda": cfg["use_cuda"],
            "run_name": "paper_best_simple_L6_XXYY",
        }

        res = run_one_trial(X_train_angles, y_train, X_test_angles, y_test, trial_cfg)
        print("[train] Finished:", res)

        # 4) Noise sweep (paper-style)
        if not args.no_noise_sweep and res.get("best_ckpt"):
            ckpt = torch.load(res["best_ckpt"], map_location="cpu")
            state = ckpt["model_state"]

            print("\n[train] Running noise sweep on test set...")
            noise_levels = ["clean", "very_low", "low", "medium", "high", "very_high"]

            sweep = {}
            for lvl in noise_levels:
                print(f"\n[train] Noise level: {lvl}")
                stats = evaluate_under_noise(
                    state, trial_cfg, X_test_angles, y_test,
                    noise_level=lvl, shots=args.noise_shots
                )
                sweep[lvl] = {k: stats[k] for k in ["f1", "accuracy", "precision", "recall", "roc_auc", "cert_mean", "cert_std"]}
                print(f"  F1={sweep[lvl]['f1']:.4f}  Acc={sweep[lvl]['accuracy']:.4f}")

            os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)
            out_path = os.path.join(PROJECT_ROOT, "results", "noise_sweep.json")
            with open(out_path, "w") as f:
                json.dump(sweep, f, indent=2)
            print(f"\n[train] Saved noise sweep to {out_path}")

    else:
        raise RuntimeError("Per ora usa --just_one. La grid search la facciamo dopo che questa pipeline è stabile.")