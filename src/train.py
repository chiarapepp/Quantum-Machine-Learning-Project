from __future__ import annotations

import os
import json
import time
import argparse
from typing import Any, Dict, Tuple

import numpy as np
import pennylane as qml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import architectures as archs
import data_utils
import dataset as ds_module
from certainty_factor import (
    certainty_from_expval,
    predict_many_from_certainty,
    confidences_from_certainties,
)


def hinge_loss_from_outputs(outputs: np.ndarray, y: np.ndarray, margin: float = 1.0) -> float:
    """
    Hinge loss with labels mapped as:
    benign (0)    -> +1
    malicious (1) -> -1
    """
    y_signed = 1.0 - 2.0 * y.astype(float)
    losses = np.maximum(0.0, margin - outputs * y_signed)
    return float(np.mean(losses))


def build_qnode(
    arch: str,
    n_feature_qubits: int,
    n_layers: int,
    layer_type: str,
    shots: int | None = None,
    noisy: bool = False,
):
    n_wires = n_feature_qubits + (1 if arch == "simple" else 0)
    backend = "default.mixed" if noisy else "default.qubit"
    dev = qml.device(backend, wires=n_wires, shots=shots)

    arch = arch.lower()

    if arch == "simple":
        qnode = archs.build_simple_qnn(
            n_feature_qubits=n_feature_qubits,
            n_layers=n_layers,
            dev=dev,
            layer_type=layer_type,
            interface="autograd",
        )
        n_params = archs.simple_num_params(n_feature_qubits, n_layers, layer_type)

    elif arch == "ttn":
        qnode = archs.build_ttn_qnn(
            n_qubits=n_feature_qubits,
            dev=dev,
            interface="autograd",
        )
        n_params = archs.ttn_num_params(n_feature_qubits)

    elif arch == "mera":
        qnode = archs.build_mera_qnn(
            n_qubits=n_feature_qubits,
            dev=dev,
            interface="autograd",
        )
        n_params = archs.mera_num_params(n_feature_qubits)

    elif arch == "qcnn":
        qnode = archs.build_qcnn_qnn(
            n_qubits=n_feature_qubits,
            dev=dev,
            interface="autograd",
        )
        n_params = archs.qcnn_num_params(n_feature_qubits)

    else:
        raise ValueError(f"Unknown architecture '{arch}'.")

    return qnode, n_params


def predict_outputs(
    qnode,
    X: np.ndarray,
    params: np.ndarray,
) -> np.ndarray:
    """
    Run the QNN on all samples and return raw expectation values.
    """
    outputs = [qnode(x, params) for x in X]
    return np.asarray(outputs, dtype=float)


def evaluate_qnode(
    qnode,
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    threshold: float = 0.0,
    benign_label: int = 0,
    malicious_label: int = 1,
) -> Dict[str, Any]:
    outputs = predict_outputs(qnode, X, params)
    certainties = np.array([certainty_from_expval(v) for v in outputs], dtype=float)

    preds = predict_many_from_certainty(
        certainties,
        threshold=threshold,
        benign_label=benign_label,
        malicious_label=malicious_label,
    )
    confs = confidences_from_certainties(certainties)

    f1 = f1_score(y, preds, pos_label=malicious_label, zero_division=0)
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, pos_label=malicious_label, zero_division=0)
    rec = recall_score(y, preds, pos_label=malicious_label, zero_division=0)

    try:
        malicious_score = (-certainties + 1.0) / 2.0
        roc = roc_auc_score(y, malicious_score)
    except Exception:
        roc = float("nan")

    return {
        "f1": float(f1),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "roc_auc": float(roc),
        "cert_mean": float(np.mean(certainties)),
        "cert_std": float(np.std(certainties)),
        "conf_mean": float(np.mean(confs)),
        "conf_std": float(np.std(confs)),
        "Cs": certainties,
        "confidence": confs,
        "y": np.asarray(y, dtype=int),
        "preds": preds,
    }


def make_minibatches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
):
    idx = np.arange(len(X))
    rng.shuffle(idx)

    for start in range(0, len(X), batch_size):
        batch_idx = idx[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    run_name = str(cfg.get("run_name", "trial"))

    qnode, n_params = build_qnode(
        arch=str(cfg["arch"]),
        n_feature_qubits=int(cfg["n_feature_qubits"]),
        n_layers=int(cfg["n_layers"]),
        layer_type=str(cfg.get("layer_type", "XXYY")),
        shots=None,
        noisy=False,
    )

    rng = np.random.default_rng(int(cfg.get("seed", 1234)))
    params = qml.numpy.array(
        rng.uniform(0.0, 2.0 * np.pi, size=n_params),
        requires_grad=True,
    )

    opt_name = str(cfg.get("optimizer", "adam")).lower()
    lr = float(cfg["lr"])

    if opt_name == "adam":
        optimizer = qml.AdamOptimizer(stepsize=lr)
    elif opt_name == "sgd":
        optimizer = qml.GradientDescentOptimizer(stepsize=lr)
    else:
        raise ValueError("Supported optimizers: 'adam', 'sgd'.")

    batch_size = int(cfg["batch_size"])
    epochs = int(cfg["epochs"])
    margin = float(cfg.get("hinge_margin", 1.0))
    output_dir = str(cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    def cost_fn(current_params, xb, yb):
        outputs = [qnode(x, current_params) for x in xb]
        outputs = qml.numpy.stack(outputs)

        y_signed = 1.0 - 2.0 * yb.astype(float)
        losses = qml.numpy.maximum(0.0, margin - outputs * y_signed)
        return qml.numpy.mean(losses)

    history = []
    best_val_f1 = -1.0
    best_epoch = -1
    best_metrics: Dict[str, Any] = {}
    best_params = None

    for epoch in range(epochs):
        t0 = time.time()
        batch_losses = []

        for xb, yb in make_minibatches(X_train, y_train, batch_size, rng):
            xb_q = qml.numpy.array(xb, requires_grad=False)
            yb_q = qml.numpy.array(yb, requires_grad=False)

            params, batch_loss = optimizer.step_and_cost(
                lambda p: cost_fn(p, xb_q, yb_q),
                params,
            )
            batch_losses.append(float(batch_loss))

        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")

        val_stats = evaluate_qnode(
            qnode=qnode,
            X=X_val,
            y=y_val,
            params=params,
        )

        summary = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_f1": float(val_stats["f1"]),
            "val_accuracy": float(val_stats["accuracy"]),
            "val_precision": float(val_stats["precision"]),
            "val_recall": float(val_stats["recall"]),
            "val_roc_auc": float(val_stats["roc_auc"]),
            "cert_mean": float(val_stats["cert_mean"]),
            "cert_std": float(val_stats["cert_std"]),
            "epoch_time_sec": float(time.time() - t0),
        }
        history.append(summary)

        print(
            f"[{run_name}] epoch {epoch + 1:03d}/{epochs:03d} | "
            f"loss={train_loss:.4f} | "
            f"f1={val_stats['f1']:.4f} | "
            f"acc={val_stats['accuracy']:.4f} | "
            f"prec={val_stats['precision']:.4f} | "
            f"rec={val_stats['recall']:.4f} | "
            f"auc={val_stats['roc_auc']:.4f}"
        )

        if float(val_stats["f1"]) > best_val_f1:
            best_val_f1 = float(val_stats["f1"])
            best_epoch = epoch + 1
            best_metrics = val_stats
            best_params = np.array(params, dtype=float)

    history_path = os.path.join(output_dir, f"{run_name}_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    best_path = os.path.join(output_dir, f"{run_name}_best.npz")
    np.savez(
        best_path,
        params=np.asarray(best_params, dtype=float),
        config=json.dumps(cfg),
        best_epoch=best_epoch,
        best_val_f1=best_val_f1,
    )

    result = {
        "best_val_f1": float(best_val_f1),
        "best_epoch": int(best_epoch),
        "best_ckpt": best_path,
        "history_json": history_path,
        "best_accuracy": float(best_metrics.get("accuracy", float("nan"))),
        "best_precision": float(best_metrics.get("precision", float("nan"))),
        "best_recall": float(best_metrics.get("recall", float("nan"))),
        "best_roc_auc": float(best_metrics.get("roc_auc", float("nan"))),
        "best_cert_mean": float(best_metrics.get("cert_mean", float("nan"))),
        "best_cert_std": float(best_metrics.get("cert_std", float("nan"))),
    }

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
        "batch_size",
        "hinge_margin",
        "epochs",
        "output_dir",
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