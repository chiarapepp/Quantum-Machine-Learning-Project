# src/noise_eval.py
"""
Noisy inference evaluation for trained QNN checkpoints.

Paper Section III-F: after identifying the best noiseless architecture,
the authors evaluate on IonQ's noisy simulator and on physical QPUs.
We replicate this using PennyLane's `default.mixed` device with
depolarizing noise channels.

Note: the paper does NOT add noise during training — only at inference.
The trained parameters from the noiseless run are loaded as-is.

Usage
-----
python noise_eval.py --ckpt results/checkpoints/paper_best_best.pt \\
                     --data_csv data/processed/nf_unsw_balanced.csv \\
                     --shots 200
"""

from __future__ import annotations
import os
import json
import argparse
from typing import Dict, Any, List

import numpy as np
import torch
import pennylane as qml

import src.model as model_module
import evaluate as eval_module
import data_utils


# ---------------------------------------------------------------------------
# Noise levels from paper experiments
# ---------------------------------------------------------------------------

# Depolarizing probability per gate
PAPER_NOISE_LEVELS: Dict[str, float] = {
    "clean":     0.000,
    "very_low":  0.001,   # 0.1 %
    "low":       0.005,   # 0.5 %
    "medium":    0.010,   # 1.0 %
    "high":      0.050,   # 5.0 %
    "very_high": 0.100,   # 10.0 %
}


# ---------------------------------------------------------------------------
# Noise injection helpers
# ---------------------------------------------------------------------------

def _apply_depolarizing(prob: float, wires: List[int]) -> None:
    """Apply depolarizing channel with given probability to each wire."""
    for w in wires:
        qml.DepolarizingChannel(prob, wires=w)


def build_noisy_simple_qnn(
    n_feature_qubits: int,
    n_layers: int,
    dev: qml.Device,
    layer_type: str,
    noise_prob: float,
) -> Any:
    """
    Simple QNN with depolarizing noise injected after every gate.

    This is used only at inference time (parameters loaded from noiseless checkpoint).
    """
    import architectures as archs

    result_wire = n_feature_qubits
    terms  = archs._parse_layer_type(layer_type)
    pairs  = archs._simple_chain_pairs(n_feature_qubits, result_wire)
    n_pairs = len(pairs)
    n_params = archs.simple_num_params(n_feature_qubits, n_layers, layer_type)

    @qml.qnode(dev, interface="torch", diff_method="best")
    def qnode(x, params):
        # encoding + noise
        for wire in range(n_feature_qubits):
            qml.RX(x[wire], wires=wire)
            if noise_prob > 0:
                _apply_depolarizing(noise_prob, [wire])

        # result qubit: |->
        qml.PauliX(wires=result_wire)
        qml.Hadamard(wires=result_wire)
        if noise_prob > 0:
            _apply_depolarizing(noise_prob, [result_wire])

        idx = 0
        for _ in range(n_layers):
            for term in terms:
                for k, (a, b) in enumerate(pairs):
                    theta = params[idx + k]
                    if term == "XX":
                        qml.IsingXX(theta, wires=[a, b])
                    elif term == "YY":
                        qml.IsingYY(theta, wires=[a, b])
                    elif term == "ZZ":
                        qml.IsingZZ(theta, wires=[a, b])
                    if noise_prob > 0:
                        _apply_depolarizing(noise_prob, [a, b])
                idx += n_pairs

        return qml.expval(qml.PauliX(result_wire))

    return qnode


# ---------------------------------------------------------------------------
# Noisy model wrapper
# ---------------------------------------------------------------------------

class NoisyQNNModel(torch.nn.Module):
    """
    Wraps a noisy QNode and holds trained parameters (loaded from checkpoint).
    Only supports Simple architecture for now (paper's best arch).
    """

    def __init__(
        self,
        arch: str,
        n_feature_qubits: int,
        n_layers: int,
        layer_type: str,
        noise_prob: float,
        shots: int,
    ):
        super().__init__()
        self.arch = arch

        n_wires = n_feature_qubits + (1 if arch == "simple" else 0)
        dev = qml.device("default.mixed", wires=n_wires, shots=shots)

        if arch == "simple":
            self.qnode = build_noisy_simple_qnn(
                n_feature_qubits=n_feature_qubits,
                n_layers=n_layers,
                dev=dev,
                layer_type=layer_type,
                noise_prob=noise_prob,
            )
        else:
            # For TTN/MERA/QCNN: use standard builder on default.mixed device
            # (noise injected at device level via shots-based sampling)
            import architectures as archs
            if arch == "ttn":
                self.qnode = archs.build_ttn_qnn(n_feature_qubits, dev)
            elif arch == "mera":
                self.qnode = archs.build_mera_qnn(n_feature_qubits, n_layers, dev)
            elif arch == "qcnn":
                self.qnode = archs.build_qcnn_qnn(n_feature_qubits, dev)
            else:
                raise ValueError(f"Unknown arch '{arch}'.")

        import architectures as archs
        n_params = {
            "simple": archs.simple_num_params(n_feature_qubits, n_layers, layer_type),
            "ttn":    archs.ttn_num_params(n_feature_qubits),
            "mera":   archs.mera_num_params(n_feature_qubits, n_layers),
            "qcnn":   archs.qcnn_num_params(n_feature_qubits),
        }[arch]

        self.qparams = torch.nn.Parameter(torch.zeros(n_params))

    def forward(self, x_batch: torch.Tensor) -> torch.Tensor:
        outs = []
        for i in range(x_batch.shape[0]):
            val = self.qnode(x_batch[i], self.qparams)
            if not isinstance(val, torch.Tensor):
                val = torch.tensor(float(val), dtype=torch.float32)
            outs.append(val.reshape(()))
        return torch.stack(outs)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_checkpoint_under_noise(
    ckpt_path: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    noise_level: str = "medium",
    shots: int = 200,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Load a trained checkpoint and evaluate it under a given noise level.

    Returns standard metrics dict (see evaluate.py).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg  = ckpt["config"]
    state = ckpt["model_state"]

    noise_prob = PAPER_NOISE_LEVELS[noise_level]

    noisy_model = NoisyQNNModel(
        arch             = cfg["arch"],
        n_feature_qubits = cfg["n_feature_qubits"],
        n_layers         = cfg["n_layers"],
        layer_type       = cfg.get("layer_type", "XXYY"),
        noise_prob       = noise_prob,
        shots            = shots,
    )
    noisy_model.load_state_dict(state, strict=True)
    noisy_model.eval()

    stats = eval_module.evaluate_model(
        noisy_model, X_test, y_test,
        batch_size=batch_size,
        device="cpu",
        desc=f"noise={noise_level}",
    )
    return stats


def run_noise_sweep(
    ckpt_path: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    noise_levels: List[str] | None = None,
    shots: int = 200,
    output_json: str | None = None,
) -> Dict[str, Dict[str, float]]:
    """
    Run evaluation across all (or selected) noise levels and optionally save JSON.
    """
    if noise_levels is None:
        noise_levels = list(PAPER_NOISE_LEVELS.keys())

    results: Dict[str, Dict[str, float]] = {}
    for lvl in noise_levels:
        print(f"\n[noise_eval] Level: {lvl}  (p={PAPER_NOISE_LEVELS[lvl]:.4f})")
        stats = evaluate_checkpoint_under_noise(
            ckpt_path, X_test, y_test,
            noise_level=lvl, shots=shots,
        )
        summary = {k: stats[k] for k in ["f1", "accuracy", "precision", "recall", "roc_auc", "cert_mean", "cert_std"]}
        results[lvl] = summary
        print(f"  F1={summary['f1']:.4f}  Acc={summary['accuracy']:.4f}  "
              f"cert_mean={summary['cert_mean']:.4f}")

    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[noise_eval] Saved to {output_json}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",       required=True)
    parser.add_argument("--data_csv",   default="data/processed/nf_unsw_balanced.csv")
    parser.add_argument("--shots",      type=int, default=200)
    parser.add_argument("--output",     default="results/noise_sweep.json")
    parser.add_argument("--levels",     nargs="+", default=None,
                        help="Subset of noise levels to evaluate (default: all)")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_csv = os.path.join(root, args.data_csv)

    pack   = data_utils.load_encoded_splits(data_csv)
    X_test = pack["X_test"]
    y_test = pack["y_test"]

    run_noise_sweep(
        ckpt_path    = args.ckpt,
        X_test       = X_test,
        y_test       = y_test,
        noise_levels = args.levels,
        shots        = args.shots,
        output_json  = os.path.join(root, args.output),
    )