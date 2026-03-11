from __future__ import annotations

import os
import json
import argparse
from typing import Any, Dict, List

import numpy as np
import pennylane as qml
import pennylane.noise as qnoise
import torch

import evaluate as eval_module
import data_utils


PAPER_NOISE_LEVELS: Dict[str, float] = {
    "clean": 0.000,
    "very_low": 0.001,
    "low": 0.005,
    "medium": 0.010,
    "high": 0.050,
    "very_high": 0.100,
}


def _parse_layer_type(layer_type: str) -> List[str]:
    """
    Normalize the Simple architecture layer type.

    Supported aliases:
    - XY  -> XXYY
    - ZX  -> ZZXX
    - ZY  -> ZZYY
    - ZXY -> ZZXXYY
    """
    lt = layer_type.upper().replace("_", "").replace("-", "")

    aliases = {
        "XY": "XXYY",
        "ZX": "ZZXX",
        "ZY": "ZZYY",
        "ZXY": "ZZXXYY",
    }
    lt = aliases.get(lt, lt)

    mapping = {
        "ZZXXYY": ["ZZ", "XX", "YY"],
        "ZZXX": ["ZZ", "XX"],
        "XXYY": ["XX", "YY"],
        "ZZYY": ["ZZ", "YY"],
    }

    if lt not in mapping:
        raise ValueError(
            f"Unknown layer_type '{layer_type}'. "
            "Supported: XY/XXYY, ZX/ZZXX, ZY/ZZYY, ZXY/ZZXXYY."
        )

    return mapping[lt]


def simple_num_params(
    n_feature_qubits: int,
    n_layers: int,
    layer_type: str,
) -> int:
    n_terms = len(_parse_layer_type(layer_type))
    return n_feature_qubits * n_layers * n_terms


def _single_qubit_condition():
    return qml.noise.op_in(
        [
            "PauliX",
            "Hadamard",
            "RX",
        ]
    )


def _two_qubit_condition():
    return qml.noise.op_in(
        [
            "IsingXX",
            "IsingYY",
            "IsingZZ",
        ]
    )


def make_depolarizing_noise_model(
    p_single: float,
    p_two: float | None = None,
):
    """
    Build a PennyLane noise model in the style shown in the notebook:
    a rule-based model attached externally via qml.add_noise().

    For this reproduction we keep the model deliberately simple:
    - depolarizing after 1-qubit gates
    - depolarizing after 2-qubit gates

    If p_two is None, the same probability is used for both cases.
    """
    if p_two is None:
        p_two = p_single

    if not (0.0 <= p_single <= 1.0):
        raise ValueError("p_single must be in [0, 1].")
    if not (0.0 <= p_two <= 1.0):
        raise ValueError("p_two must be in [0, 1].")

    single_qubit_cond = _single_qubit_condition()
    two_qubit_cond = _two_qubit_condition()

    def single_qubit_noise(op, **kwargs):
        return qml.DepolarizingChannel(p_single, wires=op.wires[0])

    def two_qubit_noise(op, **kwargs):
        return [
            qml.DepolarizingChannel(p_two, wires=op.wires[0]),
            qml.DepolarizingChannel(p_two, wires=op.wires[1]),
        ]

    return qnoise.NoiseModel(
        {
            single_qubit_cond: single_qubit_noise,
            two_qubit_cond: two_qubit_noise,
        }
    )


def build_simple_qnn_base(
    n_feature_qubits: int,
    n_layers: int,
    dev,
    layer_type: str = "XXYY",
    interface: str = "torch",
):
    """
    Base noiseless Simple circuit.

    Noise is attached externally with qml.add_noise(), not written directly
    into the circuit body.
    """
    result_wire = n_feature_qubits
    terms = _parse_layer_type(layer_type)

    @qml.qnode(dev, interface=interface, diff_method="best")
    def qnode(x, params):
        if len(x) != n_feature_qubits:
            raise ValueError(
                f"Expected input with {n_feature_qubits} encoded features, got {len(x)}."
            )

        for wire in range(n_feature_qubits):
            qml.RX(x[wire], wires=wire)

        # |-> = X then H on |0>
        qml.PauliX(wires=result_wire)
        qml.Hadamard(wires=result_wire)

        idx = 0
        for _ in range(n_layers):
            for term in terms:
                for feature_wire in range(n_feature_qubits):
                    theta = params[idx]

                    if term == "XX":
                        qml.IsingXX(theta, wires=[result_wire, feature_wire])
                    elif term == "YY":
                        qml.IsingYY(theta, wires=[result_wire, feature_wire])
                    elif term == "ZZ":
                        qml.IsingZZ(theta, wires=[result_wire, feature_wire])
                    else:
                        raise RuntimeError(f"Unexpected term '{term}'.")

                    idx += 1

        return qml.expval(qml.PauliX(result_wire))

    return qnode


class NoisySimpleQNNModel(torch.nn.Module):
    """
    Torch wrapper for noisy inference with the Simple architecture.

    The circuit itself is defined noiselessly, then transformed with
    qml.add_noise(noise_model=...).
    """

    def __init__(
        self,
        n_feature_qubits: int,
        n_layers: int,
        layer_type: str,
        noise_prob_single: float,
        noise_prob_two: float | None = None,
        shots: int = 200,
    ) -> None:
        super().__init__()

        self.n_feature_qubits = int(n_feature_qubits)
        self.n_layers = int(n_layers)
        self.layer_type = str(layer_type)

        dev = qml.device(
            "default.mixed",
            wires=self.n_feature_qubits + 1,
            shots=shots,
        )

        base_qnode = build_simple_qnn_base(
            n_feature_qubits=self.n_feature_qubits,
            n_layers=self.n_layers,
            dev=dev,
            layer_type=self.layer_type,
        )

        noise_model = make_depolarizing_noise_model(
            p_single=float(noise_prob_single),
            p_two=(
                float(noise_prob_two)
                if noise_prob_two is not None
                else float(noise_prob_single)
            ),
        )

        self.qnode = qml.add_noise(base_qnode, noise_model=noise_model)

        n_params = simple_num_params(
            self.n_feature_qubits,
            self.n_layers,
            self.layer_type,
        )
        self.qparams = torch.nn.Parameter(torch.zeros(n_params, dtype=torch.float32))

    def forward(self, x_batch: torch.Tensor) -> torch.Tensor:
        if x_batch.ndim != 2:
            raise ValueError(
                f"x_batch must have shape (batch_size, n_feature_qubits), "
                f"got {tuple(x_batch.shape)}."
            )

        if x_batch.shape[1] != self.n_feature_qubits:
            raise ValueError(
                f"Expected {self.n_feature_qubits} features per sample, "
                f"got {x_batch.shape[1]}."
            )

        outputs = []
        for x in x_batch:
            value = self.qnode(x, self.qparams)
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(float(value), dtype=torch.float32)
            outputs.append(value.reshape(()))

        return torch.stack(outputs, dim=0)


def evaluate_checkpoint_under_noise(
    ckpt_path: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    noise_level: str = "medium",
    shots: int = 200,
    batch_size: int = 32,
    two_qubit_scale: float = 1.0,
) -> Dict[str, Any]:
    """
    Load a trained checkpoint and evaluate it under a selected noise level.

    This implementation supports only the Simple architecture, in line with
    the paper's noisy-evaluation focus.

    Parameters
    ----------
    noise_level:
        One of PAPER_NOISE_LEVELS keys.
    two_qubit_scale:
        Optional multiplier for 2-qubit depolarizing probability.
        Example: 2.0 means p_two = 2 * p_single.
        Default 1.0 keeps the model simple and symmetric.
    """
    if noise_level not in PAPER_NOISE_LEVELS:
        raise ValueError(
            f"Unknown noise_level '{noise_level}'. "
            f"Supported: {list(PAPER_NOISE_LEVELS.keys())}"
        )

    if two_qubit_scale < 0.0:
        raise ValueError("two_qubit_scale must be non-negative.")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    state = ckpt["model_state"]

    arch = str(cfg["arch"]).strip().lower()
    if arch != "simple":
        raise NotImplementedError(
            "This noise_eval.py currently supports only the 'simple' architecture."
        )

    p_single = PAPER_NOISE_LEVELS[noise_level]
    p_two = min(1.0, p_single * float(two_qubit_scale))

    noisy_model = NoisySimpleQNNModel(
        n_feature_qubits=int(cfg["n_feature_qubits"]),
        n_layers=int(cfg["n_layers"]),
        layer_type=str(cfg.get("layer_type", "XXYY")),
        noise_prob_single=p_single,
        noise_prob_two=p_two,
        shots=int(shots),
    )
    noisy_model.load_state_dict(state, strict=True)
    noisy_model.eval()

    stats = eval_module.evaluate_model(
        noisy_model,
        X_test,
        y_test,
        batch_size=batch_size,
        device="cpu",
        desc=f"noise={noise_level}",
    )

    stats["noise_prob_single"] = float(p_single)
    stats["noise_prob_two"] = float(p_two)
    stats["shots"] = int(shots)
    return stats


def run_noise_sweep(
    ckpt_path: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    noise_levels: List[str] | None = None,
    shots: int = 200,
    batch_size: int = 32,
    two_qubit_scale: float = 1.0,
    output_json: str | None = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate one checkpoint across multiple paper noise levels.
    """
    if noise_levels is None:
        noise_levels = list(PAPER_NOISE_LEVELS.keys())

    results: Dict[str, Dict[str, float]] = {}

    for lvl in noise_levels:
        print(
            f"\n[noise_eval] Level: {lvl} "
            f"(p1={PAPER_NOISE_LEVELS[lvl]:.4f}, "
            f"p2={min(1.0, PAPER_NOISE_LEVELS[lvl] * two_qubit_scale):.4f})"
        )

        stats = evaluate_checkpoint_under_noise(
            ckpt_path=ckpt_path,
            X_test=X_test,
            y_test=y_test,
            noise_level=lvl,
            shots=shots,
            batch_size=batch_size,
            two_qubit_scale=two_qubit_scale,
        )

        summary = {
            "f1": float(stats["f1"]),
            "accuracy": float(stats["accuracy"]),
            "precision": float(stats["precision"]),
            "recall": float(stats["recall"]),
            "roc_auc": float(stats["roc_auc"]),
            "cert_mean": float(stats["cert_mean"]),
            "cert_std": float(stats["cert_std"]),
            "conf_mean": float(stats["conf_mean"]),
            "conf_std": float(stats["conf_std"]),
            "noise_prob_single": float(stats["noise_prob_single"]),
            "noise_prob_two": float(stats["noise_prob_two"]),
            "shots": float(stats["shots"]),
        }
        results[lvl] = summary

        print(
            f"  F1={summary['f1']:.4f} | "
            f"Acc={summary['accuracy']:.4f} | "
            f"CertMean={summary['cert_mean']:.4f}"
        )

    if output_json is not None:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[noise_eval] Saved results to: {output_json}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_csv", default="data/processed/nf_unsw_balanced.csv")
    parser.add_argument("--shots", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--two_qubit_scale", type=float, default=1.0)
    parser.add_argument("--output", default="results/noise_sweep.json")
    parser.add_argument(
        "--levels",
        nargs="+",
        default=None,
        help="Subset of noise levels to evaluate.",
    )

    args = parser.parse_args()

    pack = data_utils.load_encoded_splits(args.data_csv)
    X_test = pack["X_test"]
    y_test = pack["y_test"]

    run_noise_sweep(
        ckpt_path=args.ckpt,
        X_test=X_test,
        y_test=y_test,
        noise_levels=args.levels,
        shots=args.shots,
        batch_size=args.batch_size,
        two_qubit_scale=args.two_qubit_scale,
        output_json=args.output,
    )