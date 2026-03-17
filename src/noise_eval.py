from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np
import pennylane as qml
import pennylane.noise as qnoise
import torch

import architectures as archs
import data_utils
import evaluate as eval_module
from encoding import apply_rx_encoding


PAPER_NOISE_LEVELS: Dict[str, float] = {
    "clean": 0.000,
    "very_low": 0.001,
    "low": 0.005,
    "medium": 0.010,
    "high": 0.050,
    "very_high": 0.100,
}


def _single_qubit_condition():
    return qml.noise.op_in([
        "PauliX",
        "Hadamard",
        "RX",
    ])


def _two_qubit_condition():
    return qml.noise.op_in([
        "IsingXX",
        "IsingYY",
        "IsingZZ",
    ])


def make_depolarizing_noise_model(
    p_single: float,
    p_two: float | None = None,
):
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


def build_simple_qnn_samples(
    n_feature_qubits: int,
    n_layers: int,
    dev,
    layer_type: str = "XXYY",
    interface: str = "torch",
):
    result_wire = n_feature_qubits
    terms = archs._parse_layer_type(layer_type)

    @qml.qnode(dev, interface=interface, diff_method="best")
    def qnode(x, params):
        if len(x) != n_feature_qubits:
            raise ValueError(
                f"Expected input with {n_feature_qubits} encoded features, got {len(x)}."
            )

        apply_rx_encoding(x)

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

        return qml.sample(qml.PauliX(result_wire))

    return qnode


class NoisySimpleQNNModel(torch.nn.Module):
    def __init__(
        self,
        n_feature_qubits: int,
        n_layers: int,
        layer_type: str,
        noise_prob_single: float,
        noise_prob_two: float | None = None,
        shots: int = 200,
        inference_mode: str = "shots",
    ) -> None:
        super().__init__()

        if inference_mode not in {"expval", "shots"}:
            raise ValueError("inference_mode must be 'expval' or 'shots'.")

        self.n_feature_qubits = int(n_feature_qubits)
        self.n_layers = int(n_layers)
        self.layer_type = str(layer_type)
        self.inference_mode = str(inference_mode)
        self.shots = int(shots)

        dev = qml.device(
            "default.mixed",
            wires=self.n_feature_qubits + 1,
            shots=None if self.inference_mode == "expval" else self.shots,
        )

        noise_model = make_depolarizing_noise_model(
            p_single=float(noise_prob_single),
            p_two=(
                float(noise_prob_two)
                if noise_prob_two is not None
                else float(noise_prob_single)
            ),
        )

        base_expval_qnode = archs.build_simple_qnn(
            n_feature_qubits=self.n_feature_qubits,
            n_layers=self.n_layers,
            dev=dev,
            layer_type=self.layer_type,
        )
        self.expval_qnode = qml.add_noise(base_expval_qnode, noise_model=noise_model)

        base_sample_qnode = build_simple_qnn_samples(
            n_feature_qubits=self.n_feature_qubits,
            n_layers=self.n_layers,
            dev=dev,
            layer_type=self.layer_type,
        )
        self.sample_qnode = qml.add_noise(base_sample_qnode, noise_model=noise_model)

        n_params = archs.simple_num_params(
            self.n_feature_qubits,
            self.n_layers,
            self.layer_type,
        )
        self.qparams = torch.nn.Parameter(torch.zeros(n_params, dtype=torch.float32))

    def forward(self, x_batch: torch.Tensor) -> torch.Tensor:
        if x_batch.ndim != 2:
            raise ValueError(
                f"x_batch must have shape (batch_size, n_feature_qubits), got {tuple(x_batch.shape)}."
            )

        if x_batch.shape[1] != self.n_feature_qubits:
            raise ValueError(
                f"Expected {self.n_feature_qubits} features per sample, got {x_batch.shape[1]}."
            )

        outputs = []
        for x in x_batch:
            if self.inference_mode == "expval":
                value = self.expval_qnode(x, self.qparams)
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor(float(value), dtype=torch.float32)
                outputs.append(value.reshape(()))
            else:
                samples = self.sample_qnode(x, self.qparams)
                if isinstance(samples, torch.Tensor):
                    samples_np = samples.detach().cpu().numpy().reshape(-1)
                else:
                    samples_np = np.asarray(samples, dtype=float).reshape(-1)

                certainty = float(np.clip(np.mean(samples_np), -1.0, 1.0))
                outputs.append(torch.tensor(certainty, dtype=torch.float32).reshape(()))

        return torch.stack(outputs, dim=0)


def _validate_noise_inputs(noise_level: str, two_qubit_scale: float) -> None:
    if noise_level not in PAPER_NOISE_LEVELS:
        raise ValueError(
            f"Unknown noise_level '{noise_level}'. Supported: {list(PAPER_NOISE_LEVELS.keys())}"
        )
    if two_qubit_scale < 0.0:
        raise ValueError("two_qubit_scale must be non-negative.")


def _attach_eval_metadata(
    stats: Dict[str, Any],
    p_single: float,
    p_two: float,
    shots: int,
    inference_mode: str,
) -> Dict[str, Any]:
    stats["noise_prob_single"] = float(p_single)
    stats["noise_prob_two"] = float(p_two)
    stats["shots"] = int(shots)
    stats["inference_mode"] = str(inference_mode)
    return stats


def evaluate_checkpoint_under_noise(
    ckpt_path: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    noise_level: str = "medium",
    shots: int = 200,
    batch_size: int = 32,
    two_qubit_scale: float = 1.0,
    inference_mode: str = "shots",
) -> Dict[str, Any]:
    _validate_noise_inputs(noise_level=noise_level, two_qubit_scale=two_qubit_scale)

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
        inference_mode=inference_mode,
    )
    noisy_model.load_state_dict(state, strict=True)
    noisy_model.eval()

    stats = eval_module.evaluate_model(
        noisy_model,
        X_test,
        y_test,
        batch_size=batch_size,
        device="cpu",
        desc=f"noise={noise_level}|mode={inference_mode}",
    )
    return _attach_eval_metadata(stats, p_single, p_two, shots, inference_mode)


def evaluate_weights_under_noise(
    weights_path: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_feature_qubits: int,
    n_layers: int,
    layer_type: str = "XXYY",
    noise_level: str = "medium",
    shots: int = 200,
    batch_size: int = 32,
    two_qubit_scale: float = 1.0,
    inference_mode: str = "shots",
) -> Dict[str, Any]:
    _validate_noise_inputs(noise_level=noise_level, two_qubit_scale=two_qubit_scale)

    p_single = PAPER_NOISE_LEVELS[noise_level]
    p_two = min(1.0, p_single * float(two_qubit_scale))

    weights = np.load(weights_path)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)

    noisy_model = NoisySimpleQNNModel(
        n_feature_qubits=int(n_feature_qubits),
        n_layers=int(n_layers),
        layer_type=str(layer_type),
        noise_prob_single=p_single,
        noise_prob_two=p_two,
        shots=int(shots),
        inference_mode=inference_mode,
    )

    expected = noisy_model.qparams.numel()
    if len(weights) != expected:
        raise ValueError(
            f"Weight length mismatch: file has {len(weights)} params, but model expects {expected}."
        )

    with torch.no_grad():
        noisy_model.qparams.copy_(torch.tensor(weights, dtype=torch.float32))

    noisy_model.eval()

    stats = eval_module.evaluate_model(
        noisy_model,
        X_test,
        y_test,
        batch_size=batch_size,
        device="cpu",
        desc=f"noise={noise_level}|mode={inference_mode}",
    )
    return _attach_eval_metadata(stats, p_single, p_two, shots, inference_mode)


def run_noise_sweep(
    X_test: np.ndarray,
    y_test: np.ndarray,
    ckpt_path: str | None = None,
    weights_path: str | None = None,
    arch: str = "simple",
    n_layers: int | None = None,
    layer_type: str = "XXYY",
    noise_levels: List[str] | None = None,
    shots: int = 200,
    batch_size: int = 32,
    two_qubit_scale: float = 1.0,
    inference_mode: str = "shots",
    output_json: str | None = None,
) -> Dict[str, Dict[str, float]]:
    if noise_levels is None:
        noise_levels = list(PAPER_NOISE_LEVELS.keys())

    if ckpt_path is None and weights_path is None:
        raise ValueError("You must provide either ckpt_path or weights_path.")
    if ckpt_path is not None and weights_path is not None:
        raise ValueError("Provide only one of ckpt_path or weights_path.")

    results: Dict[str, Dict[str, float]] = {}

    for lvl in noise_levels:
        print(
            f"\n[noise_eval] Level: {lvl} "
            f"(p1={PAPER_NOISE_LEVELS[lvl]:.4f}, "
            f"p2={min(1.0, PAPER_NOISE_LEVELS[lvl] * two_qubit_scale):.4f}, "
            f"mode={inference_mode})"
        )

        if ckpt_path is not None:
            stats = evaluate_checkpoint_under_noise(
                ckpt_path=ckpt_path,
                X_test=X_test,
                y_test=y_test,
                noise_level=lvl,
                shots=shots,
                batch_size=batch_size,
                two_qubit_scale=two_qubit_scale,
                inference_mode=inference_mode,
            )
        else:
            if arch != "simple":
                raise NotImplementedError(
                    "Raw-weights mode currently supports only the 'simple' architecture."
                )
            if n_layers is None:
                raise ValueError("--n-layers is required when using --weights.")

            stats = evaluate_weights_under_noise(
                weights_path=str(weights_path),
                X_test=X_test,
                y_test=y_test,
                n_feature_qubits=X_test.shape[1],
                n_layers=n_layers,
                layer_type=layer_type,
                noise_level=lvl,
                shots=shots,
                batch_size=batch_size,
                two_qubit_scale=two_qubit_scale,
                inference_mode=inference_mode,
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

    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Optional PyTorch checkpoint with config + model_state.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to .npy weights file, e.g. best_weights.npy.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="simple",
        choices=["simple"],
        help="Architecture name for raw-weights mode.",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=None,
        help="Number of variational layers for raw-weights mode.",
    )
    parser.add_argument(
        "--layer-type",
        type=str,
        default="XXYY",
        choices=["XXYY", "ZZXX", "ZZYY", "ZZXXYY"],
        help="Entangling layer type for raw-weights mode.",
    )

    parser.add_argument("--data_csv", default="data/processed/nf_unsw_balanced.csv")
    parser.add_argument("--shots", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--two_qubit_scale", type=float, default=1.0)
    parser.add_argument("--mode", choices=["expval", "shots"], default="shots")
    parser.add_argument("--output", default="results/noise_sweep.json")
    parser.add_argument(
        "--levels",
        nargs="+",
        default=None,
        help="Subset of noise levels to evaluate.",
    )

    args = parser.parse_args()

    if args.ckpt is None and args.weights is None:
        raise ValueError("Provide either --ckpt or --weights.")
    if args.ckpt is not None and args.weights is not None:
        raise ValueError("Provide only one of --ckpt or --weights.")

    pack = data_utils.load_encoded_splits(args.data_csv)
    X_test = pack["X_test"]
    y_test = pack["y_test"]

    run_noise_sweep(
        X_test=X_test,
        y_test=y_test,
        ckpt_path=args.ckpt,
        weights_path=args.weights,
        arch=args.arch,
        n_layers=args.n_layers,
        layer_type=args.layer_type,
        noise_levels=args.levels,
        shots=args.shots,
        batch_size=args.batch_size,
        two_qubit_scale=args.two_qubit_scale,
        inference_mode=args.mode,
        output_json=args.output,
    )
