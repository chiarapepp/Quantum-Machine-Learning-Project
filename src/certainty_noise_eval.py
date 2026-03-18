from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import pennylane as qml
import pennylane.noise as qnoise
from pennylane import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import architectures as archs
import dataset as ds_module
import encoding as enc_module
from encoding import apply_rx_encoding


LABEL_COLUMN = "Label"
FEATURE_COLUMNS = list(enc_module.FEATURE_COLUMNS)

PAPER_NOISE_LEVELS: Dict[str, float] = {
    "clean": 0.000,
    "very_low": 0.001,
    "low": 0.005,
    "medium": 0.010,
    "high": 0.050,
    "very_high": 0.100,
}


def load_encoded_splits_custom(
    processed_csv: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 1,
    n_bins: int = 100,
    encoder_fit_scope: str = "all",
) -> Dict[str, Any]:
    if not os.path.exists(processed_csv):
        raise FileNotFoundError(f"Processed CSV not found: {processed_csv}")

    df = pd.read_csv(processed_csv)
    X_all = df[FEATURE_COLUMNS].copy()
    y_all = df[LABEL_COLUMN].to_numpy(dtype=int)

    X_trainval_df, X_test_df, y_trainval, y_test = train_test_split(
        X_all,
        y_all,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=y_all,
    )

    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_trainval_df,
        y_trainval,
        test_size=val_size,
        random_state=random_state,
        shuffle=True,
        stratify=y_trainval,
    )

    X_train_df = X_train_df.reset_index(drop=True)
    X_val_df = X_val_df.reset_index(drop=True)
    X_test_df = X_test_df.reset_index(drop=True)

    y_train = np.asarray(y_train, dtype=int)
    y_val = np.asarray(y_val, dtype=int)
    y_test = np.asarray(y_test, dtype=int)

    encoder = enc_module.QuantumEncoder(n_bins=n_bins)
    if encoder_fit_scope == "train":
        encoder.fit(X_train_df)
    elif encoder_fit_scope == "all":
        encoder.fit(
            pd.concat([X_train_df, X_val_df, X_test_df], axis=0).reset_index(drop=True)
        )
    else:
        raise ValueError("encoder_fit_scope must be either 'train' or 'all'.")

    X_train = encoder.transform(X_train_df)
    X_val = encoder.transform(X_val_df)
    X_test = encoder.transform(X_test_df)

    print(
        f"[certainty_noise] Encoded splits -> "
        f"train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}"
    )
    print(
        f"[certainty_noise] Label balance -> "
        f"train: {np.bincount(y_train)}, val: {np.bincount(y_val)}, test: {np.bincount(y_test)}"
    )
    print(f"[certainty_noise] Encoder fit scope: {encoder_fit_scope}")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "encoder": encoder,
    }


def _single_qubit_condition():
    return qml.noise.op_in(["PauliX", "Hadamard", "RX"])


def _two_qubit_condition():
    return qml.noise.op_in(["IsingXX", "IsingYY", "IsingZZ"])


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

                value = float(np.clip(np.mean(samples_np), -1.0, 1.0))
                outputs.append(torch.tensor(value, dtype=torch.float32).reshape(()))

        return torch.stack(outputs, dim=0)


def predict_labels(raw_outputs: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    raw_outputs = np.asarray(raw_outputs, dtype=float)
    return (raw_outputs >= threshold).astype(int)


def certainty_factor_from_output(raw_outputs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    raw_outputs = np.clip(np.asarray(raw_outputs, dtype=float), -1.0, 1.0)
    y_true = np.asarray(y_true, dtype=int)
    y_pm = 2 * y_true - 1
    return y_pm * raw_outputs


def confidence_from_certainty(certainty_factor: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(certainty_factor, dtype=float))


def certainty_stats(
    certainty_factor: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    certainty_factor = np.asarray(certainty_factor, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    conf = np.abs(certainty_factor)
    correct = y_true == y_pred
    incorrect = ~correct

    stats = {
        "all_mean": float(np.mean(certainty_factor)),
        "all_std": float(np.std(certainty_factor)),
        "all_conf_mean": float(np.mean(conf)),
        "all_conf_std": float(np.std(conf)),
        "correct_mean": float(np.mean(certainty_factor[correct])) if np.any(correct) else float("nan"),
        "correct_std": float(np.std(certainty_factor[correct])) if np.any(correct) else float("nan"),
        "incorrect_mean": float(np.mean(certainty_factor[incorrect])) if np.any(incorrect) else float("nan"),
        "incorrect_std": float(np.std(certainty_factor[incorrect])) if np.any(incorrect) else float("nan"),
        "correct_conf_mean": float(np.mean(conf[correct])) if np.any(correct) else float("nan"),
        "incorrect_conf_mean": float(np.mean(conf[incorrect])) if np.any(incorrect) else float("nan"),
        "frac_above_zero": float(np.mean(certainty_factor > 0.0)),
        "frac_below_zero": float(np.mean(certainty_factor < 0.0)),
        "frac_abs_lt_0.1": float(np.mean(conf < 0.1)),
        "frac_abs_lt_0.2": float(np.mean(conf < 0.2)),
        "frac_abs_lt_0.5": float(np.mean(conf < 0.5)),
    }
    return stats


def evaluate_certainty_under_noise(
    model: NoisySimpleQNNModel,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.0,
    batch_size: int = 32,
) -> Dict[str, Any]:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)

    if len(X) == 0:
        raise ValueError("Empty dataset passed to evaluate_certainty_under_noise().")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length.")

    outputs = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = X[start : start + batch_size]
            xb_t = torch.tensor(xb, dtype=torch.float32)
            out = model(xb_t).detach().cpu().numpy().reshape(-1)
            outputs.append(out)

    raw_outputs = np.concatenate(outputs, axis=0).astype(float)
    raw_outputs = np.clip(raw_outputs, -1.0, 1.0)

    preds = predict_labels(raw_outputs, threshold=threshold)
    certainty_factor = certainty_factor_from_output(raw_outputs, y)
    confidence = confidence_from_certainty(certainty_factor)

    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, pos_label=1, zero_division=0)
    rec = recall_score(y, preds, pos_label=1, zero_division=0)
    f1 = f1_score(y, preds, pos_label=1, zero_division=0)

    try:
        malicious_score = (raw_outputs + 1.0) / 2.0
        roc = roc_auc_score(y, malicious_score)
    except Exception:
        roc = float("nan")

    stats = certainty_stats(certainty_factor, y, preds)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc),
        "raw_outputs": raw_outputs,
        "certainty_factor": certainty_factor,
        "confidence": confidence,
        "y_true": y,
        "preds": preds,
        "correct": (preds == y).astype(int),
        "cf_mean": float(np.mean(certainty_factor)),
        "cf_std": float(np.std(certainty_factor)),
        "conf_mean": float(np.mean(confidence)),
        "conf_std": float(np.std(confidence)),
        "certainty_stats": stats,
    }


def make_samples_dataframe(
    result: Dict[str, Any],
    label: str,
    arch: str,
    split: str,
    noise_level: str,
    inference_mode: str,
    shots: int,
    p_single: float,
    p_two: float,
) -> pd.DataFrame:
    y_true = np.asarray(result["y_true"], dtype=int)
    preds = np.asarray(result["preds"], dtype=int)
    raw_outputs = np.asarray(result["raw_outputs"], dtype=float)
    cf = np.asarray(result["certainty_factor"], dtype=float)
    conf = np.asarray(result["confidence"], dtype=float)

    return pd.DataFrame(
        {
            "sample_idx": np.arange(len(y_true)),
            "y_true": y_true,
            "y_pred": preds,
            "raw_output": raw_outputs,
            "certainty_factor": cf,
            "confidence": conf,
            "correct": (y_true == preds).astype(int),
            "label": label,
            "arch": arch,
            "split": split,
            "noise_level": noise_level,
            "inference_mode": inference_mode,
            "shots": shots,
            "noise_prob_single": p_single,
            "noise_prob_two": p_two,
        }
    )


def save_violin_plot(df: pd.DataFrame, save_path: Path, title: str):
    labels = list(df["label"].dropna().unique())
    data = [
        df.loc[df["label"] == lbl, "certainty_factor"].to_numpy(dtype=float)
        for lbl in labels
    ]

    plt.figure(figsize=(8, 6))
    plt.violinplot(
        data,
        positions=range(1, len(labels) + 1),
        showmeans=False,
        showmedians=True,
        showextrema=True,
    )
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.axhline(0.0, linestyle="--", linewidth=1.2)
    plt.ylim(-1.05, 1.05)
    plt.xlabel("")
    plt.ylabel("Certainty factor")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_histogram_plot(df: pd.DataFrame, save_path: Path, title: str):
    correct = df.loc[df["correct"] == 1, "certainty_factor"].to_numpy(dtype=float)
    wrong = df.loc[df["correct"] == 0, "certainty_factor"].to_numpy(dtype=float)

    plt.figure(figsize=(8, 6))
    plt.hist(correct, bins=30, alpha=0.7, label="correct")
    plt.hist(wrong, bins=30, alpha=0.7, label="wrong")
    plt.axvline(0.0, linestyle="--", linewidth=1.2)
    plt.xlim(-1.05, 1.05)
    plt.xlabel("Certainty factor")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_summary_json(
    result: Dict[str, Any],
    args,
    save_path: Path,
    p_single: float,
    p_two: float,
):
    summary = {
        "arch": args.arch,
        "split": args.split,
        "label": args.label,
        "weights_path": args.weights_path,
        "threshold": args.threshold,
        "processed_csv": args.processed_csv,
        "test_size": args.test_size,
        "val_size": args.val_size,
        "random_state": args.random_state,
        "n_bins": args.n_bins,
        "encoder_fit_scope": args.encoder_fit_scope,
        "noise_level": args.noise_level,
        "noise_prob_single": p_single,
        "noise_prob_two": p_two,
        "two_qubit_scale": args.two_qubit_scale,
        "inference_mode": args.mode,
        "shots": args.shots,
        "metrics": {
            "f1": result["f1"],
            "accuracy": result["accuracy"],
            "precision": result["precision"],
            "recall": result["recall"],
            "roc_auc": result["roc_auc"],
            "cf_mean": result["cf_mean"],
            "cf_std": result["cf_std"],
            "conf_mean": result["conf_mean"],
            "conf_std": result["conf_std"],
        },
        "certainty_stats": result["certainty_stats"],
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Certainty factor evaluation under synthetic depolarizing noise"
    )

    parser.add_argument("--arch", type=str, default="simple", choices=["simple"])
    parser.add_argument("--weights-path", type=str, required=True, help="Path to .npy weights file")

    parser.add_argument("--processed-csv", type=str, default="data/processed/nf_unsw_balanced.csv")
    parser.add_argument("--raw-csv", type=str, default="data/raw/NF-UNSW-NB15-v2.csv")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=1)
    parser.add_argument("--n-bins", type=int, default=100)
    parser.add_argument(
        "--encoder-fit-scope",
        type=str,
        default="all",
        choices=["all", "train"],
        help="Use 'all' to reproduce your current runs; use 'train' for stricter encoding.",
    )

    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--label", type=str, default=None)

    parser.add_argument("--n-layers", type=int, required=True)
    parser.add_argument(
        "--layer-type",
        type=str,
        default="XXYY",
        choices=["XXYY", "ZZXX", "ZZYY", "ZZXXYY"],
    )

    parser.add_argument(
        "--noise-level",
        type=str,
        default="medium",
        choices=list(PAPER_NOISE_LEVELS.keys()),
    )
    parser.add_argument("--shots", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--two-qubit-scale", type=float, default=1.0)
    parser.add_argument("--mode", type=str, default="shots", choices=["expval", "shots"])

    parser.add_argument("--save-dir", type=str, default="outputs/certainty_noise")
    parser.add_argument("--save-prefix", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    processed_csv = args.processed_csv
    if not os.path.exists(processed_csv):
        print("[certainty_noise] processed CSV not found, generating it from raw CSV...")
        os.makedirs(os.path.dirname(processed_csv) or ".", exist_ok=True)
        ds_module.build_processed_nf_unsw(
            csv_path=args.raw_csv,
            save_processed_csv=processed_csv,
        )

    if not os.path.exists(args.weights_path):
        raise FileNotFoundError(f"Weights file not found: {args.weights_path}")

    if args.two_qubit_scale < 0.0:
        raise ValueError("--two-qubit-scale must be non-negative.")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    pack = load_encoded_splits_custom(
        processed_csv=processed_csv,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        n_bins=args.n_bins,
        encoder_fit_scope=args.encoder_fit_scope,
    )

    split_key_x = f"X_{args.split}"
    split_key_y = f"y_{args.split}"
    X = np.asarray(pack[split_key_x], dtype=float)
    y = np.asarray(pack[split_key_y], dtype=int)
    n_feature_qubits = X.shape[1]

    p_single = PAPER_NOISE_LEVELS[args.noise_level]
    p_two = min(1.0, p_single * float(args.two_qubit_scale))

    model = NoisySimpleQNNModel(
        n_feature_qubits=n_feature_qubits,
        n_layers=args.n_layers,
        layer_type=args.layer_type,
        noise_prob_single=p_single,
        noise_prob_two=p_two,
        shots=int(args.shots),
        inference_mode=args.mode,
    )

    weights = np.load(args.weights_path)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)

    expected = model.qparams.numel()
    if len(weights) != expected:
        raise ValueError(
            f"Weight length mismatch: file has {len(weights)} params, "
            f"but model expects {expected}."
        )

    with torch.no_grad():
        model.qparams.copy_(torch.tensor(weights, dtype=torch.float32))

    result = evaluate_certainty_under_noise(
        model=model,
        X=X,
        y=y,
        threshold=args.threshold,
        batch_size=args.batch_size,
    )

    label = args.label or args.noise_level
    default_prefix = (
        f"{args.arch}_{args.n_layers}layers_{args.layer_type}_{args.noise_level}_{args.mode}"
    )
    prefix = args.save_prefix or default_prefix

    samples_df = make_samples_dataframe(
        result=result,
        label=label,
        arch=args.arch,
        split=args.split,
        noise_level=args.noise_level,
        inference_mode=args.mode,
        shots=int(args.shots),
        p_single=float(p_single),
        p_two=float(p_two),
    )

    csv_path = save_dir / f"{prefix}_samples.csv"
    json_path = save_dir / f"{prefix}_summary.json"
    violin_path = save_dir / f"{prefix}_violin.png"
    hist_path = save_dir / f"{prefix}_hist.png"

    samples_df.to_csv(csv_path, index=False)
    save_summary_json(
        result,
        args=args,
        save_path=json_path,
        p_single=float(p_single),
        p_two=float(p_two),
    )
    save_violin_plot(
        samples_df,
        save_path=violin_path,
        title=(
            f"Certainty factor distribution - "
            f"{args.arch} ({args.noise_level}, {args.split}, {args.mode})"
        ),
    )
    save_histogram_plot(
        samples_df,
        save_path=hist_path,
        title=(
            f"Certainty factor histogram - "
            f"{args.arch} ({args.noise_level}, {args.split}, {args.mode})"
        ),
    )

    print(
        f"[certainty_noise] {args.split} | "
        f"noise={args.noise_level} | mode={args.mode} | "
        f"F1={result['f1']:.4f} | "
        f"Acc={result['accuracy']:.4f} | "
        f"AUC={result['roc_auc']:.4f} | "
        f"CF mean={result['cf_mean']:.4f} | "
        f"Conf mean={result['conf_mean']:.4f}"
    )
    print(f"[certainty_noise] saved samples CSV to:   {csv_path}")
    print(f"[certainty_noise] saved summary JSON to:  {json_path}")
    print(f"[certainty_noise] saved violin plot to:   {violin_path}")
    print(f"[certainty_noise] saved histogram to:     {hist_path}")


if __name__ == "__main__":
    main()