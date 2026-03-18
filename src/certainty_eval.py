"""
Run post-training certainty-factor evaluation for trained QNN weight files.

This script rebuilds encoded data splits, loads architecture-specific QNodes,
computes prediction and certainty metrics on a selected split, and saves per-sample
outputs, summary JSON metrics, and diagnostic plots.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict
import matplotlib.pyplot as plt
import pandas as pd
import pennylane as qml
from pennylane import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import data_utils
import dataset as ds_module
from architectures import (
    build_mera_qnn,
    build_qcnn_qnn,
    build_simple_qnn,
    build_ttn_qnn,
)


def build_qnode_from_args(args, n_feature_qubits: int):
    arch = args.arch.lower()

    if arch == "simple":
        dev = qml.device("lightning.qubit", wires=n_feature_qubits + 1)
        qnode = build_simple_qnn(
            n_feature_qubits=n_feature_qubits,
            n_layers=args.n_layers,
            dev=dev,
            layer_type=args.layer_type,
            interface="autograd",
        )
        return qnode

    if arch == "ttn":
        dev = qml.device("lightning.qubit", wires=n_feature_qubits)
        qnode = build_ttn_qnn(
            n_qubits=n_feature_qubits,
            dev=dev,
            interface="autograd",
        )
        return qnode

    if arch == "mera":
        dev = qml.device("lightning.qubit", wires=n_feature_qubits)
        qnode = build_mera_qnn(
            n_qubits=n_feature_qubits,
            dev=dev,
            interface="autograd",
        )
        return qnode

    if arch == "qcnn":
        dev = qml.device("lightning.qubit", wires=n_feature_qubits)
        qnode = build_qcnn_qnn(
            n_qubits=n_feature_qubits,
            dev=dev,
            interface="autograd",
        )
        return qnode

    raise ValueError(f"Unknown architecture '{args.arch}'.")


def predict_outputs(qnode, X: np.ndarray, params: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return np.asarray([qnode(x, params) for x in X], dtype=float).reshape(-1)


def predict_labels(raw_outputs: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    raw_outputs = np.asarray(raw_outputs, dtype=float)
    return (raw_outputs >= threshold).astype(int)


def certainty_factor_from_output(raw_outputs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Signed certainty for binary labels.

    raw_output is expected in [-1, 1] where:
      raw_output >= 0 -> malicious (1)
      raw_output <  0 -> benign (0)

    We multiply by the true label in {-1, +1} so that:
      > 0 : correct prediction
      < 0 : wrong prediction
      ~ 0 : uncertain prediction
    """
    raw_outputs = np.clip(np.asarray(raw_outputs, dtype=float), -1.0, 1.0)
    y_true = np.asarray(y_true, dtype=int)
    y_pm = 2 * y_true - 1
    # definition of certainty factor: signed distance from the decision boundary, coherent with training and prediction
    return y_pm * raw_outputs


def confidence_from_certainty(certainty_factor: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(certainty_factor, dtype=float))


def certainty_stats(certainty_factor: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
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
        # measure how many predictions are close to the decision boundary (i.e. low confidence) highly uncertain 
        "frac_abs_lt_0.1": float(np.mean(conf < 0.1)),
        "frac_abs_lt_0.2": float(np.mean(conf < 0.2)),
        "frac_abs_lt_0.5": float(np.mean(conf < 0.5)),
    }
    return stats


def evaluate_certainty(
    qnode,
    X: np.ndarray,
    y: np.ndarray,
    params: np.ndarray,
    threshold: float = 0.0,
) -> Dict[str, Any]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    params = np.asarray(params, dtype=float)

    if len(X) == 0:
        raise ValueError("Empty dataset passed to evaluate_certainty().")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length.")

    raw_outputs = predict_outputs(qnode, X, params)
    preds = predict_labels(raw_outputs, threshold=threshold)
    certainty_factor = certainty_factor_from_output(raw_outputs, y)
    confidence = confidence_from_certainty(certainty_factor)

    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, pos_label=1, zero_division=0)
    rec = recall_score(y, preds, pos_label=1, zero_division=0)
    f1 = f1_score(y, preds, pos_label=1, zero_division=0)

    try:
        malicious_score = (np.clip(raw_outputs, -1.0, 1.0) + 1.0) / 2.0
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


def make_samples_dataframe(result: Dict[str, Any], label: str, arch: str, split: str) -> pd.DataFrame:
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


def save_summary_json(result: Dict[str, Any], args, save_path: Path):
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
        description="Post-training certainty factor evaluation for QNN checkpoints"
    )

    parser.add_argument("--arch", type=str, required=True, choices=["simple", "ttn", "mera", "qcnn"])
    parser.add_argument("--weights-path", type=str, required=True, help="Path to .npy weights file")
    parser.add_argument("--processed-csv", type=str, default="data/processed/nf_unsw_balanced.csv")
    parser.add_argument("--raw-csv", type=str, default="data/raw/NF-UNSW-NB15-v2.csv")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=1)
    parser.add_argument("--n-bins", type=int, default=100)

    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--label", type=str, default=None, help="Label shown in plots/CSV, e.g. clean, medium, aria1")

    parser.add_argument("--n-layers", type=int, default=2, help="Only used for simple architecture")
    parser.add_argument(
        "--layer-type",
        type=str,
        default="XXYY",
        choices=["XXYY", "ZZXX", "ZZYY", "ZZXXYY"],
        help="Only used for simple architecture",
    )

    parser.add_argument("--save-dir", type=str, default="outputs/certainty")
    parser.add_argument("--save-prefix", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    processed_csv = args.processed_csv
    if not os.path.exists(processed_csv):
        print("[certainty] processed CSV not found, generating it from raw CSV...")
        os.makedirs(os.path.dirname(processed_csv) or ".", exist_ok=True)
        ds_module.build_processed_nf_unsw(csv_path=args.raw_csv, save_processed_csv=processed_csv)

    if not os.path.exists(args.weights_path):
        raise FileNotFoundError(f"Weights file not found: {args.weights_path}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    pack = data_utils.load_encoded_splits(
        processed_csv=processed_csv,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        n_bins=args.n_bins,
    )

    split_key_x = f"X_{args.split}"
    split_key_y = f"y_{args.split}"
    X = np.asarray(pack[split_key_x], dtype=float)
    y = np.asarray(pack[split_key_y], dtype=int)
    n_feature_qubits = X.shape[1]

    qnode = build_qnode_from_args(args, n_feature_qubits=n_feature_qubits)
    params = np.asarray(np.load(args.weights_path), dtype=float).reshape(-1)

    result = evaluate_certainty(
        qnode=qnode,
        X=X,
        y=y,
        params=params,
        threshold=args.threshold,
    )

    label = args.label or args.split
    default_prefix = f"{args.arch}_{label}"
    if args.arch == "simple":
        default_prefix = f"{args.arch}_{args.n_layers}layers_{args.layer_type}_{label}"
    prefix = args.save_prefix or default_prefix

    samples_df = make_samples_dataframe(result, label=label, arch=args.arch, split=args.split)
    csv_path = save_dir / f"{prefix}_samples.csv"
    json_path = save_dir / f"{prefix}_summary.json"
    violin_path = save_dir / f"{prefix}_violin.png"
    hist_path = save_dir / f"{prefix}_hist.png"

    samples_df.to_csv(csv_path, index=False)
    save_summary_json(result, args=args, save_path=json_path)
    save_violin_plot(
        samples_df,
        save_path=violin_path,
        title=f"Certainty factor distribution - {args.arch} ({label}, {args.split})",
    )
    save_histogram_plot(
        samples_df,
        save_path=hist_path,
        title=f"Certainty factor histogram - {args.arch} ({label}, {args.split})",
    )

    print(
        f"[certainty] {args.split} | F1={result['f1']:.4f} | "
        f"Acc={result['accuracy']:.4f} | AUC={result['roc_auc']:.4f} | "
        f"CF mean={result['cf_mean']:.4f} | Conf mean={result['conf_mean']:.4f}"
    )
    print(f"[certainty] saved samples CSV to:   {csv_path}")
    print(f"[certainty] saved summary JSON to:  {json_path}")
    print(f"[certainty] saved violin plot to:   {violin_path}")
    print(f"[certainty] saved histogram to:     {hist_path}")


if __name__ == "__main__":
    main()