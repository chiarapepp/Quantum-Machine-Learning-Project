"""
Certainty-factor evaluation under synthetic depolarizing noise.

This script loads encoded splits, rebuilds a noisy Simple QNN from saved
weights, computes classification and certainty metrics on a selected split, and
saves per-sample outputs, summary JSON, and diagnostic plots.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import data_utils
import dataset as ds_module
from certainty_eval import (
    certainty_factor_from_output,
    confidence_from_certainty,
    certainty_stats,
    make_samples_dataframe,
    save_violin_plot,
    save_histogram_plot,
)
from noise_eval import (
    PAPER_NOISE_LEVELS,
    NoisySimpleQNNModel,
)


def evaluate_model_certainty(
    model: NoisySimpleQNNModel,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.0,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Evaluate a noisy PyTorch QNN model and compute paper-aligned certainty-factor statistics.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)

    if len(X) == 0:
        raise ValueError("Empty dataset passed to evaluate_model_certainty().")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length.")

    outputs = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.tensor(X[start:start + batch_size], dtype=torch.float32)
            out = model(xb).detach().cpu().numpy().reshape(-1)
            outputs.append(out)

    raw_outputs = np.concatenate(outputs, axis=0).astype(float)
    raw_outputs = np.clip(raw_outputs, -1.0, 1.0)

    preds = (raw_outputs >= threshold).astype(int)
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


def save_summary_json_noise(
    result: Dict[str, Any],
    args,
    save_path: Path,
    p_single: float,
    p_two: float,
) -> None:
    """
    Save a summary JSON for noisy certainty evaluation.
    """
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
        "noise_level": args.noise_level,
        "noise_prob_single": float(p_single),
        "noise_prob_two": float(p_two),
        "two_qubit_scale": float(args.two_qubit_scale),
        "inference_mode": args.mode,
        "shots": int(args.shots),
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

    result = evaluate_model_certainty(
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
    )
    samples_df["noise_level"] = args.noise_level
    samples_df["inference_mode"] = args.mode
    samples_df["shots"] = int(args.shots)
    samples_df["noise_prob_single"] = float(p_single)
    samples_df["noise_prob_two"] = float(p_two)

    csv_path = save_dir / f"{prefix}_samples.csv"
    json_path = save_dir / f"{prefix}_summary.json"
    violin_path = save_dir / f"{prefix}_violin.png"
    hist_path = save_dir / f"{prefix}_hist.png"

    samples_df.to_csv(csv_path, index=False)
    save_summary_json_noise(
        result=result,
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