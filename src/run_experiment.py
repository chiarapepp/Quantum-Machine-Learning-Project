from __future__ import annotations

import os
import json
import argparse
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

import dataset as ds_module
import data_utils
import train as train_module
import noise_eval as noise_module


# ---------------------------------------------------------------------------
# Paper grid
# ---------------------------------------------------------------------------

PAPER_BATCH_SIZES = [16, 32]
PAPER_LRS = [0.1, 0.05, 0.02, 0.015, 0.01, 0.005, 0.001]
PAPER_OPTIMIZERS = ["adam", "sgd"]
PAPER_MOMENTA = [0.0, 0.2, 0.3]          # only for SGD
PAPER_LR_DECAYS = [0.0, 0.001, 0.01]
PAPER_ARCHS = ["ttn", "mera", "qcnn", "simple"]

PAPER_SIMPLE_LAYERS = [1, 2, 4, 6]
PAPER_SIMPLE_LAYER_TYPES = ["ZZXXYY", "ZZXX", "XXYY", "ZZYY"]

PAPER_DEFAULT_LAYERS = 2


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def iter_paper_trials() -> List[Dict[str, Any]]:
    """
    Build the paper-style hyperparameter grid.

    Paper search:
    - batch sizes: 16, 32
    - learning rates: 0.1, 0.05, 0.02, 0.015, 0.01, 0.005, 0.001
    - optimizers: Adam, SGD
    - momentum: 0, 0.2, 0.3
    - decay: 0, 0.001, 0.01
    - architectures: TTN, MERA, QCNN, Simple
    - for Simple only: layers {1,2,4,6} and entanglement sets
      {ZZXXYY, ZZXX, XXYY, ZZYY}
    """
    trials: List[Dict[str, Any]] = []

    for arch in PAPER_ARCHS:
        if arch == "simple":
            layer_list = PAPER_SIMPLE_LAYERS
            layer_types = PAPER_SIMPLE_LAYER_TYPES
        else:
            layer_list = [PAPER_DEFAULT_LAYERS]
            layer_types = ["na"]

        for n_layers in layer_list:
            for layer_type in layer_types:
                for batch_size in PAPER_BATCH_SIZES:
                    for lr in PAPER_LRS:
                        for optimizer in PAPER_OPTIMIZERS:
                            for lr_decay in PAPER_LR_DECAYS:
                                momenta = PAPER_MOMENTA if optimizer == "sgd" else [0.0]

                                for momentum in momenta:
                                    trials.append(
                                        {
                                            "arch": arch,
                                            "n_layers": n_layers,
                                            "layer_type": layer_type,
                                            "batch_size": batch_size,
                                            "lr": lr,
                                            "optimizer": optimizer,
                                            "momentum": momentum,
                                            "lr_decay": lr_decay,
                                        }
                                    )

    return trials


def make_run_name(trial: Dict[str, Any]) -> str:
    """
    Deterministic run name for resume support.
    """
    arch = trial["arch"]
    n_layers = trial["n_layers"]
    layer_type = trial["layer_type"]
    batch_size = trial["batch_size"]
    lr = trial["lr"]
    optimizer = trial["optimizer"]
    momentum = trial["momentum"]
    lr_decay = trial["lr_decay"]

    return (
        f"{arch}"
        f"_L{n_layers}"
        f"_T{layer_type}"
        f"_BS{batch_size}"
        f"_LR{lr}"
        f"_OPT{optimizer}"
        f"_M{momentum}"
        f"_D{lr_decay}"
    )


def best_overall(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        raise RuntimeError("No results available.")
    return max(results, key=lambda r: float(r["best_val_f1"]))


def best_per_arch(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not results:
        raise RuntimeError("No results available.")

    output: Dict[str, Dict[str, Any]] = {}
    for arch in PAPER_ARCHS:
        arch_runs = [r for r in results if r["config"]["arch"] == arch]
        if arch_runs:
            output[arch] = max(arch_runs, key=lambda r: float(r["best_val_f1"]))
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", default=train_module.DEFAULT_CFG["data_csv"])
    parser.add_argument("--raw_csv", default=train_module.DEFAULT_CFG["raw_csv"])
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--epochs", type=int, default=train_module.DEFAULT_CFG["epochs"])
    parser.add_argument("--n_bins", type=int, default=train_module.DEFAULT_CFG["n_bins"])
    parser.add_argument("--n_feature_qubits", type=int, default=train_module.DEFAULT_CFG["n_feature_qubits"])
    parser.add_argument("--split_random_state", type=int, default=train_module.DEFAULT_CFG["split_random_state"])
    parser.add_argument("--test_size", type=float, default=train_module.DEFAULT_CFG["test_size"])
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--run_noise", action="store_true")
    parser.add_argument("--noise_shots", type=int, default=200)
    parser.add_argument("--noise_batch_size", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    grid_summary_path = os.path.join(args.output_dir, "grid_summary.json")
    best_overall_path = os.path.join(args.output_dir, "best_overall.json")
    best_per_arch_path = os.path.join(args.output_dir, "best_per_arch.json")
    noise_summary_path = os.path.join(args.output_dir, "noise_sweep_best.json")

    if not os.path.exists(args.data_csv):
        print("[run_experiment] processed CSV not found, generating it from raw CSV...")
        ds_module.build_processed_nf_unsw(
            csv_path=args.raw_csv,
            save_processed_csv=args.data_csv,
        )

    pack = data_utils.load_encoded_splits(
        processed_csv=args.data_csv,
        test_size=args.test_size,
        random_state=args.split_random_state,
        n_bins=args.n_bins,
    )

    X_train = pack["X_train"]
    y_train = pack["y_train"]
    X_test = pack["X_test"]
    y_test = pack["y_test"]

    print(
        f"[run_experiment] encoded data loaded | "
        f"train={X_train.shape}, test={X_test.shape}"
    )
    print(
        f"[run_experiment] label balance | "
        f"train={np.bincount(y_train)}, test={np.bincount(y_test)}"
    )

    results: List[Dict[str, Any]] = _load_json(grid_summary_path, default=[])
    done = {r["run_name"] for r in results if "run_name" in r}

    trials = iter_paper_trials()
    print(f"[run_experiment] total trials: {len(trials)}")
    if done:
        print(f"[run_experiment] resume mode: {len(done)} trials already completed.")

    pbar = tqdm(trials, desc="paper-grid", dynamic_ncols=True)

    for trial in pbar:
        run_name = make_run_name(trial)
        if run_name in done:
            continue

        cfg = train_module.DEFAULT_CFG.copy()
        cfg.update(
            {
                "arch": trial["arch"],
                "n_feature_qubits": args.n_feature_qubits,
                "n_layers": trial["n_layers"],
                "layer_type": trial["layer_type"],
                "batch_size": trial["batch_size"],
                "lr": trial["lr"],
                "optimizer": trial["optimizer"],
                "momentum": trial["momentum"],
                "lr_decay": trial["lr_decay"],
                "epochs": args.epochs,
                "n_bins": args.n_bins,
                "test_size": args.test_size,
                "split_random_state": args.split_random_state,
                "data_csv": args.data_csv,
                "raw_csv": args.raw_csv,
                "output_dir": os.path.join(args.output_dir, "checkpoints"),
                "use_cuda": args.use_cuda,
                "use_wandb": args.use_wandb,
                "run_name": run_name,
            }
        )

        result = train_module.run_trial(
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            cfg=cfg,
        )

        entry = {
            "run_name": run_name,
            "config": cfg,
            **result,
        }
        results.append(entry)
        done.add(run_name)

        _save_json(grid_summary_path, results)

        pbar.set_postfix(
            current=run_name,
            best_so_far=max(float(r["best_val_f1"]) for r in results),
        )

    if not results:
        raise RuntimeError("No experiment results were produced.")

    best_all = best_overall(results)
    best_arch = best_per_arch(results)

    _save_json(best_overall_path, best_all)
    _save_json(best_per_arch_path, best_arch)

    print("\n[run_experiment] best overall:")
    print(json.dumps(best_all, indent=2))

    print("\n[run_experiment] best per architecture:")
    print(json.dumps(best_arch, indent=2))

    if args.run_noise:
        best_ckpt = best_all.get("best_ckpt")
        best_cfg = best_all.get("config", {})
        best_arch_name = str(best_cfg.get("arch", "")).lower()

        if not best_ckpt:
            raise RuntimeError("Best checkpoint path missing, cannot run noise sweep.")

        if best_arch_name != "simple":
            print(
                "[run_experiment] skipping noise sweep: current noise_eval.py "
                "supports only the simple architecture."
            )
            return

        noise_results = noise_module.run_noise_sweep(
            ckpt_path=best_ckpt,
            X_test=X_test,
            y_test=y_test,
            shots=args.noise_shots,
            batch_size=args.noise_batch_size,
            output_json=noise_summary_path,
        )

        print("\n[run_experiment] noise sweep on best checkpoint:")
        print(json.dumps(noise_results, indent=2))


if __name__ == "__main__":
    main()