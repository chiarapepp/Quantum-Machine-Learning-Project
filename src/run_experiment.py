# src/experiments_paper.py
"""
Paper-exact experiment runner.

Runs the full hyperparameter grid from the paper (as close as possible) and saves:
- results/grid_summary.json  (all trials + best_val_f1 + ckpt path)
- results/best_overall.json  (best trial)
- results/best_per_arch.json (best trial per architecture)

Optionally runs noise sweep for the best checkpoint and saves:
- results/noise_sweep_best.json

IMPORTANT:
- This grid is HUGE and will take a lot of compute (paper-level).
- It assumes train.py exposes:
    - DEFAULT_CONFIG
    - prepare_angles_from_processed_csv(processed_csv_path)
    - run_one_trial(X_train, y_train, X_val, y_val, cfg)
    - evaluate_under_noise(trained_state_dict, cfg, X_test, y_test, noise_level, shots)
"""

import os
import json
import itertools
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Import your existing training pipeline
import train as train_module


# -------------------------
# Paper grid (exact values)
# -------------------------
PAPER_BATCH_SIZES = [16, 32]
PAPER_LRS = [0.1, 0.05, 0.02, 0.015, 0.01, 0.005, 0.001]
PAPER_OPTIMIZERS = ["adam", "sgd"]
PAPER_MOMENTA = [0.0, 0.2, 0.3]            # only for SGD
PAPER_WEIGHT_DECAYS = [0.0, 0.001, 0.01]
PAPER_ARCHS = ["ttn", "mera", "qcnn", "simple"]
PAPER_SIMPLE_LAYERS = [1, 2, 4, 6]
PAPER_SIMPLE_LAYER_TYPES = ["ZZXXYY", "ZZXX", "XXYY", "ZZYY"]  # paper layer types
PAPER_DEFAULT_LAYERS = 2

PAPER_NOISE_LEVELS = ["clean", "very_low", "low", "medium", "high", "very_high"]


# -------------------------
# Utils
# -------------------------
def _project_root() -> str:
    # repo root = one folder above src/
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _results_dir() -> str:
    return os.path.join(_project_root(), "results")


def _load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r") as f:
        return json.load(f)


def _save_json(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def iter_paper_trials() -> List[Dict[str, Any]]:
    """
    Build the full set of trials to run, matching paper grid:
    - arch in {ttn, mera, qcnn, simple}
    - for simple: layer_type in {ZZXXYY, ZZXX, XXYY, ZZYY} and n_layers in {1,2,4,6}
    - for others: n_layers = PAPER_DEFAULT_LAYERS and layer_type="na"
    - batch_size, lr, optimizer, weight_decay, momentum (only sgd)
    """
    trials = []
    for arch in PAPER_ARCHS:
        if arch == "simple":
            layers_list = PAPER_SIMPLE_LAYERS
            layer_types = PAPER_SIMPLE_LAYER_TYPES
        else:
            layers_list = [PAPER_DEFAULT_LAYERS]
            layer_types = ["na"]

        for n_layers in layers_list:
            for layer_type in layer_types:
                for bs, lr, opt, wd in itertools.product(
                    PAPER_BATCH_SIZES, PAPER_LRS, PAPER_OPTIMIZERS, PAPER_WEIGHT_DECAYS
                ):
                    if opt == "sgd":
                        for mom in PAPER_MOMENTA:
                            trials.append({
                                "arch": arch,
                                "n_layers": n_layers,
                                "layer_type": layer_type,
                                "batch_size": bs,
                                "lr": lr,
                                "optimizer": opt,
                                "momentum": mom,
                                "weight_decay": wd
                            })
                    else:
                        trials.append({
                            "arch": arch,
                            "n_layers": n_layers,
                            "layer_type": layer_type,
                            "batch_size": bs,
                            "lr": lr,
                            "optimizer": opt,
                            "momentum": 0.0,
                            "weight_decay": wd
                        })
    return trials


def make_run_name(t: Dict[str, Any]) -> str:
    return (
        f"{t['arch']}_L{t['n_layers']}"
        f"_lt{t.get('layer_type','na')}"
        f"_bs{t['batch_size']}"
        f"_lr{t['lr']}"
        f"_opt{t['optimizer']}"
        f"_mom{t.get('momentum',0.0)}"
        f"_wd{t['weight_decay']}"
    )


def best_per_arch(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    best = {}
    for r in results:
        arch = r["config"]["arch"]
        if arch not in best or r["best_val_f1"] > best[arch]["best_val_f1"]:
            best[arch] = r
    return best


def best_overall(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        raise RuntimeError("No results to select best from.")
    return max(results, key=lambda x: x["best_val_f1"])


# -------------------------
# Main
# -------------------------
def main():
    root = _project_root()
    resdir = _results_dir()

    grid_path = os.path.join(resdir, "grid_summary.json")
    best_overall_path = os.path.join(resdir, "best_overall.json")
    best_per_arch_path = os.path.join(resdir, "best_per_arch.json")
    noise_path = os.path.join(resdir, "noise_sweep_best.json")

    # Load / encode once
    cfg0 = dict(train_module.DEFAULT_CONFIG)
    # Use paths relative to repo root
    data_csv = os.path.join(root, cfg0["data_csv"])
    if not os.path.exists(data_csv):
        raise FileNotFoundError(
            f"Processed CSV not found at {data_csv}. "
            f"Generate it first (python src/dataset.py) or set DEFAULT_CONFIG['data_csv']."
        )

    print(f"[paper] Using processed CSV: {data_csv}")
    pack = train_module.prepare_angles_from_processed_csv(data_csv)
    X_train, y_train = pack["X_train"], pack["y_train"]
    X_test, y_test = pack["X_test"], pack["y_test"]

    print(f"[paper] Encoded shapes train={X_train.shape} test={X_test.shape}")
    print(f"[paper] Label balance train={np.bincount(y_train)} test={np.bincount(y_test)}")

    # Resume support: load existing results and skip already-done run_names
    results = _load_json(grid_path, default=[])
    done = set(r.get("run_name") for r in results)

    trials = iter_paper_trials()
    print(f"[paper] Total trials in paper grid: {len(trials)}")
    if results:
        print(f"[paper] Resuming: already completed {len(results)} trials.")

    # Run all trials (paper-exact grid)
    pbar = tqdm(trials, desc="paper grid", dynamic_ncols=True)
    for t in pbar:
        run_name = make_run_name(t)
        if run_name in done:
            continue

        trial_cfg = {
            "arch": t["arch"],
            "n_layers": t["n_layers"],
            "layer_type": t.get("layer_type", "na"),
            "n_qubits": cfg0["n_qubits"],
            "batch_size": t["batch_size"],
            "lr": t["lr"],
            "optimizer": t["optimizer"],
            "momentum": t.get("momentum", 0.0),
            "weight_decay": t["weight_decay"],
            "epochs": cfg0["epochs"],              # paper-level (you can change if needed)
            "hinge_margin": cfg0["hinge_margin"],
            "output_dir": os.path.join(root, cfg0["output_dir"]),
            "wandb_project": cfg0["wandb_project"],
            "use_cuda": cfg0["use_cuda"],
            "run_name": run_name,
        }

        # NOTE: For non-simple architectures, layer_type is ignored by the builder
        # (but leaving it in config is harmless).
        res = train_module.run_one_trial(X_train, y_train, X_test, y_test, trial_cfg)
        entry = {"run_name": run_name, "config": trial_cfg, **res}
        results.append(entry)
        done.add(run_name)

        # Save continuously (so you can stop/resume)
        _save_json(grid_path, results)

        # live update
        pbar.set_postfix(best=max(r["best_val_f1"] for r in results))

    # Save best summaries
    best_all = best_overall(results)
    best_arch = best_per_arch(results)

    _save_json(best_overall_path, best_all)
    _save_json(best_per_arch_path, best_arch)

    print(f"[paper] Saved grid results: {grid_path}")
    print(f"[paper] Saved best overall: {best_overall_path}")
    print(f"[paper] Saved best per arch: {best_per_arch_path}")
    print(f"[paper] Best overall F1: {best_all['best_val_f1']:.6f}")
    print(f"[paper] Best overall config: {best_all['config']}")

    # Noise sweep for best checkpoint (paper)
    if best_all.get("best_ckpt"):
        ckpt = torch.load(best_all["best_ckpt"], map_location="cpu")
        state = ckpt["model_state"]

        sweep = {}
        for lvl in PAPER_NOISE_LEVELS:
            print(f"[paper] Noise sweep level: {lvl}")
            stats = train_module.evaluate_under_noise(
                trained_state_dict=state,
                cfg=best_all["config"],
                X_test=X_test,
                y_test=y_test,
                noise_level=lvl,
                shots=200
            )
            sweep[lvl] = {k: stats[k] for k in ["f1", "accuracy", "precision", "recall", "roc_auc", "cert_mean", "cert_std"]}
            print(f"  F1={sweep[lvl]['f1']:.4f} Acc={sweep[lvl]['accuracy']:.4f}")

        _save_json(noise_path, sweep)
        print(f"[paper] Saved noise sweep: {noise_path}")
    else:
        print("[paper] No best_ckpt found; cannot run noise sweep.")


if __name__ == "__main__":
    main()