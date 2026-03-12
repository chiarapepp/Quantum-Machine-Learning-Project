import argparse
import os
from pathlib import Path

import pennylane as qml
from pennylane import numpy as np
from tqdm.auto import tqdm
import wandb

import dataset as ds_module
import data_utils
from architectures import build_mera_qnn, mera_num_params
from training_common import (
    accuracy,
    cost_function,
    current_lr,
    hinge_loss,
    init_metrics_log,
    make_optimizer,
    predict_dataset,
    to_pm_one_labels,
)


def train(
    X_train,
    y_train,
    X_val,
    y_val,
    lr=0.01,
    batch_size=16,
    epochs=10,
    seed=123,
    optimizer_name="adam",
    sgd_momentum=0.0,
    sgd_decay=0.0,
    wandb_project="qml-project",
    wandb_run_name=None,
    wandb_mode="online",
    log_batch_metrics=False,
):
    np.random.seed(seed)

    y_train = to_pm_one_labels(y_train)
    y_val = to_pm_one_labels(y_val)

    X_train = np.asarray(X_train, dtype=float)
    X_val = np.asarray(X_val, dtype=float)

    n_feature_qubits = X_train.shape[1]
    n_train = len(X_train)
    n_val = len(X_val)

    num_params = mera_num_params(n_feature_qubits)
    weights = np.random.random(num_params, requires_grad=True) * np.pi

    dev = qml.device("default.qubit", wires=n_feature_qubits)

    qnode = build_mera_qnn(
        n_qubits=n_feature_qubits,
        dev=dev,
        interface="autograd",
    )

    default_run_name = f"mera_qnn_opt-{optimizer_name}_fq{n_feature_qubits}_lr{lr}"
    if optimizer_name == "sgd":
        default_run_name += f"_mom{sgd_momentum}_decay{sgd_decay}"

    run_name = wandb_run_name or default_run_name

    wandb.init(
        project=wandb_project,
        name=run_name,
        mode=wandb_mode,
        config={
            "model": "mera_qnn",
            "optimizer": optimizer_name,
            "sgd_momentum": sgd_momentum if optimizer_name == "sgd" else None,
            "sgd_decay": sgd_decay if optimizer_name == "sgd" else None,
            "n_feature_qubits": n_feature_qubits,
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "seed": seed,
            "num_params": num_params,
            "train_size": n_train,
            "val_size": n_val,
        },
    )

    metrics_log = init_metrics_log()

    best_val_loss = float("inf")
    best_epoch = -1
    best_weights = np.array(weights, requires_grad=False)

    global_step = 0

    print(
        f"[train] start | train={X_train.shape}, val={X_val.shape}, "
        f"qubits={n_feature_qubits}, params={num_params}, "
        f"optimizer={optimizer_name}, sgd_momentum={sgd_momentum}, sgd_decay={sgd_decay}"
    )

    epoch_bar = tqdm(range(epochs), desc="Training", leave=True)

    for epoch in epoch_bar:
        perm = np.random.permutation(n_train)
        X_train_shuf = X_train[perm]
        y_train_shuf = y_train[perm]

        batch_costs = []
        last_effective_lr = lr

        batch_bar = tqdm(range(0, n_train, batch_size), desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        for start in batch_bar:
            stop = start + batch_size
            xb = X_train_shuf[start:stop]
            yb = y_train_shuf[start:stop]

            effective_lr = current_lr(
                base_lr=lr,
                global_step=global_step,
                optimizer_name=optimizer_name,
                sgd_decay=sgd_decay,
            )
            last_effective_lr = effective_lr

            optimizer = make_optimizer(
                optimizer_name=optimizer_name,
                lr=effective_lr,
                sgd_momentum=sgd_momentum,
            )

            weights, cost = optimizer.step_and_cost(
                lambda w: cost_function(w, xb, yb, qnode),
                weights,
            )

            cost_value = float(cost)
            batch_costs.append(cost_value)

            if log_batch_metrics:
                wandb.log(
                    {
                        "batch/loss": cost_value,
                        "batch/lr": effective_lr,
                        "batch/size": len(xb),
                        "global_step": global_step,
                        "epoch": epoch + 1,
                    },
                    step=global_step,
                )

            batch_bar.set_postfix(batch_loss=f"{cost_value:.4f}", lr=f"{effective_lr:.5f}")
            global_step += 1

        train_preds = predict_dataset(weights, X_train, qnode)
        val_preds = predict_dataset(weights, X_val, qnode)

        train_loss = float(hinge_loss(y_train, train_preds))
        val_loss = float(hinge_loss(y_val, val_preds))
        train_acc = float(accuracy(y_train, train_preds))
        val_acc = float(accuracy(y_val, val_preds))
        epoch_mean_batch_loss = float(np.mean(np.array(batch_costs)))

        metrics_log["train_loss"].append(train_loss)
        metrics_log["train_acc"].append(train_acc)
        metrics_log["val_loss"].append(val_loss)
        metrics_log["val_acc"].append(val_acc)
        metrics_log["epoch_mean_batch_loss"].append(epoch_mean_batch_loss)
        metrics_log["effective_lr"].append(last_effective_lr)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_weights = np.array(weights, requires_grad=False)

        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            train_acc=f"{train_acc:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_acc=f"{val_acc:.4f}",
        )

        print(
            f"Epoch {epoch + 1:03d} | "
            f"mean_batch_loss={epoch_mean_batch_loss:.4f} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
            f"lr={last_effective_lr:.6f} | "
            f"best_val_loss={best_val_loss:.4f} @ epoch {best_epoch:03d}"
        )

        wandb.log(
            {
                "epoch": epoch + 1,
                "epoch/mean_batch_loss": epoch_mean_batch_loss,
                "epoch/train_loss": train_loss,
                "epoch/train_acc": train_acc,
                "epoch/val_loss": val_loss,
                "epoch/val_acc": val_acc,
                "epoch/lr": last_effective_lr,
                "epoch/best_val_loss": best_val_loss,
                "epoch/best_epoch": best_epoch,
            },
            step=global_step,
        )

    wandb.summary["best_val_loss"] = best_val_loss
    wandb.summary["best_epoch"] = best_epoch
    wandb.summary["final_train_loss"] = metrics_log["train_loss"][-1]
    wandb.summary["final_train_acc"] = metrics_log["train_acc"][-1]
    wandb.summary["final_val_loss"] = metrics_log["val_loss"][-1]
    wandb.summary["final_val_acc"] = metrics_log["val_acc"][-1]

    wandb.finish()

    return {
        "weights": weights,
        "best_weights": best_weights,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "metrics_log": metrics_log,
        "qnode": qnode,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train MERA QNN on encoded NF-UNSW data")

    parser.add_argument("--processed-csv", type=str, default="data/processed/nf_unsw_balanced.csv", help="Path to processed CSV")
    parser.add_argument("--raw-csv", type=str, default="data/raw/NF-UNSW-NB15-v2.csv", help="Path to raw CSV used if processed CSV is missing")
    parser.add_argument("--test-size", type=float, default=0.15, help="Validation/test split fraction")
    parser.add_argument("--random-state", type=int, default=123, help="Random state for dataset split")
    parser.add_argument("--n-bins", type=int, default=100, help="Number of percentile bins for encoding")

    parser.add_argument("--lr", type=float, default=0.01, choices=[0.1, 0.05, 0.02, 0.015, 0.01, 0.005, 0.001], help="Learning rate from paper grid")
    parser.add_argument("--batch-size", type=int, default=16, choices=[16, 32], help="Mini-batch size from paper grid")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")

    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"], help="Optimizer from paper grid")
    parser.add_argument("--sgd-momentum", type=float, default=0.0, choices=[0.0, 0.2, 0.3], help="SGD momentum from paper grid")
    parser.add_argument("--sgd-decay", type=float, default=0.0, choices=[0.0, 0.001, 0.01], help="SGD decay rate from paper grid")

    parser.add_argument("--wandb-project", type=str, default="qml-project", help="Weights & Biases project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Weights & Biases run name")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"], help="Weights & Biases mode")
    parser.add_argument("--log-batch-metrics", action="store_true", help="Log batch loss to wandb")

    parser.add_argument("--save-dir", type=str, default="outputs/mera", help="Directory where final artifacts are saved")
    parser.add_argument("--save-best-weights", action="store_true", help="Save best validation weights to disk")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    processed_csv = args.processed_csv
    raw_csv = args.raw_csv
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(processed_csv):
        print("[train] processed CSV not found, generating it from raw CSV...")
        os.makedirs(os.path.dirname(processed_csv) or ".", exist_ok=True)
        ds_module.build_processed_nf_unsw(csv_path=raw_csv, save_processed_csv=processed_csv)

    pack = data_utils.load_encoded_splits(
        processed_csv=processed_csv,
        test_size=args.test_size,
        random_state=args.random_state,
        n_bins=args.n_bins,
    )

    print(f"[train] encoded data loaded | train={pack['X_train'].shape}, test={pack['X_test'].shape}")

    result = train(
        X_train=pack["X_train"],
        y_train=pack["y_train"],
        X_val=pack["X_test"],
        y_val=pack["y_test"],
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        optimizer_name=args.optimizer,
        sgd_momentum=args.sgd_momentum,
        sgd_decay=args.sgd_decay,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
        log_batch_metrics=args.log_batch_metrics,
    )

    if args.save_best_weights:
        best_weights_path = save_dir / "best_weights.npy"
        np.save(str(best_weights_path), np.asarray(result["best_weights"], dtype=float))
        print(f"[train] saved best weights to {best_weights_path}")

    final_weights_path = save_dir / "final_weights.npy"
    np.save(str(final_weights_path), np.asarray(result["weights"], dtype=float))
    print(f"[train] saved final weights to {final_weights_path}")

    metrics_path = save_dir / "metrics_log.npz"
    np.savez(
        str(metrics_path),
        train_loss=np.asarray(result["metrics_log"]["train_loss"], dtype=float),
        train_acc=np.asarray(result["metrics_log"]["train_acc"], dtype=float),
        val_loss=np.asarray(result["metrics_log"]["val_loss"], dtype=float),
        val_acc=np.asarray(result["metrics_log"]["val_acc"], dtype=float),
        epoch_mean_batch_loss=np.asarray(result["metrics_log"]["epoch_mean_batch_loss"], dtype=float),
        effective_lr=np.asarray(result["metrics_log"]["effective_lr"], dtype=float),
    )
    print(f"[train] saved metrics log to {metrics_path}")