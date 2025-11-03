"""evaluate.py

Evaluation utilities for trained Simple QNN models.

Given a trained run directory (manifest + checkpoints) and an encoded
.npz (or CSV to re-prepare), this script computes:
- Test set predictions (sign of PQC expectation -> class)
- Precision / recall / F1 and confusion matrix
- Certainty factor per sample (using loader's formula)
- Plots: violin of certainty distribution and histogram binned by certainty

The script loads the model architecture from the run manifest (n_layers,
layer_type), builds the same model, loads weights, and extracts the PQC
layer output (raw expectation in [-1,1]) to compute probabilities and the
certainty factor.

Usage examples:
  python evaluate.py --run-dir experiments/run_... --npz nf_encoded.npz --out-dir eval_out

Requirements: tensorflow, tensorflow-quantum, cirq, seaborn, matplotlib, sklearn
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def load_manifest(run_dir: str):
    man = Path(run_dir) / 'manifest.json'
    if not man.exists():
        raise FileNotFoundError(f"manifest.json not found in {run_dir}")
    with open(man, 'r') as f:
        return json.load(f)


def find_pqc_layer(model):
    # Try to find the TFQ PQC layer inside model.layers
    for layer in model.layers:
        # Avoid importing TFQ at top-level in case not available earlier
        layer_cls_name = layer.__class__.__name__
        if 'PQC' in layer_cls_name or 'PQC' == layer_cls_name:
            return layer
    # fallback: look for layer with attribute 'circuit'
    for layer in model.layers:
        if hasattr(layer, 'circuit'):
            return layer
    raise RuntimeError('Could not find PQC layer in the model')


def plot_certainty_violin(certainty: np.ndarray, out_path: str):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception:
        print('matplotlib/seaborn not available, skipping violin plot')
        return

    sns.set(style='whitegrid')
    plt.figure(figsize=(6, 4))
    sns.violinplot(data=certainty, inner='quartile')
    plt.axhline(0.0, color='k', linestyle='--', label='zero certainty')
    plt.ylabel('Certainty factor (C)')
    plt.title('Certainty factor distribution')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_certainty_histogram(certainty: np.ndarray, predictions_correct: np.ndarray, out_path: str, bins=20):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print('matplotlib not available, skipping histogram')
        return

    plt.figure(figsize=(8, 4))
    # Plot correct vs incorrect stacked histogram
    correct_vals = certainty[predictions_correct]
    incorrect_vals = certainty[~predictions_correct]
    plt.hist([correct_vals, incorrect_vals], bins=bins, label=['correct', 'incorrect'], color=['g', 'r'], stacked=False, alpha=0.7)
    plt.axvline(0.0, color='k', linestyle='--')
    plt.xlabel('Certainty factor (C)')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    p = argparse.ArgumentParser(description='Evaluate trained Simple QNN run')
    p.add_argument('--run-dir', required=True, help='Run directory containing manifest and checkpoint (run_x)')
    p.add_argument('--ckpt', default=None, help='Path to checkpoint file (overrides manifest/default)')
    p.add_argument('--npz', help='Path to pre-encoded .npz produced by nf_dataset_interface')
    p.add_argument('--csv', help='Path to CSV to re-prepare the dataset (alternative to --npz)')
    p.add_argument('--out-dir', default=None, help='Directory to save evaluation outputs')
    args = p.parse_args()

    run_dir = args.run_dir
    manifest = load_manifest(run_dir)

    out_dir = args.out_dir or os.path.join(run_dir, 'evaluation')
    ensure_dir(out_dir)

    # Load data
    from nf_dataset_interface import NFDataInterface

    iface = NFDataInterface(args.csv if args.csv else '')
    if args.npz:
        print('Loading encoded NPZ...')
        X_train_a, X_test_a, y_train, y_test = iface.load_npz(args.npz)
    else:
        print('Preparing data from CSV...')
        X_train_a, X_test_a, y_train, y_test = iface.prepare_data()

    # Build model and load weights
    try:
        import tensorflow as tf
        import tensorflow_quantum as tfq
    except Exception as e:
        raise ImportError('tensorflow and tensorflow_quantum are required for evaluation; install them to run this script') from e

    from qnn_nids_skeleton import build_pqc_model, circuits_from_angle_matrix

    n_layers = manifest.get('n_layers', 2)
    layer_type = manifest.get('layer_type', 'XY')

    print(f'Building model with n_layers={n_layers}, layer_type={layer_type}')
    model, _ = build_pqc_model(n_layers=n_layers, layer_type=layer_type)

    # Determine checkpoint path
    ckpt = args.ckpt or os.path.join(run_dir, 'best_model.h5')
    if not os.path.exists(ckpt):
        # fallback: final_weights.h5
        ckpt = os.path.join(run_dir, 'final_weights.h5')
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f'Could not find checkpoint at {ckpt}')

    # Load weights into model
    try:
        model.load_weights(ckpt)
    except Exception as e:
        print('Warning: failed to load weights into model via load_weights; attempting to load full model via tf.keras.models.load_model')
        try:
            loaded = tf.keras.models.load_model(ckpt)
            model = loaded
        except Exception as e2:
            raise RuntimeError('Failed to load model weights or full model') from e2

    # Convert test set to circuits/tensor
    test_circs = circuits_from_angle_matrix(X_test_a)
    x_test_tensor = tfq.convert_to_tensor(test_circs)

    # Extract PQC layer and build a model to output raw expectation
    pqc_layer = find_pqc_layer(model)
    pqc_model = tf.keras.Model(inputs=model.input, outputs=pqc_layer.output)

    # Get raw expectations in [-1,1]
    expectations = pqc_model.predict(x_test_tensor).flatten()
    probabilities = (expectations + 1.0) / 2.0

    # Predictions by expectation sign: expectation >=0 -> predict +1 (benign), else -1 (malicious)
    preds = np.where(expectations >= 0.0, 1, -1)

    # Compute binary labels for sklearn (malicious=1)
    y_true = np.asarray(y_test).flatten()
    y_true_bin = (y_true == -1).astype(int)
    y_pred_bin = (preds == -1).astype(int)

    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

    f1 = f1_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin)
    rec = recall_score(y_true_bin, y_pred_bin)
    cm = confusion_matrix(y_true_bin, y_pred_bin).tolist()

    metrics = {'f1': float(f1), 'precision': float(prec), 'recall': float(rec), 'confusion_matrix': cm}
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Certainty factor
    certainty = iface.certainty_from_probabilities(probabilities, y_true)

    # Which predictions were correct?
    correct = (preds == y_true)

    # Save certainty array
    np.savez_compressed(os.path.join(out_dir, 'certainty.npz'), certainty=certainty, correct=correct, expectations=expectations, probabilities=probabilities)

    # Plots
    plot_certainty_violin(certainty, os.path.join(out_dir, 'certainty_violin.png'))
    plot_certainty_histogram(certainty, correct, os.path.join(out_dir, 'certainty_hist.png'), bins=30)

    print('Evaluation complete. Metrics saved to', os.path.join(out_dir, 'metrics.json'))
    print('Plots saved to', out_dir)


if __name__ == '__main__':
    main()
