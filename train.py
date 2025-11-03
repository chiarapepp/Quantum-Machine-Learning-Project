"""train.py

Training pipeline for the Simple QNN described in the paper.

Features:
- Load dataset from CSV (via `nf_dataset_interface.NFDataInterface`) or from a
  pre-encoded `.npz` produced by that interface.
- Convert angle arrays into TFQ circuit tensors.
- Build variational PQC model from `qnn_nids_skeleton.build_pqc_model`.
- Train using hinge loss with labels in {-1, 1} and support basic hyperparameter
  grid search. Save best model, training logs, and an experiment manifest.

Note: This script requires Cirq and TensorFlow Quantum (TFQ) installed to
operate. It will raise an informative ImportError otherwise.
"""

import argparse
import json
import os
import time
from itertools import product
from datetime import datetime

import numpy as np

from nf_dataset_interface import NFDataInterface


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def build_and_train(x_train_tensor, y_train, x_val_tensor, y_val, n_layers, layer_type, optimizer_name, lr, batch_size, epochs, out_dir):
    """Build model and train once. Returns path to saved best model and the history dict."""
    # Lazy imports for TFQ and model
    try:
        import tensorflow as tf
        import tensorflow_quantum as tfq
    except Exception as e:
        raise ImportError("TensorFlow and TensorFlow Quantum are required to run training. Install 'tensorflow' and 'tensorflow-quantum'.") from e

    from qnn_nids_skeleton import build_pqc_model

    model, params = build_pqc_model(n_layers=n_layers, layer_type=layer_type)

    # Choose optimizer
    if optimizer_name.lower() == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=lr, decay=1e-3, momentum=0.0)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=lr)

    # Recompile model with chosen optimizer
    model.compile(optimizer=opt, loss=tf.keras.losses.Hinge(), metrics=[tf.keras.metrics.BinaryAccuracy()])

    # Callbacks
    ckpt_path = os.path.join(out_dir, 'best_model.h5')
    csv_log = os.path.join(out_dir, 'train_log.csv')
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_loss', mode='min'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(csv_log)
    ]

    # Fit
    history = model.fit(x_train_tensor, y_train, validation_data=(x_val_tensor, y_val), batch_size=batch_size, epochs=epochs, callbacks=callbacks)

    # Save final weights (best saved by checkpoint already)
    final_weights = os.path.join(out_dir, 'final_weights.h5')
    model.save_weights(final_weights)

    return ckpt_path, history.history


def evaluate_model(model_path, x_test_tensor, y_test, out_dir):
    """Load best model weights and evaluate on test set, saving metrics to JSON."""
    try:
        import tensorflow as tf
    except Exception as e:
        raise ImportError("TensorFlow required for evaluation") from e

    from qnn_nids_skeleton import build_pqc_model
    # We need to build a model with the same architecture. The manifest should
    # contain n_layers and layer_type; caller should ensure consistent build.

    # For simplicity assume caller saved manifest with these fields and we can
    # read them from out_dir/manifest.json
    man_path = os.path.join(out_dir, 'manifest.json')
    if not os.path.exists(man_path):
        raise FileNotFoundError(f"manifest.json not found in {out_dir}; cannot reconstruct model architecture")
    with open(man_path, 'r') as f:
        manifest = json.load(f)

    n_layers = manifest.get('n_layers', 2)
    layer_type = manifest.get('layer_type', 'XY')

    model, _ = build_pqc_model(n_layers=n_layers, layer_type=layer_type)

    # Load weights
    model.load_weights(model_path)

    # Predict
    preds = model.predict(x_test_tensor).flatten()

    # Preds are in tanh range [-1,1]; map sign to class -1/1
    pred_sign = np.sign(preds)
    pred_sign[pred_sign == 0] = 1

    # y_test uses -1 for malicious, 1 for benign (per DataLoader)
    y_true = y_test

    # Convert to binary for sklearn: malicious (y==-1) -> 1
    y_true_bin = (y_true == -1).astype(int)
    y_pred_bin = (pred_sign == -1).astype(int)

    # Compute metrics
    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

    f1 = f1_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin)
    rec = recall_score(y_true_bin, y_pred_bin)
    cm = confusion_matrix(y_true_bin, y_pred_bin).tolist()

    metrics = {'f1': float(f1), 'precision': float(prec), 'recall': float(rec), 'confusion_matrix': cm}

    with open(os.path.join(out_dir, 'evaluation.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    p = argparse.ArgumentParser(description='Train Simple QNN per paper')
    p.add_argument('--csv', help='Path to NF-UNSW-NB15 CSV (will run preprocessing/encoding)')
    p.add_argument('--npz', help='Path to pre-encoded .npz file (skips preprocessing)')
    p.add_argument('--out-dir', default=None, help='Directory to store experiment artifacts')
    p.add_argument('--n-epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--learning-rate', type=float, default=0.02)
    p.add_argument('--optimizer', choices=['sgd', 'adam'], default='sgd')
    p.add_argument('--n-layers', type=int, default=2)
    p.add_argument('--layer-type', choices=['XY', 'XX', 'YY'], default='XY')
    p.add_argument('--grid', action='store_true', help='Run small grid search over reasonable defaults')
    p.add_argument('--sample-limit', type=int, default=None, help='Optional limit on number of samples for quick runs')
    args = p.parse_args()

    if args.csv is None and args.npz is None:
        raise ValueError('Either --csv or --npz must be provided')

    # Create output dir
    timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    out_dir = args.out_dir or f"experiments/run_{timestamp}"
    ensure_dir(out_dir)

    # Load data
    iface = NFDataInterface(args.csv) if args.csv else NFDataInterface('')
    if args.npz:
        print('Loading pre-encoded NPZ...')
        X_train_a, X_test_a, y_train, y_test = iface.load_npz(args.npz)
    else:
        print('Preparing data from CSV (this may take a while)...')
        X_train_a, X_test_a, y_train, y_test = iface.prepare_data(sample_limit=args.sample_limit)

    # Split training into train/val
    # use 10% of training as validation
    n_train = X_train_a.shape[0]
    val_split = int(0.1 * n_train)
    if val_split < 1:
        val_split = 1
    X_val = X_train_a[:val_split]
    y_val = y_train[:val_split]
    X_tr = X_train_a[val_split:]
    y_tr = y_train[val_split:]

    # Convert to TFQ circuits
    try:
        import tensorflow_quantum as tfq
    except Exception as e:
        raise ImportError('tensorflow_quantum is required to convert circuits. Install it to proceed.') from e

    from qnn_nids_skeleton import circuits_from_angle_matrix

    print('Converting training circuits...')
    train_circs = circuits_from_angle_matrix(X_tr)
    val_circs = circuits_from_angle_matrix(X_val)
    test_circs = circuits_from_angle_matrix(X_test_a)

    x_train_tensor = tfq.convert_to_tensor(train_circs)
    x_val_tensor = tfq.convert_to_tensor(val_circs)
    x_test_tensor = tfq.convert_to_tensor(test_circs)

    y_tr = np.array(y_tr).astype(np.float32)
    y_val = np.array(y_val).astype(np.float32)
    y_test = np.array(y_test).astype(np.float32)

    # Experiment manifest
    manifest = {
        'timestamp': timestamp,
        'sample_limit': args.sample_limit,
        'n_train': int(X_tr.shape[0]),
        'n_val': int(X_val.shape[0]),
        'n_test': int(X_test_a.shape[0]),
    }

    # Grid search or single run
    runs = []
    if args.grid:
        lrs = [0.1, 0.05, 0.02, 0.01]
        batches = [16, 32]
        layers = [1, 2, 4]
        layer_types = ['XY', 'XX']
        opts = ['sgd', 'adam']
        for lr, batch, nl, lt, opt in product(lrs, batches, layers, layer_types, opts):
            runs.append({'learning_rate': lr, 'batch_size': batch, 'n_layers': nl, 'layer_type': lt, 'optimizer': opt})
    else:
        runs.append({'learning_rate': args.learning_rate, 'batch_size': args.batch_size, 'n_layers': args.n_layers, 'layer_type': args.layer_type, 'optimizer': args.optimizer})

    best_val_loss = float('inf')
    best_run = None

    for i, r in enumerate(runs):
        run_dir = os.path.join(out_dir, f'run_{i}')
        ensure_dir(run_dir)
        print(f"Starting run {i+1}/{len(runs)}: {r}")

        # save run manifest
        run_manifest = {**manifest, **r}
        with open(os.path.join(run_dir, 'manifest.json'), 'w') as f:
            json.dump(run_manifest, f, indent=2)

        ckpt_path, history = build_and_train(x_train_tensor, y_tr, x_val_tensor, y_val,
                                            n_layers=r['n_layers'], layer_type=r['layer_type'], optimizer_name=r['optimizer'],
                                            lr=r['learning_rate'], batch_size=r['batch_size'], epochs=args.n_epochs, out_dir=run_dir)

        # simple val loss check
        val_loss = history.get('val_loss', [float('inf')])[-1]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_run = {'run_index': i, 'run_dir': run_dir, 'manifest': run_manifest, 'ckpt': ckpt_path}

    # Save top-level manifest and best run info
    with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
        json.dump({**manifest, 'n_runs': len(runs), 'best_run': best_run}, f, indent=2)

    # Evaluate best run
    if best_run is not None:
        print('Evaluating best run:', best_run)
        metrics = evaluate_model(best_run['ckpt'], x_test_tensor, y_test, best_run['run_dir'])
        print('Test metrics:', metrics)


if __name__ == '__main__':
    main()
