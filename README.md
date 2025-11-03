
# Quantum Network Anomaly Detection — Reproduction

This repository contains code to reproduce the methodology from the paper
"Network Anomaly Detection Using Quantum Neural Networks on Noisy Quantum
Computers" (Kukliansky et al.). The project implements:

- A quantized angle encoding of NF-UNSW-NB15 NetFlow features into qubit
  rotations (one qubit per feature with 0.25° granularity).
- A lightweight ("Simple") QNN built from Rxx / Ryy two-qubit blocks and a
  dedicated result qubit measured in the X basis.
- Training and evaluation pipelines using TensorFlow Quantum (TFQ) and
  Cirq for circuit construction.

Files you will care about:
- `dataset.py` — dataset loader and encoding builder (core preprocessing logic).
- `nf_dataset_interface.py` — high-level interface to prepare, save and load
  the encoded angle arrays and convert them to TFQ circuit tensors.
- `qnn_nids_skeleton.py` — implementation of the Simple QNN (circuit builder,
  PQC model builder, certainty factor helper).
- `train.py` — training pipeline with optional (small) grid search, checkpointing
  and logging.
- `evaluate.py` — evaluation script that computes F1/precision/recall, certainty
  factor distributions and produces diagnostic plots.
- `requirements.txt` — pinned dependencies used for development and paper
  reproduction (TF 2.7.0, TFQ 0.7.2).

## Quick setup (Windows PowerShell)

1. Create a Python environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

2. Install dependencies (pinned):

```powershell
python -m pip install -r requirements.txt
```

Notes:
- TensorFlow Quantum (TFQ) has specific compatibility constraints. The
  `requirements.txt` pins TF 2.7.0 and TFQ 0.7.2 which were used in the paper.
  If you need a newer TF version, check TFQ compatibility first.
- TFQ may be slow on CPU-only systems. For faster training, install CUDA and
  matching cuDNN compatible with TensorFlow 2.7.0. Follow TensorFlow's
  official GPU installation guide.

## Preprocess / Encode (produce `.npz`)

Use the high-level interface to preprocess the NF-UNSW-NB15 CSV and produce a
compressed `.npz` containing the angle-encoded arrays and encoding tables.

```powershell
# from repository root
python -m nf_dataset_interface path\to\NF-UNSW-NB15.csv --out nf_encoded.npz
```

Or, from Python:

```python
from nf_dataset_interface import NFDataInterface
iface = NFDataInterface('path/to/NF-UNSW-NB15.csv')
X_train_a, X_test_a, y_train, y_test = iface.prepare_data()
iface.save_npz('nf_encoded.npz')
```

The preprocessing follows the paper: balance benign/malicious via resampling
(`sklearn.utils.resample` with seed 123), selected features, and quantized
angle mapping with 0.25° granularity.

## Train (small example)

Train using the encoded `.npz` (or point to CSV to re-run preprocessing):

```powershell
python .\train.py --npz nf_encoded.npz --out-dir experiments/test_run --n-epochs 20 --sample-limit 1000
```

Notes on arguments:
- `--npz` : path to pre-encoded `.npz` from `nf_dataset_interface`.
- `--csv` : alternatively provide CSV and the interface will preprocess.
- `--sample-limit` : limit samples for quick debugging or hardware-limited runs.
- `--grid` : enable a small grid search over learning rates, batch sizes, layers.

Training artifacts are saved under `--out-dir` as subfolders `run_0`, `run_1`, …
Each run folder contains `manifest.json`, `train_log.csv`, `best_model.h5`, and
`final_weights.h5`.

## Evaluate

After training, evaluate the best run (or a chosen checkpoint):

```powershell
python .\evaluate.py --run-dir experiments/test_run\run_0 --npz nf_encoded.npz --out-dir eval_out
```

Outputs written to `eval_out`:
- `metrics.json` — F1, precision, recall, confusion matrix
- `certainty.npz` — arrays: `certainty`, `correct`, `expectations`, `probabilities`
- `certainty_violin.png`, `certainty_hist.png` — diagnostic plots

## Hardware / Noisy Simulation (notes)

- This repo includes helpers to build Cirq circuits compatible with IonQ/Braket
  (see `qnn_nids_skeleton.py` and `nf_dataset_interface.get_tfq_circuit_tensor`).
- Submitting jobs to Aria/Harmony and using IonQ noise simulators involves
  provider APIs (Amazon Braket / IonQ). For safety, no credentials are stored
  here. If you want templates for Braket submission or IonQ serialization, I
  can add `braket_submit.py` which will be a parameterized template (you must
  add credentials locally).

## Reproducibility

- Each run saves a `manifest.json` containing hyperparameters and dataset
  sizes. Save these and the encoding tables (`nf_dataset_interface.save_npz`) to
  reproduce experiments exactly.
- The code uses hinge loss with labels in {-1,1} as in the paper.

## Development & Tests

- Add unit tests under a `tests/` folder to validate encoding correctness,
  circuit shapes, and the certainty factor computation. If you want, I can add
  minimal tests that mock TFQ so CI can run without GPU.

## Contact / Credits

This code was implemented to match the methods from Kukliansky et al.'s paper
"Network Anomaly Detection Using Quantum Neural Networks on Noisy Quantum
Computers". See `paper.md` in this repository for the full text used as the
reference.

---
If you'd like, I can also add a short example Jupyter notebook that runs a
tiny end-to-end experiment with synthetic data so you can experiment without
installing TFQ. Which would you prefer I add next: hardware templates or the
synthetic demo notebook?
# Quantum-Machine-Learning-Project