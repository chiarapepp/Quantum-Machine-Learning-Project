
# Quantum Network Anomaly Detection — Reproduction

This repository contains code to reproduce a quantum-assisted network anomaly
detection pipeline (implementation inspired by Kukliansky et al.). The code in
this workspace provides preprocessing and a lightweight quantum neural network
implementation along with training tooling.

Key ideas implemented here:

- Quantized-angle encoding of NetFlow features into qubit rotation angles.
- A small "Simple" QNN architecture built from two-qubit rotation blocks.
- Training and utilities to save checkpoints and diagnostics.

Project layout (top-level items you'll care about):

- `src/` — main Python sources
  - `src/dataset.py` — dataset loading and preprocessing helpers
  - `src/encoding.py` — quantized-angle encoding helpers
  - `src/preprocessing.py` — preprocessing utilities and feature selection
  - `src/qnn_simple.py` — quantum circuit / model builder for the Simple QNN
  - `src/certainty_factor.py` — certainty / confidence utilities
  - `src/architectures.py` — model architecture variants and helpers
  - `src/train.py` — training pipeline, checkpointing and basic logging
- `data/` — raw and processed datasets (CSV and encoded artifacts)
- `results/` — training outputs and checkpoints
- `configs/` — example configuration (e.g. `training.yml`)
- `requirements.txt` — pinned Python dependencies used during development
- `paper.md` — reference paper and notes

Note: filenames above match the current repository. If you previously used a
different fork or release that referenced other script names, please use the
files under `src/` in this workspace.

## Quick setup (Linux / bash)

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Tip: check `requirements.txt` for TF / TFQ versions if you plan to run on GPU or
use TensorFlow Quantum — these libraries have strict compatibility requirements.

## Preprocess / Encode

Preprocessing and encoding helpers live under `src/` (see `src/dataset.py` and
`src/encoding.py`). Use those modules to convert the raw CSV under
`data/raw/` to processed arrays or a compressed `.npz` with your chosen
encodings. The repository does not enforce a single CLI entrypoint for
preprocessing; import the helpers from Python or extend `src/train.py` to add
a dedicated CLI command if you need one.

Example (run from repository root) — run the training script which can accept
an encoded `.npz` or re-run preprocessing depending on flags supported by the
script:

```bash
python src/train.py --npz data/processed/nf_encoded.npz --out-dir experiments/test_run --n-epochs 20 --sample-limit 1000
```

The `--npz`/`--csv` flags and other arguments are handled by the script in
`src/train.py` (inspect that file for the exact CLI). `--sample-limit` is
useful for quick debugging.

## Results & Checkpoints

Training outputs are written to `results/` (or to the directory you pass as
`--out-dir`). Checkpoints and logs are saved so you can evaluate runs later.

## Reproducibility

- Save any generated encoding tables and `manifest.json` (if produced by a run)
  to reproduce experiments.
- The repository follows deterministic preprocessing choices where possible
  (e.g., fixed random seeds for resampling). Check the preprocessing code for
  the exact seeds and resampling approach.

## Development & Tests (suggestions)

- Add unit tests under a `tests/` folder to validate encoding correctness,
  circuit shapes, and the certainty factor computation. Consider mocking TFQ
  in tests so CI can run without specialized hardware.
- Add a short example Jupyter notebook that runs a tiny end-to-end experiment
  with synthetic data if you'd like a quick demo that doesn't require TFQ.

## Notes and next steps

- I updated this README to reflect the current repository layout and Linux
  usage. If you'd like, I can:
  1. Add a small example notebook (`notebooks/demo.ipynb`) with synthetic data.
  2. Add a dedicated CLI for preprocessing (if you prefer `python -m src.preprocess`).
  3. Add unit tests for encoding and the certainty factor.

If you'd like one of the follow-ups implemented now, tell me which and I'll add
it.
