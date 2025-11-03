#!/usr/bin/env bash
set -euo pipefail

# Convenience runner for common tasks in the project. This script is
# intentionally simple: it shows commands and also provides a small
# subcommand interface (setup, preprocess, train_sample, train_full,
# grid, evaluate, demo).

# Edit these defaults as needed
CSV_PATH="path/to/NF-UNSW-NB15.csv"
NPZ_PATH="nf_encoded.npz"
OUT_DIR="experiments/run"
SAMPLE_LIMIT=1000
EPOCHS=20
BATCH_SIZE=32
LR=0.02
N_LAYERS=2
LAYER_TYPE="XY"

usage(){
  cat <<EOF
Usage: $0 <command>

Commands:
  setup             Create venv and install requirements (Linux/macOS)
  preprocess        Run preprocessing and save encoded .npz
  train_sample      Train on a small sample (quick test)
  train_full        Train on full encoded dataset
  grid              Run small grid search
  evaluate          Evaluate a run on the test set
  demo              Run the small model demo in qnn_nids_skeleton
  help              Show this help

Examples:
  $0 setup
  $0 preprocess
  $0 train_sample
  $0 evaluate
EOF
}

cmd_setup(){
  echo "==> Creating virtualenv .venv and installing requirements"
  python3 -m venv .venv
  . .venv/bin/activate
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  echo "Setup complete. Activate with: source .venv/bin/activate"
}

cmd_preprocess(){
  if [ ! -f "$CSV_PATH" ]; then
    echo "CSV not found: $CSV_PATH"
    exit 1
  fi
  echo "==> Preprocessing and encoding: $CSV_PATH -> $NPZ_PATH"
  python -m nf_dataset_interface "$CSV_PATH" --out "$NPZ_PATH"
  echo "Saved encoded arrays to $NPZ_PATH"
}

cmd_train_sample(){
  echo "==> Training small sample (sample limit $SAMPLE_LIMIT)"
  mkdir -p "$OUT_DIR"
  python train.py --npz "$NPZ_PATH" --out-dir "$OUT_DIR" --n-epochs $EPOCHS --sample-limit $SAMPLE_LIMIT --batch-size $BATCH_SIZE --learning-rate $LR --n-layers $N_LAYERS --layer-type $LAYER_TYPE
}

cmd_train_full(){
  echo "==> Training full dataset (no sample limit)"
  mkdir -p "$OUT_DIR"
  python train.py --npz "$NPZ_PATH" --out-dir "$OUT_DIR" --n-epochs $EPOCHS --batch-size $BATCH_SIZE --learning-rate $LR --n-layers $N_LAYERS --layer-type $LAYER_TYPE
}

cmd_grid(){
  echo "==> Running grid search (short defaults)"
  mkdir -p "$OUT_DIR"
  python train.py --npz "$NPZ_PATH" --out-dir "$OUT_DIR" --grid --n-epochs $EPOCHS --sample-limit 10000
}

cmd_evaluate(){
  if [ -z "$1" ]; then
    echo "Usage: $0 evaluate <run_dir> [npz]"
    exit 1
  fi
  RUN_DIR="$1"
  NPZ=${2:-$NPZ_PATH}
  EVAL_OUT=${3:-"${RUN_DIR}/evaluation_out"}
  mkdir -p "$EVAL_OUT"
  python evaluate.py --run-dir "$RUN_DIR" --npz "$NPZ" --out-dir "$EVAL_OUT"
  echo "Evaluation results in $EVAL_OUT"
}

cmd_demo(){
  echo "==> Running qnn_nids_skeleton demo (builds model + forward pass)."
  # This will print a friendly message if TFQ/Cirq are not installed
  python -c "import qnn_nids_skeleton"
}

# Parse command
if [ $# -lt 1 ]; then
  usage
  exit 1
fi

case "$1" in
  setup) cmd_setup ;;
  preprocess) cmd_preprocess ;;
  train_sample) cmd_train_sample ;;
  train_full) cmd_train_full ;;
  grid) cmd_grid ;;
  evaluate) shift; cmd_evaluate "$@" ;;
  demo) cmd_demo ;;
  help|--help|-h) usage ;;
  *) echo "Unknown command: $1"; usage; exit 2 ;;
esac

exit 0
