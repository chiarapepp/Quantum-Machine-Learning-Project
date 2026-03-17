
# Quantum Machine Learning for Network Anomaly Detection

## Overview 
This project investigates the use of **Quantum Neural Networks** (QNNs) for **network anomaly detection**, following the methodology presented in the paper:
[*“Network Anomaly Detection Using Quantum Neural Networks on Noisy Quantum Computers”*](https://ieeexplore.ieee.org/document/10415536).

The goal is to reproduce and analyze several QNN architectures designed to operate on **NISQ** (Noisy Intermediate-Scale Quantum) hardware, evaluating their capability to classify network traffic as benign or anomalous.

The project implements and compares four quantum models proposed in the paper:
- Simple variational circuit
- Tree Tensor Network (TTN)
- Multi-scale Entanglement Renormalization Ansatz (MERA)
- Quantum Convolutional Neural Network (QCNN)

All architectures share the same **input encoding scheme** and produce a **single-qubit measurement** used for binary classification (for more details on the architecture reference the report in the folder)

The experiments are performed using PennyLane, enabling simulation of quantum circuits and experimentation with noisy quantum devices.

### Main Objective
- Reproduce the architectures proposed in the paper.
- Study how different quantum circuit topologies affect classification performance.
- Investigate the trade-off between expressibility and hardware feasibility on NISQ devices.
- Analyze the impact of noise models on QNN performance.

### Project Structure
```
Quantum-Machine-Learning-Project/
│
├── src/
│   ├── architectures.py        # QNN architectures (Simple, TTN, MERA, QCNN)
│   ├── data_utils.py           # Data utilities and helpers
│   ├── dataset.py              # Dataset loading and preprocessing
│   ├── draw_circuits.py        # Circuit diagram generation
│   ├── encoding.py             # Feature encoding into quantum states
│   ├── evaluate.py             # Evaluation metrics and testing
│   ├── noise_eval.py           # Noise model evaluation
│   ├── train_mera.py           # MERA training pipeline
│   ├── train_qcnn.py           # QCNN training pipeline
│   ├── train_simple.py         # Simple circuit training pipeline
│   ├── train_ttn.py            # TTN training pipeline
│   └── training_common.py      # Shared training logic
│
├── data/
│   ├── processed/
│   │   └── nf_unsw_balanced.csv
│   └── raw/
│       └── NF-UNSW-NB15-v2.csv
│
├── figures/
│   └── circuits/               # Saved circuit diagrams
├── requirements.txt
└── README.md
```

### Requirements 

1. Clone the repository:
   ```bash
   git clone https://github.com/chiarapepp/Quantum-Machine-Learning-Project.git
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional but recommended) Log in to Weights & Biases:
    ```bash
    wandb login
    ```

### Dataset
Place the raw dataset in `data/raw/`:
```
data/raw/NF-UNSW-NB15-v2.csv
```
To generate the processed dataset, run:
```bash
python src/dataset.py
```
The balanced CSV will be saved to `data/processed/nf_unsw_balanced.csv`. If the processed CSV is missing, the training scripts will create it automatically.

## Usage

Each architecture has its own training script. Run from the repository root:

```bash
python src/train_simple.py
python src/train_ttn.py
python src/train_mera.py
python src/train_qcnn.py
```

All training scripts share a common set of arguments. Example with custom options:

```bash
python src/train_simple.py --epochs 20 --lr 0.005 --batch-size 32 --optimizer adam --save-best-weights
```

### Fixed Parameters (Data, Split, Reproducibility, Outputs)

These are the arguments you typically keep fixed across runs:

- `--processed-csv` (default: `data/processed/nf_unsw_balanced.csv`) -> processed dataset path
- `--raw-csv` (default: `data/raw/NF-UNSW-NB15-v2.csv`) -> raw dataset path used if processed CSV is missing
- `--test-size` (default: `0.15`) -> test split fraction
- `--val-size` (default: `0.15`) -> validation split fraction (of the training split)
- `--n-bins` (default: `100`) -> number of percentile bins for quantum encoding
- `--random-state` (default: `1`) -> split reproducibility
- `--seed` (default: `123`) -> initialization reproducibility
- `--save-dir` (default: `outputs/<arch>`) -> output directory for metrics and weights

### Weights & Biases (Optional)

You can customize logging with:

- `--wandb-project` to choose the project name
- `--wandb-run-name` to set a custom run name

If not provided, run naming is handled automatically.

### Training Configuration

| Argument | Default | Description |
|---|---|---|
| `--epochs` | `10` | Number of training epochs. Increase for longer training, reduce for quick tests. |
| `--lr` | `0.01` | Learning rate. |
| `--batch-size` | `16` | Mini-batch size. |
| `--optimizer` | `adam` | Optimizer type: `adam` or `sgd`. |
| `--sgd-momentum` | `0.0` | SGD momentum (used only when `--optimizer sgd`; allowed: `0.0`, `0.2`, `0.3`). |
| `--sgd-decay` | `0.0` | SGD decay rate (used only when `--optimizer sgd`. |
| `--save-best-weights` | `False` | If enabled, saves the best validation weights to disk. |
| `--n-layers` | `2` | Number of variational layers; available only in `train_simple.py`. |
| `--layer-type` | `XXYY` | Entangling layer type for `train_simple.py`: `XXYY`, `ZZXX`, `ZZYY`, or `ZZXXYY`. |

### Visualization
To visualize the quantum circuits:
```bash
python src/draw_circuits.py
```
| Simple circuit | QCNN circuit  |
|---------------|----------------|
| ![s](figures/circuits/simple_8fq_2layers_xxyy.png) | ![q](figures/circuits/qcnn_8q.png) |

| TTN circuit | MERA circuit  |
|---------------|----------------|
| ![t](figures/circuits/ttn_8q.png) | ![m](figures/circuits/mera_8q_3scales.png) |

### Noise Evaluation
To evaluate a checkpoint under depolarizing noise:
```bash
python src/noise_eval.py
```


## Results (NF-UNSW-NB15-v2, best validation F1)

| Architecture | Best Val F1 |
|--------------|-------------|
| TTN (L=2)    | 0.748       |


## References

- [Dataset: NF-UNSW-NB15-v2 (NetFlow-based network intrusion detection) ](hhttps://espace.library.uq.edu.au/view/UQ:ffbb0c1) *, The University of Queensland.* 
- [Network Anomaly Detection Using Quantum Neural Networks on Noisy Quantum Computers](https://ieeexplore.ieee.org/document/10415536)*, Kukliansky et al., EEE Transactions on Quantum
Engineering, 5:1–11, 2024.*


