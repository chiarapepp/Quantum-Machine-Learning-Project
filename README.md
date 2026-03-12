
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
Quantum_Anomaly_Detection/
│
├── src/
│   ├── architectures.py        # QNN architectures (Simple, TTN, MERA, QCNN)
│   ├── encoding.py             # Feature encoding into quantum states
│   ├── model.py                # QNN model wrapper and circuit construction
│   ├── dataset.py              # Dataset loading and preprocessing
│   ├── train.py                # Training loop
│   ├── evaluate.py             # Evaluation metrics and testing
│   └── certainty_factor.py     # Prediction confidence computation
│
├── draw_circuits.py            # Utility to visualize quantum circuits
├── main.py                     # Main script for training experiments
│
├── data/                       # Processed network traffic dataset
├── images/                     # Circuit visualizations and plots
│
└── README.md                   # Project documentation
```

### Requiriments 

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
A balanced preprocessed version is already provided at `data/processed/nf_unsw_balanced.csv`.

## Usage

Each architecture has its own training script. Run from the repository root:

```bash
python src/train_simple.py
python src/train_ttn.py
python src/train_mera.py
python src/train_qcnn.py
```

To visualize the quantum circuits:
```bash
python src/draw_circuits.py
```

To evaluate a checkpoint under depolarizing noise:
```bash
python src/noise_eval.py
```

## Project Layout

- `src/architectures.py` — quantum circuit builders for all four architectures
- `src/encoding.py` — quantized-angle encoding (`QuantumEncoder`)
- `src/dataset.py` — dataset loading and class-balanced splitting
- `src/certainty_factor.py` — certainty and confidence utilities
- `src/train_simple.py`, `train_ttn.py`, `train_mera.py`, `train_qcnn.py` — training pipelines
- `src/evaluate.py` — evaluation metrics (accuracy, F1, AUC, certainty stats)
- `src/noise_eval.py` — depolarizing noise model and noisy evaluation
- `src/draw_circuits.py` — circuit diagram generation
- `data/` — raw and processed datasets
- `results/` — training checkpoints and grid-search summaries
- `figures/circuits/` — saved circuit diagrams
- `configs/` — example run configurations

## Model Parameters (8 qubits)

| Architecture | Number of Parameters |
|--------------|----------------------|
| Simple (L=2) | 64                   |
| TTN          | 42                   |
| MERA         | 66                   |
| QCNN         | 30                   |

## Results (NF-UNSW-NB15-v2, best validation F1)

| Architecture | Best Val F1 |
|--------------|-------------|
| TTN (L=2)    | 0.748       |


## References

- [Dataset: NF-UNSW-NB15-v2 (NetFlow-based network intrusion detection) ](hhttps://espace.library.uq.edu.au/view/UQ:ffbb0c1) *, The University of Queensland.* 
- [Network Anomaly Detection Using Quantum Neural Networks on Noisy Quantum Computers](https://ieeexplore.ieee.org/document/10415536)*, Kukliansky et al., EEE Transactions on Quantum
Engineering, 5:1–11, 2024.*


