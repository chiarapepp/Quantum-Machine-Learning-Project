"""
Circuit visualization utilities for project QNN architectures.

This module instantiates reference circuits for Simple, TTN, MERA, and QCNN
models and exports rendered diagrams to the figures directory.
"""

import os
import matplotlib.pyplot as plt
import pennylane as qml
import torch

from architectures import (
    build_simple_qnn,
    build_ttn_qnn,
    build_mera_qnn,
    build_qcnn_qnn,
    simple_num_params,
    ttn_num_params,
    mera_num_params,
    qcnn_num_params,
)

SAVE_DIR = "figures/circuits"
N_QUBITS = 8

def draw_and_save(qnode, x, params, name: str, figsize=(18, 6)) -> None:
    """Draw a circuit with qml.draw_mpl and save it as a PNG."""
    fig, ax = qml.draw_mpl(qnode, level="device")(x, params)
    fig.set_size_inches(*figsize)

    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, f"{name}.png")

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")

def draw_simple() -> None:
    n_feature_qubits = N_QUBITS
    n_layers = 2
    layer_type = "XXYY"

    dev = qml.device("default.qubit", wires=n_feature_qubits + 1)

    qnode = build_simple_qnn(
        n_feature_qubits=n_feature_qubits,
        n_layers=n_layers,
        dev=dev,
        layer_type=layer_type,
    )

    x = torch.zeros(n_feature_qubits, dtype=torch.float32)

    params = torch.zeros(
        simple_num_params(n_feature_qubits, n_layers, layer_type),
        dtype=torch.float32,
    )
    draw_and_save(
        qnode,
        x,
        params,
        f"simple_{n_feature_qubits}fq_{n_layers}layers_{layer_type.lower()}",
    )

def draw_ttn() -> None:
    n_qubits = N_QUBITS
    dev = qml.device("default.qubit", wires=n_qubits)

    qnode = build_ttn_qnn(
        n_qubits=n_qubits,
        dev=dev,
    )
    x = torch.zeros(n_qubits, dtype=torch.float32)
    params = torch.zeros(ttn_num_params(n_qubits), dtype=torch.float32)

    draw_and_save(qnode, x, params, f"ttn_{n_qubits}q")

def draw_mera() -> None:
    n_qubits = N_QUBITS
    dev = qml.device("default.qubit", wires=n_qubits)

    qnode = build_mera_qnn(
        n_qubits=n_qubits,
        dev=dev,
    )
    x = torch.zeros(n_qubits, dtype=torch.float32)
    params = torch.zeros(mera_num_params(n_qubits), dtype=torch.float32)

    draw_and_save(qnode, x, params, f"mera_{n_qubits}q")

def draw_qcnn() -> None:
    n_qubits = N_QUBITS
    dev = qml.device("default.qubit", wires=n_qubits)

    qnode = build_qcnn_qnn(
        n_qubits=n_qubits,
        dev=dev,
    )
    x = torch.zeros(n_qubits, dtype=torch.float32)
    params = torch.zeros(qcnn_num_params(n_qubits), dtype=torch.float32)

    draw_and_save(qnode, x, params, f"qcnn_{n_qubits}q")

def main() -> None:
    draw_simple()
    draw_ttn()
    draw_mera()
    draw_qcnn()
    print("All circuit figures generated successfully.")

if __name__ == "__main__":
    main()