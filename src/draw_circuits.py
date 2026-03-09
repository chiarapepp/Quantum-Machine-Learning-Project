# src/draw_circuits.py

import os
import torch
import pennylane as qml
import matplotlib.pyplot as plt

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


def draw_and_save(qnode, x, params, name, figsize=(18, 6)):
    """Draw a circuit with qml.draw_mpl and save it as a PNG."""
    fig, ax = qml.draw_mpl(qnode, expansion_strategy="device")(x, params)
    fig.set_size_inches(*figsize)

    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, f"{name}.png")

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def draw_simple():
    n_qubits = N_QUBITS
    n_layers = 2
    layer_type = "XXYY"

    # 8 feature qubits + 1 dedicated result qubit
    dev = qml.device("default.qubit", wires=n_qubits + 1)

    qnode = build_simple_qnn(
        n_qubits=n_qubits,
        n_layers=n_layers,
        dev=dev,
        layer_type=layer_type,
    )

    x = torch.zeros(n_qubits)
    params = torch.zeros(simple_num_params(n_qubits, n_layers, layer_type))

    draw_and_save(qnode, x, params, f"simple_{n_qubits}q_{layer_type.lower()}")


def draw_ttn():
    n_qubits = N_QUBITS

    dev = qml.device("default.qubit", wires=n_qubits)

    qnode = build_ttn_qnn(
        n_qubits=n_qubits,
        dev=dev,
    )

    x = torch.zeros(n_qubits)
    params = torch.zeros(ttn_num_params(n_qubits))

    draw_and_save(qnode, x, params, f"ttn_{n_qubits}q")


def draw_mera():
    n_qubits = N_QUBITS
    n_scales = 3

    dev = qml.device("default.qubit", wires=n_qubits)

    qnode = build_mera_qnn(
        n_qubits=n_qubits,
        n_scales=n_scales,
        dev=dev,
    )

    x = torch.zeros(n_qubits)
    params = torch.zeros(mera_num_params(n_qubits, n_scales))

    draw_and_save(qnode, x, params, f"mera_{n_qubits}q_{n_scales}scales")


def draw_qcnn():
    n_qubits = N_QUBITS

    dev = qml.device("default.qubit", wires=n_qubits)

    qnode = build_qcnn_qnn(
        n_qubits=n_qubits,
        dev=dev,
    )

    x = torch.zeros(n_qubits)
    params = torch.zeros(qcnn_num_params(n_qubits))

    draw_and_save(qnode, x, params, f"qcnn_{n_qubits}q")


def main():
    draw_simple()
    draw_ttn()
    draw_mera()
    draw_qcnn()


if __name__ == "__main__":
    main()