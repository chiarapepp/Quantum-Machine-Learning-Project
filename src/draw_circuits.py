import os
import matplotlib.pyplot as plt
import pennylane as qml
import torch

from architectures import (
    build_mera_qnn,
    build_qcnn_qnn,
    build_simple_qnn,
    build_ttn_qnn,
)

SAVE_DIR = "figures/circuits"
N_QUBITS = 8

def _parse_layer_type(layer_type: str) -> list[str]:
    """Normalize Simple layer type to the corresponding interaction terms."""
    lt = layer_type.upper().replace("_", "").replace("-", "")

    aliases = {
        "XY": "XXYY",
        "ZX": "ZZXX",
        "ZY": "ZZYY",
        "ZXY": "ZZXXYY",
    }
    lt = aliases.get(lt, lt)

    mapping = {
        "ZZXXYY": ["ZZ", "XX", "YY"],
        "ZZXX": ["ZZ", "XX"],
        "XXYY": ["XX", "YY"],
        "ZZYY": ["ZZ", "YY"],
    }

    if lt not in mapping:
        raise ValueError(
            f"Unknown layer_type '{layer_type}'. Supported: "
            "XY/XXYY, ZX/ZZXX, ZY/ZZYY, ZXY/ZZXXYY."
        )

    return mapping[lt]


def simple_num_params(n_feature_qubits: int, n_layers: int, layer_type: str) -> int:
    """
    Number of trainable parameters for the Simple architecture.

    For each layer:
    - one parameter per selected Pauli interaction
    - applied between the result qubit and each feature qubit
    """
    n_terms = len(_parse_layer_type(layer_type))
    return n_layers * n_terms * n_feature_qubits


def ttn_num_params(n_qubits: int) -> int:
    """
    Number of trainable parameters for TTN.

    Each two-qubit block uses 6 parameters.
    A TTN over n qubits uses one block per merge, i.e. n - 1 blocks.
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be at least 1.")
    return 6 * max(0, n_qubits - 1)


def mera_num_params(n_qubits: int, n_scales: int) -> int:
    """
    Number of trainable parameters for MERA.

    Per scale:
    - disentanglers on shifted pairs: floor((len(active) - 1) / 2)
    - isometries on even pairs: floor(len(active) / 2)

    Each block uses 6 parameters.
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be at least 1.")
    if n_scales < 1:
        raise ValueError("n_scales must be at least 1.")

    active = list(range(n_qubits))
    blocks = 0

    for _ in range(n_scales):
        if len(active) < 2:
            break

        blocks += (len(active) - 1) // 2
        blocks += len(active) // 2
        active = active[::2]

    return 6 * blocks


def qcnn_num_params(n_qubits: int) -> int:
    """
    Number of trainable parameters for QCNN.

    For each scale while len(active) > 2:
    - 1 shared convolution block: 6 parameters
    - 1 shared pooling block: 6 parameters

    Then one final 2-qubit block: 6 parameters
    """
    if n_qubits < 2:
        raise ValueError("QCNN requires at least 2 qubits.")

    active_count = n_qubits
    scales = 0

    while active_count > 2:
        scales += 1
        active_count = (active_count + 1) // 2

    return 6 * (2 * scales + 1)


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

    # feature qubits + 1 dedicated result qubit
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
    n_scales = 3
    dev = qml.device("default.qubit", wires=n_qubits)

    qnode = build_mera_qnn(
        n_qubits=n_qubits,
        n_scales=n_scales,
        dev=dev,
    )

    x = torch.zeros(n_qubits, dtype=torch.float32)
    params = torch.zeros(mera_num_params(n_qubits, n_scales), dtype=torch.float32)

    draw_and_save(qnode, x, params, f"mera_{n_qubits}q_{n_scales}scales")


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