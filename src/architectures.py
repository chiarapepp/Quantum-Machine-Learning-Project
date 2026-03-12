from typing import Callable, List
import pennylane as qml
from pennylane import numpy as np
from encoding import apply_rx_encoding


def _rot_rot_cnot(params, wires: List[int]) -> None:
    """Two-qubit block used for TTN / MERA / QCNN: Rot ⊗ Rot then CNOT."""
    if len(wires) != 2:
        raise ValueError("_rot_rot_cnot requires exactly 2 wires.")
    if len(params) != 6:
        raise ValueError("_rot_rot_cnot requires exactly 6 parameters.")

    a, b = wires
    qml.Rot(params[0], params[1], params[2], wires=a)
    qml.Rot(params[3], params[4], params[5], wires=b)
    qml.CNOT(wires=[a, b])


def _parse_layer_type(layer_type: str) -> List[str]:
    mapping = {
        "ZZXXYY": ["ZZ", "XX", "YY"],
        "ZZXX": ["ZZ", "XX"],
        "XXYY": ["XX", "YY"],
        "ZZYY": ["ZZ", "YY"],
    }
    if layer_type not in mapping:
        raise ValueError(
            f"Unknown layer_type '{layer_type}'. Supported: XXYY, ZZXX, ZZYY, ZZXXYY."
        )
    return mapping[layer_type]


def simple_num_params(
    n_feature_qubits: int,
    n_layers: int,
    layer_type: str = "XXYY",
) -> int:
    """
    Number of trainable parameters for the Simple architecture.

    For each layer:
    - one parameter per interaction term
    - applied between the result qubit and every feature qubit
    """
    terms = _parse_layer_type(layer_type)
    return n_layers * len(terms) * n_feature_qubits


def ttn_num_params(n_qubits: int) -> int:
    """
    Number of trainable parameters for the TTN architecture.

    Each two-qubit block has 6 parameters.
    """
    blocks = 0
    active_count = n_qubits

    while active_count > 1:
        blocks += active_count // 2
        active_count = (active_count // 2) + (active_count % 2)

    return 6 * blocks


def mera_num_params(n_qubits: int) -> int:
    """
    Number of trainable parameters for the fixed MERA architecture.

    Each disentangler and isometry block has 6 parameters.
    The hierarchy is determined automatically by n_qubits.
    """
    if n_qubits < 2:
        raise ValueError("MERA requires at least 2 qubits.")

    active = list(range(n_qubits))
    blocks = 0

    while len(active) > 1:
        blocks += max(0, (len(active) - 1) // 2)  # disentanglers
        blocks += len(active) // 2                # isometries
        active = active[::2]

    return 6 * blocks


def qcnn_num_params(n_qubits: int) -> int:
    """
    Number of trainable parameters for the QCNN architecture.

    For each scale:
    - one shared convolution block: 6 parameters
    - one shared pooling block: 6 parameters

    After the multiscale stages:
    - one final two-qubit block: 6 parameters
    """
    if n_qubits < 2:
        raise ValueError("QCNN requires at least 2 qubits.")

    n_scales = 0
    active_count = n_qubits

    while active_count > 2:
        n_scales += 1
        active_count = (active_count + 1) // 2

    return 12 * n_scales + 6


def build_simple_qnn(
    n_feature_qubits: int,
    n_layers: int,
    dev,
    layer_type: str = "XXYY",
    interface: str = "torch",
) -> Callable:
    """
    Build the paper's Simple architecture.

    - wires 0..n-1 hold the encoded features
    - wire n is the dedicated result qubit
    - the result qubit is initialized to |-> = X|0> then H|0>
    - each layer applies the chosen Pauli interaction between the
      result qubit and every feature qubit
    - measurement is <X> on the result qubit
    """
    result_wire = n_feature_qubits
    terms = _parse_layer_type(layer_type)

    @qml.qnode(dev, interface=interface, diff_method="best")
    def qnode(x, params):
        apply_rx_encoding(x)

        # Prepare |-> on the result qubit
        qml.PauliX(wires=result_wire)
        qml.Hadamard(wires=result_wire)

        idx = 0
        for _ in range(n_layers):
            for term in terms:
                for feature_wire in range(n_feature_qubits):
                    theta = params[idx]
                    if term == "XX":
                        qml.IsingXX(theta, wires=[result_wire, feature_wire])
                    elif term == "YY":
                        qml.IsingYY(theta, wires=[result_wire, feature_wire])
                    elif term == "ZZ":
                        qml.IsingZZ(theta, wires=[result_wire, feature_wire])
                    else:
                        raise RuntimeError(f"Unexpected term: {term}")
                    idx += 1

        return qml.expval(qml.PauliX(result_wire))

    return qnode


def build_ttn_qnn(
    n_qubits: int,
    dev,
    interface: str = "torch",
) -> Callable:
    """
    Build a TTN-style circuit matching the paper's tree-structure.

    - input encoding: RX on each qubit
    - repeated pairwise two-qubit blocks
    - survivors propagate upward in the tree
    - final readout: <Z> on the remaining qubit
    """
    @qml.qnode(dev, interface=interface, diff_method="best")
    def qnode(x, params):
        apply_rx_encoding(x)

        idx = 0
        active = list(range(n_qubits))

        while len(active) > 1:
            new_active = []

            for i in range(0, len(active) - 1, 2):
                a, b = active[i], active[i + 1]
                _rot_rot_cnot(params[idx: idx + 6], [a, b])
                idx += 6
                new_active.append(a)

            if len(active) % 2 == 1:
                new_active.append(active[-1])

            active = new_active

        return qml.expval(qml.PauliZ(active[0]))

    return qnode

def build_mera_qnn(
    n_qubits: int,
    dev,
    interface: str = "torch",
) -> Callable:
    """
    Build a MERA-style circuit for a fixed input size.

    At each hierarchical scale:
    - apply disentanglers on shifted neighbouring pairs
    - apply isometries on even pairs
    - coarse grain by keeping every other qubit

    Final readout is <Z> on the last surviving qubit.
    """
    if n_qubits < 2:
        raise ValueError("MERA requires at least 2 qubits.")

    @qml.qnode(dev, interface=interface, diff_method="best")
    def qnode(x, params):
        apply_rx_encoding(x)

        idx = 0
        active = list(range(n_qubits))

        while len(active) > 1:
            # Disentanglers: shifted pairs (1,2), (3,4), ...
            for i in range(1, len(active) - 1, 2):
                _rot_rot_cnot(params[idx: idx + 6], [active[i], active[i + 1]])
                idx += 6

            # Isometries: even pairs (0,1), (2,3), ...
            for i in range(0, len(active) - 1, 2):
                _rot_rot_cnot(params[idx: idx + 6], [active[i], active[i + 1]])
                idx += 6

            # Coarse graining: keep every other qubit
            active = active[::2]

        return qml.expval(qml.PauliZ(active[0]))

    return qnode

def build_qcnn_qnn(
    n_qubits: int,
    dev,
    interface: str = "torch",
) -> Callable:
    """
    Build a QCNN circuit.

    For each scale:
    - one shared convolution block is applied on even pairs and then shifted odd pairs
    - one shared pooling block is applied on even pairs
    - coarse graining keeps every other qubit

    After the multiscale stages, one final two-qubit block is applied
    to the last two active qubits.

    Final readout is <Z> on the remaining output qubit.
    """
    if n_qubits < 2:
        raise ValueError("QCNN requires at least 2 qubits.")

    @qml.qnode(dev, interface=interface, diff_method="best")
    def qnode(x, params):
        apply_rx_encoding(x)

        idx = 0
        active = list(range(n_qubits))

        while len(active) > 2:
            conv_params = params[idx: idx + 6]
            idx += 6

            for i in range(0, len(active) - 1, 2):
                _rot_rot_cnot(conv_params, [active[i], active[i + 1]])

            for i in range(1, len(active) - 1, 2):
                _rot_rot_cnot(conv_params, [active[i], active[i + 1]])

            pool_params = params[idx: idx + 6]
            idx += 6

            for i in range(0, len(active) - 1, 2):
                _rot_rot_cnot(pool_params, [active[i], active[i + 1]])

            active = active[::2]

        final_params = params[idx: idx + 6]
        _rot_rot_cnot(final_params, [active[0], active[1]])

        return qml.expval(qml.PauliZ(active[0]))

    return qnode
