# src/architectures.py

from __future__ import annotations

from typing import Callable, List
import math

import pennylane as qml


def _rot_rot_cnot(params, wires: List[int]) -> None:
    """Apply a two-qubit block: Rot on each qubit followed by CNOT."""
    if len(wires) != 2:
        raise ValueError("_rot_rot_cnot requires exactly 2 wires.")
    if len(params) != 6:
        raise ValueError("_rot_rot_cnot requires exactly 6 parameters.")

    a, b = wires
    qml.Rot(params[0], params[1], params[2], wires=a)
    qml.Rot(params[3], params[4], params[5], wires=b)
    qml.CNOT(wires=[a, b])


def _parse_layer_type(layer_type: str) -> List[str]:
    """Return the ordered Pauli terms used in one Simple layer."""
    lt = layer_type.upper().replace("_", "")
    mapping = {
        "ZZXXYY": ["ZZ", "XX", "YY"],
        "ZZXX": ["ZZ", "XX"],
        "XXYY": ["XX", "YY"],
        "ZZYY": ["ZZ", "YY"],
    }

    if lt not in mapping:
        raise ValueError(
            f"Unknown layer_type '{layer_type}'. "
            "Supported values: ZZXXYY, ZZXX, XXYY, ZZYY."
        )

    return mapping[lt]


def _simple_chain_pairs(n_feature_qubits: int, result_wire: int) -> List[List[int]]:
    """Return adjacent pairs along the chain [result, 0, 1, ..., n-1]."""
    chain = [result_wire] + list(range(n_feature_qubits))
    return [[chain[i], chain[i + 1]] for i in range(len(chain) - 1)]


def _validate_1d_length(values, expected: int, name: str) -> None:
    """Validate the length of a 1D tensor or array when shape information is available."""
    if not hasattr(values, "shape"):
        return

    shape = tuple(values.shape)
    if len(shape) != 1:
        raise ValueError(f"{name} must be a 1D tensor or array, got shape {shape}.")

    if int(shape[0]) != expected:
        raise ValueError(f"{name} must have length {expected}, got {int(shape[0])}.")


def simple_num_params(
    n_feature_qubits: int,
    n_layers: int,
    layer_type: str = "XXYY",
) -> int:
    """Return the number of trainable parameters for the Simple architecture."""
    if n_feature_qubits < 1:
        raise ValueError("n_feature_qubits must be at least 1.")
    if n_layers < 1:
        raise ValueError("n_layers must be at least 1.")

    terms = _parse_layer_type(layer_type)
    return n_layers * len(terms) * n_feature_qubits


def ttn_num_params(n_qubits: int) -> int:
    """Return the number of trainable parameters for the TTN."""
    if n_qubits < 2:
        raise ValueError("TTN requires at least 2 qubits.")

    blocks = 0
    active = n_qubits

    while active > 1:
        blocks += active // 2
        active = (active // 2) + (active % 2)

    return 6 * blocks


def mera_num_params(n_qubits: int, n_scales: int) -> int:
    """Return the number of trainable parameters for the MERA-style circuit."""
    if n_qubits < 2:
        raise ValueError("MERA requires at least 2 qubits.")
    if n_scales < 1:
        raise ValueError("n_scales must be at least 1.")

    blocks = 0
    active = list(range(n_qubits))

    for _ in range(n_scales):
        if len(active) < 2:
            break

        blocks += max(0, (len(active) - 1) // 2)
        blocks += len(active) // 2
        active = active[::2]

    return 6 * blocks


def qcnn_num_params(n_qubits: int) -> int:
    """Return the number of trainable parameters for the QCNN."""
    if n_qubits < 2:
        raise ValueError("QCNN requires at least 2 qubits.")

    scales = 0
    active = n_qubits

    while active > 2:
        scales += 1
        active = math.ceil(active / 2)

    return 6 * (2 * scales + 1)


def build_simple_qnn(
    n_feature_qubits: int,
    n_layers: int,
    dev,
    layer_type: str = "XXYY",
) -> Callable:
    """Build the Simple QNN with a dedicated result qubit measured in the X basis."""
    if n_feature_qubits < 1:
        raise ValueError("n_feature_qubits must be at least 1.")
    if n_layers < 1:
        raise ValueError("n_layers must be at least 1.")

    result_wire = n_feature_qubits
    expected_params = simple_num_params(n_feature_qubits, n_layers, layer_type)
    terms = _parse_layer_type(layer_type)
    pairs = _simple_chain_pairs(n_feature_qubits, result_wire)
    params_per_term = len(pairs)

    @qml.qnode(dev, interface="torch", diff_method="best")
    def qnode(x, params):
        _validate_1d_length(x, n_feature_qubits, "x")
        _validate_1d_length(params, expected_params, "params")

        for wire in range(n_feature_qubits):
            qml.RX(x[wire], wires=wire)

        qml.PauliX(wires=result_wire)
        qml.Hadamard(wires=result_wire)

        idx = 0
        for _ in range(n_layers):
            for term in terms:
                for k, (a, b) in enumerate(pairs):
                    theta = params[idx + k]

                    if term == "XX":
                        qml.IsingXX(theta, wires=[a, b])
                    elif term == "YY":
                        qml.IsingYY(theta, wires=[a, b])
                    elif term == "ZZ":
                        qml.IsingZZ(theta, wires=[a, b])

                idx += params_per_term

        return qml.expval(qml.PauliX(result_wire))

    return qnode


def build_ttn_qnn(
    n_qubits: int,
    dev,
) -> Callable:
    """Build a TTN classifier with Z-basis readout on the final active qubit."""
    if n_qubits < 2:
        raise ValueError("TTN requires at least 2 qubits.")

    expected_params = ttn_num_params(n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="best")
    def qnode(x, params):
        _validate_1d_length(x, n_qubits, "x")
        _validate_1d_length(params, expected_params, "params")

        for wire in range(n_qubits):
            qml.RX(x[wire], wires=wire)

        idx = 0
        active = list(range(n_qubits))

        while len(active) > 1:
            new_active = []

            for i in range(0, len(active) - 1, 2):
                a, b = active[i], active[i + 1]
                _rot_rot_cnot(params[idx : idx + 6], [a, b])
                idx += 6
                new_active.append(a)

            if len(active) % 2 == 1:
                new_active.append(active[-1])

            active = new_active

        return qml.expval(qml.PauliZ(active[0]))

    return qnode


def build_mera_qnn(
    n_qubits: int,
    n_scales: int,
    dev,
) -> Callable:
    """Build a MERA-style classifier with Z-basis readout."""
    if n_qubits < 2:
        raise ValueError("MERA requires at least 2 qubits.")
    if n_scales < 1:
        raise ValueError("n_scales must be at least 1.")

    expected_params = mera_num_params(n_qubits, n_scales)

    @qml.qnode(dev, interface="torch", diff_method="best")
    def qnode(x, params):
        _validate_1d_length(x, n_qubits, "x")
        _validate_1d_length(params, expected_params, "params")

        for wire in range(n_qubits):
            qml.RX(x[wire], wires=wire)

        idx = 0
        active = list(range(n_qubits))

        for _ in range(n_scales):
            if len(active) < 2:
                break

            for i in range(1, len(active) - 1, 2):
                a, b = active[i], active[i + 1]
                _rot_rot_cnot(params[idx : idx + 6], [a, b])
                idx += 6

            for i in range(0, len(active) - 1, 2):
                a, b = active[i], active[i + 1]
                _rot_rot_cnot(params[idx : idx + 6], [a, b])
                idx += 6

            active = active[::2]

        return qml.expval(qml.PauliZ(active[0]))

    return qnode


def build_qcnn_qnn(
    n_qubits: int,
    dev,
) -> Callable:
    """Build a QCNN classifier with stage-wise parameter sharing and Z-basis readout."""
    if n_qubits < 2:
        raise ValueError("QCNN requires at least 2 qubits.")

    expected_params = qcnn_num_params(n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="best")
    def qnode(x, params):
        _validate_1d_length(x, n_qubits, "x")
        _validate_1d_length(params, expected_params, "params")

        for wire in range(n_qubits):
            qml.RX(x[wire], wires=wire)

        idx = 0
        active = list(range(n_qubits))

        while len(active) > 2:
            conv_params = params[idx : idx + 6]
            idx += 6

            for i in range(0, len(active) - 1, 2):
                _rot_rot_cnot(conv_params, [active[i], active[i + 1]])

            pool_params = params[idx : idx + 6]
            idx += 6

            for i in range(0, len(active) - 1, 2):
                _rot_rot_cnot(pool_params, [active[i], active[i + 1]])

            active = active[::2]

        _rot_rot_cnot(params[idx : idx + 6], [active[0], active[1]])

        return qml.expval(qml.PauliZ(active[0]))

    return qnode