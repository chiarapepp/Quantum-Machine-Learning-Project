# src/architectures.py
"""
Quantum neural network architectures inspired by:

"Network Anomaly Detection Using Quantum Neural Networks on Noisy Quantum Computers"

Implemented design choices:
- TTN, MERA, and QCNN use a generic two-qubit block built from
  two generic one-qubit rotations followed by one CNOT.
- The Simple architecture uses a dedicated result qubit initialized in |->,
  applies repeated Pauli interaction sweeps along the chain
  [result, 0, 1, ..., n_qubits - 1], and measures the result qubit in the X basis.
- All other architectures measure a single output qubit in the Z basis.
- A utility is provided to visualize circuits with qml.draw_mpl.
"""

from typing import Callable, List, Optional, Tuple
import math

import pennylane as qml


# ---------------------------------------------------------------------
# Primitive blocks
# ---------------------------------------------------------------------
def _two_qubit_block_rot_cnot(params, wires: List[int]) -> None:
    """
    Generic two-qubit block used in TTN, MERA, and QCNN.

    The block matches the paper description:
    - one generic one-qubit gate on the first wire
    - one generic one-qubit gate on the second wire
    - one CNOT

    A generic one-qubit gate is implemented as qml.Rot(phi, theta, omega).

    Expected parameter layout:
        params[0:3] -> first wire
        params[3:6] -> second wire
    """
    if len(wires) != 2:
        raise ValueError("A two-qubit block requires exactly two wires.")
    if len(params) != 6:
        raise ValueError("A Rot+CNOT block requires exactly 6 parameters.")

    a, b = wires

    qml.Rot(params[0], params[1], params[2], wires=a)
    qml.Rot(params[3], params[4], params[5], wires=b)
    qml.CNOT(wires=[a, b])


def _simple_layer_terms(layer_type: str) -> List[str]:
    """
    Return the ordered Pauli sweeps used by one Simple repetition.

    Supported layer types:
        - "ZZXXYY" -> ["ZZ", "XX", "YY"]
        - "ZZXX"   -> ["ZZ", "XX"]
        - "XXYY"   -> ["XX", "YY"]
        - "ZZYY"   -> ["ZZ", "YY"]
    """
    lt = layer_type.upper()

    if lt == "ZZXXYY":
        return ["ZZ", "XX", "YY"]
    if lt == "ZZXX":
        return ["ZZ", "XX"]
    if lt == "XXYY":
        return ["XX", "YY"]
    if lt == "ZZYY":
        return ["ZZ", "YY"]

    raise ValueError(
        f"Unknown layer_type '{layer_type}'. "
        "Supported values are: ZZXXYY, ZZXX, XXYY, ZZYY."
    )


def _simple_chain_pairs(n_qubits: int, result_wire: int) -> List[List[int]]:
    """
    Build the ordered nearest-neighbor chain used by the Simple architecture.

    The chain is:
        [result_wire, 0, 1, ..., n_qubits - 1]

    The returned pairs are:
        (result, 0), (0, 1), (1, 2), ..., (n_qubits - 2, n_qubits - 1)
    """
    chain = [result_wire] + list(range(n_qubits))
    return [[chain[i], chain[i + 1]] for i in range(len(chain) - 1)]


def _apply_simple_pauli_sweep(
    term: str,
    pairs: List[List[int]],
    params,
) -> None:
    """
    Apply one full Pauli sweep across the Simple architecture chain.

    Each sweep applies the same Pauli interaction type to every adjacent pair
    in the ordered chain. The number of parameters must match the number of pairs.
    """
    if len(params) != len(pairs):
        raise ValueError(
            f"Simple sweep expects {len(pairs)} parameters, but got {len(params)}."
        )

    for k, (a, b) in enumerate(pairs):
        theta = params[k]

        if term == "XX":
            qml.IsingXX(theta, wires=[a, b])
        elif term == "YY":
            qml.IsingYY(theta, wires=[a, b])
        elif term == "ZZ":
            qml.IsingZZ(theta, wires=[a, b])
        else:
            raise ValueError(f"Unknown Simple sweep term '{term}'.")


# ---------------------------------------------------------------------
# Parameter counting helpers
# ---------------------------------------------------------------------
def simple_num_params(n_qubits: int, n_layers: int, layer_type: str = "XXYY") -> int:
    """
    Return the number of trainable parameters for the Simple architecture.

    n_qubits is the number of encoded feature qubits.
    The result qubit is separate and does not contribute input features.

    One Simple repetition contains:
    - one full sweep per Pauli term in layer_type
    - one parameter per nearest-neighbor pair in the chain
      [result, 0, 1, ..., n_qubits - 1]

    The chain has exactly n_qubits adjacent pairs.
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be at least 1.")
    if n_layers < 1:
        raise ValueError("n_layers must be at least 1.")

    n_terms = len(_simple_layer_terms(layer_type))
    n_pairs = n_qubits
    return n_layers * n_terms * n_pairs


def ttn_num_blocks(n_qubits: int) -> int:
    """Return the number of generic two-qubit blocks in the TTN."""
    if n_qubits < 2:
        raise ValueError("TTN requires at least 2 qubits.")

    blocks = 0
    active = n_qubits

    while active > 1:
        blocks += active // 2
        active = (active // 2) + (active % 2)

    return blocks


def ttn_num_params(n_qubits: int) -> int:
    """Return the number of trainable parameters for the TTN."""
    return 6 * ttn_num_blocks(n_qubits)


def mera_num_blocks(n_qubits: int, n_scales: int) -> int:
    """
    Return the number of generic two-qubit blocks in the MERA-style circuit.

    Each scale applies:
    - one layer on odd neighboring pairs
    - one layer on even neighboring pairs
    Then the active set is reduced by keeping every other wire.
    """
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

    return blocks


def mera_num_params(n_qubits: int, n_scales: int) -> int:
    """Return the number of trainable parameters for the MERA-style circuit."""
    return 6 * mera_num_blocks(n_qubits, n_scales)


def qcnn_num_scales(n_qubits: int) -> int:
    """
    Return the number of convolution/pooling scales before the final classifier.

    The QCNN applies shared convolution and pooling blocks while the number
    of active wires is greater than 2, then applies one final classifier block.
    """
    if n_qubits < 2:
        raise ValueError("QCNN requires at least 2 qubits.")

    scales = 0
    active = n_qubits

    while active > 2:
        scales += 1
        active = math.ceil(active / 2)

    return scales


def qcnn_num_params(n_qubits: int) -> int:
    """
    Return the number of trainable parameters for the QCNN.

    Parameter sharing is stage-wise:
    - one shared convolution block per scale
    - one shared pooling block per scale
    - one final classifier block on the last two active wires
    """
    return 6 * (2 * qcnn_num_scales(n_qubits) + 1)


def _validate_flat_param_length(params, expected: int, name: str) -> None:
    """Validate a flat parameter tensor or array when shape information is available."""
    if not hasattr(params, "shape"):
        return

    if len(params.shape) != 1:
        raise ValueError(
            f"{name} expects a flat 1D parameter tensor with length {expected}, "
            f"but got shape {tuple(params.shape)}."
        )

    actual = int(params.shape[0])
    if actual != expected:
        raise ValueError(f"{name} expects {expected} parameters, but got {actual}.")


# ---------------------------------------------------------------------
# Architecture builders
# ---------------------------------------------------------------------
def build_simple_qnn(
    n_qubits: int,
    n_layers: int,
    dev,
    result_wire: Optional[int] = None,
    shots: Optional[int] = None,
    layer_type: str = "XXYY",
) -> Callable:
    """
    Build the Simple architecture.

    Architectural assumptions:
    - n_qubits is the number of encoded feature qubits.
    - The result qubit is a separate wire and defaults to wire n_qubits.
    - The device must therefore expose at least n_qubits + 1 wires when
      result_wire is not provided.
    - The result qubit is initialized in |-> and measured in the X basis.
    - Each repetition applies ordered Pauli sweeps along the chain
      [result, 0, 1, ..., n_qubits - 1].
    """
    del shots

    if n_qubits < 1:
        raise ValueError("Simple architecture requires at least 1 feature qubit.")
    if n_layers < 1:
        raise ValueError("Simple architecture requires at least 1 layer.")

    result_wire = n_qubits if result_wire is None else int(result_wire)

    if 0 <= result_wire < n_qubits:
        raise ValueError(
            "result_wire must be separate from the encoded feature wires "
            "0..n_qubits-1."
        )

    expected_params = simple_num_params(n_qubits, n_layers, layer_type)
    layer_terms = _simple_layer_terms(layer_type)
    chain_pairs = _simple_chain_pairs(n_qubits, result_wire)
    params_per_sweep = len(chain_pairs)

    @qml.qnode(dev, interface="torch", diff_method="best")
    def qnode(x, params):
        _validate_flat_param_length(params, expected_params, "build_simple_qnn")

        if hasattr(x, "shape") and len(x.shape) > 0 and int(x.shape[0]) != n_qubits:
            raise ValueError(
                f"Simple architecture expects {n_qubits} input features, "
                f"but got {int(x.shape[0])}."
            )

        for wire in range(n_qubits):
            qml.RX(x[wire], wires=wire)

        qml.Hadamard(wires=result_wire)
        qml.PauliZ(wires=result_wire)

        idx = 0
        for _ in range(n_layers):
            for term in layer_terms:
                sweep_params = params[idx : idx + params_per_sweep]
                idx += params_per_sweep
                _apply_simple_pauli_sweep(
                    term=term,
                    pairs=chain_pairs,
                    params=sweep_params,
                )

        return qml.expval(qml.PauliX(result_wire))

    return qnode


def build_ttn_qnn(
    n_qubits: int,
    dev,
    shots: Optional[int] = None,
) -> Callable:
    """
    Build a TTN classifier.

    The circuit repeatedly applies unique generic two-qubit blocks to adjacent
    active pairs and keeps the first wire of each pair as the next active wire.
    The final active wire is measured in the Z basis.
    """
    del shots

    if n_qubits < 2:
        raise ValueError("TTN requires at least 2 qubits.")

    expected_params = ttn_num_params(n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="best")
    def qnode(x, params):
        _validate_flat_param_length(params, expected_params, "build_ttn_qnn")

        if hasattr(x, "shape") and len(x.shape) > 0 and int(x.shape[0]) != n_qubits:
            raise ValueError(
                f"TTN expects {n_qubits} input features, but got {int(x.shape[0])}."
            )

        for wire in range(n_qubits):
            qml.RX(x[wire], wires=wire)

        idx = 0
        active = list(range(n_qubits))

        while len(active) > 1:
            new_active = []

            for i in range(0, len(active) - 1, 2):
                a = active[i]
                b = active[i + 1]

                block_params = params[idx : idx + 6]
                idx += 6

                _two_qubit_block_rot_cnot(block_params, [a, b])
                new_active.append(a)

            if len(active) % 2 == 1:
                new_active.append(active[-1])

            active = new_active

        out_wire = active[0]
        return qml.expval(qml.PauliZ(out_wire))

    return qnode


def build_mera_qnn(
    n_qubits: int,
    n_scales: int,
    dev,
    shots: Optional[int] = None,
) -> Callable:
    """
    Build a MERA-style classifier.

    Each scale applies:
    - one layer on odd neighboring pairs
    - one layer on even neighboring pairs
    Then the active set is reduced by keeping every other wire.

    All two-qubit gates are implemented as Rot + Rot + CNOT blocks.
    The output is measured in the Z basis.
    """
    del shots

    if n_qubits < 2:
        raise ValueError("MERA requires at least 2 qubits.")
    if n_scales < 1:
        raise ValueError("MERA requires at least 1 scale.")

    expected_params = mera_num_params(n_qubits, n_scales)

    @qml.qnode(dev, interface="torch", diff_method="best")
    def qnode(x, params):
        _validate_flat_param_length(params, expected_params, "build_mera_qnn")

        if hasattr(x, "shape") and len(x.shape) > 0 and int(x.shape[0]) != n_qubits:
            raise ValueError(
                f"MERA expects {n_qubits} input features, but got {int(x.shape[0])}."
            )

        for wire in range(n_qubits):
            qml.RX(x[wire], wires=wire)

        idx = 0
        active = list(range(n_qubits))

        for _ in range(n_scales):
            if len(active) < 2:
                break

            for i in range(1, len(active) - 1, 2):
                a = active[i]
                b = active[i + 1]

                block_params = params[idx : idx + 6]
                idx += 6

                _two_qubit_block_rot_cnot(block_params, [a, b])

            for i in range(0, len(active) - 1, 2):
                a = active[i]
                b = active[i + 1]

                block_params = params[idx : idx + 6]
                idx += 6

                _two_qubit_block_rot_cnot(block_params, [a, b])

            active = active[::2]

        out_wire = active[0]
        return qml.expval(qml.PauliZ(out_wire))

    return qnode


def build_qcnn_qnn(
    n_qubits: int,
    dev,
    shots: Optional[int] = None,
) -> Callable:
    """
    Build a QCNN classifier with shared parameters across repeated blocks.

    At each scale:
    - one shared convolution block is applied to every adjacent active pair
    - one shared pooling block is applied to every adjacent active pair
    - every second wire is kept for the next scale

    After the hierarchy reduces the active set to two wires, a final classifier
    block is applied once. The final output qubit is measured in the Z basis.

    Parameter sharing is implemented at the stage level:
    all repeated blocks within the same stage use identical parameters.
    """
    del shots

    if n_qubits < 2:
        raise ValueError("QCNN requires at least 2 qubits.")

    expected_params = qcnn_num_params(n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="best")
    def qnode(x, params):
        _validate_flat_param_length(params, expected_params, "build_qcnn_qnn")

        if hasattr(x, "shape") and len(x.shape) > 0 and int(x.shape[0]) != n_qubits:
            raise ValueError(
                f"QCNN expects {n_qubits} input features, but got {int(x.shape[0])}."
            )

        for wire in range(n_qubits):
            qml.RX(x[wire], wires=wire)

        idx = 0
        active = list(range(n_qubits))

        while len(active) > 2:
            conv_params = params[idx : idx + 6]
            idx += 6

            for i in range(0, len(active) - 1, 2):
                a = active[i]
                b = active[i + 1]
                _two_qubit_block_rot_cnot(conv_params, [a, b])

            pool_params = params[idx : idx + 6]
            idx += 6

            for i in range(0, len(active) - 1, 2):
                a = active[i]
                b = active[i + 1]
                _two_qubit_block_rot_cnot(pool_params, [a, b])

            active = active[::2]

        final_params = params[idx : idx + 6]
        a, b = active[0], active[1]
        _two_qubit_block_rot_cnot(final_params, [a, b])

        out_wire = a
        return qml.expval(qml.PauliZ(out_wire))

    return qnode


# ---------------------------------------------------------------------
# Drawing utilities
# ---------------------------------------------------------------------
def make_demo_inputs(
    arch: str,
    n_qubits: int,
    n_layers: int = 1,
    n_scales: int = 1,
    layer_type: str = "XXYY",
    interface: str = "torch",
):
    """
    Create zero-valued demo inputs for circuit visualization.

    Returns:
        x, params
    """
    if interface == "torch":
        import torch

        x = torch.zeros(n_qubits)

        if arch == "simple":
            params = torch.zeros(simple_num_params(n_qubits, n_layers, layer_type))
        elif arch == "ttn":
            params = torch.zeros(ttn_num_params(n_qubits))
        elif arch == "mera":
            params = torch.zeros(mera_num_params(n_qubits, n_scales))
        elif arch == "qcnn":
            params = torch.zeros(qcnn_num_params(n_qubits))
        else:
            raise ValueError(f"Unknown architecture '{arch}'.")

        return x, params

    raise ValueError("Currently only interface='torch' is supported.")


def draw_qnn(
    qnode: Callable,
    x,
    params,
    figsize: Tuple[float, float] = (14, 6),
    expansion_strategy: str = "device",
):
    """
    Draw a QNode with qml.draw_mpl.

    Example:
        fig, ax = draw_qnn(qnode, x, params)
    """
    fig, ax = qml.draw_mpl(qnode, expansion_strategy=expansion_strategy)(x, params)
    fig.set_size_inches(*figsize)
    return fig, ax