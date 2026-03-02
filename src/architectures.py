# src/architectures.py
"""
QNN architectures used in:
"Network Anomaly Detection Using Quantum Neural Networks on Noisy Quantum Computers"

Updates:
- diff_method="best" (avoids crash when evaluating with shots/noise)
- Simple architecture now supports paper layer types:
  ZZXXYY, ZZXX, XXYY, ZZYY  (best reported: "XY" ~= XXYY)
- Optional noise injection via src/noise.py after each gate + readout noise
"""

from typing import Callable, Optional, List
import math

import pennylane as qml

# Lazy import to avoid hard dependency issues if noise not used
try:
    import noise as noise_module
except Exception:
    noise_module = None


# -------------------------
# Helpers
# -------------------------
def _apply_noise(noise_model, wires: List[int]):
    if noise_model is None:
        return
    if noise_module is None:
        raise ImportError("noise.py not found but noise_model was provided.")
    noise_module.apply_gate_noise(noise_model, wires)


def _two_qubit_block_rotcnot(params, wires: List[int], noise_model=None):
    """
    Generic 2-qubit block: Rot on each qubit + CNOT.
    Params: 6 scalars -> Rot(3) on a + Rot(3) on b
    """
    a, b = wires
    qml.Rot(params[0], params[1], params[2], wires=a)
    _apply_noise(noise_model, [a])
    qml.Rot(params[3], params[4], params[5], wires=b)
    _apply_noise(noise_model, [b])
    qml.CNOT(wires=[a, b])
    _apply_noise(noise_model, [a, b])


def _two_qubit_block_paper_layer(params, wires: List[int], layer_type: str, noise_model=None):
    """
    Paper-like 2-qubit interaction layer using IsingXX/IsingYY/IsingZZ.
    We still keep NISQ-friendly structure and allow noise after each gate.

    layer_type in {"ZZXXYY","ZZXX","XXYY","ZZYY"}.
    Parameter count depends on type:
      ZZXXYY: 3 params (zz, xx, yy)
      ZZXX:   2 params (zz, xx)
      XXYY:   2 params (xx, yy)   <-- "XY" in paper corresponds to XXYY
      ZZYY:   2 params (zz, yy)
    """
    a, b = wires
    lt = layer_type.upper()

    # We implement as Ising gates + (optional) local rotations could be added,
    # but paper focuses on these interaction layers.
    # Keep exact order as name: ZZ then XX then YY (or subset).
    idx = 0

    if lt in ("ZZXXYY", "ZZXX", "ZZYY"):
        qml.IsingZZ(params[idx], wires=[a, b])
        _apply_noise(noise_model, [a, b])
        idx += 1

    if lt in ("ZZXXYY", "ZZXX", "XXYY"):
        qml.IsingXX(params[idx], wires=[a, b])
        _apply_noise(noise_model, [a, b])
        idx += 1

    if lt in ("ZZXXYY", "XXYY", "ZZYY"):
        qml.IsingYY(params[idx], wires=[a, b])
        _apply_noise(noise_model, [a, b])
        idx += 1


def _params_per_pair_for_simple(layer_type: str) -> int:
    lt = layer_type.upper()
    if lt == "ZZXXYY":
        return 3
    if lt in ("ZZXX", "XXYY", "ZZYY"):
        return 2
    raise ValueError(f"Unknown layer_type '{layer_type}'. Use: ZZXXYY, ZZXX, XXYY, ZZYY.")


# -------------------------
# Builders
# -------------------------
def build_simple_qnn(
    n_qubits: int,
    n_layers: int,
    dev,
    result_wire: int = 7,
    shots: Optional[int] = None,
    noise_model=None,
    layer_type: str = "XXYY",   # paper "XY" corresponds to XXYY
) -> Callable:
    """
    Simple architecture:
    - RX encoding on each wire
    - result qubit initialized in |-> (H then Z)
    - Apply paper interaction layers across pairs per layer
    - Measure X on result_wire

    layer_type controls the entangling pattern (paper).
    """

    p_per_pair = _params_per_pair_for_simple(layer_type)
    n_pairs = max(1, n_qubits // 2)
    expected_params = n_layers * n_pairs * p_per_pair

    @qml.qnode(dev, interface="torch", diff_method="best")
    def qnode(x, params):
        # Encode
        for i in range(n_qubits):
            qml.RX(x[i], wires=i)
            _apply_noise(noise_model, [i])

        # Init result wire in |-> : H then Z
        qml.Hadamard(wires=result_wire)
        _apply_noise(noise_model, [result_wire])
        qml.PauliZ(wires=result_wire)
        _apply_noise(noise_model, [result_wire])

        # Safety check (helps debugging)
        # (Do not raise hard in graph mode too often; but here is okay.)
        if hasattr(params, "shape") and int(params.shape[0]) != expected_params:
            raise ValueError(
                f"Simple QNN expected {expected_params} params for layer_type={layer_type}, "
                f"got {int(params.shape[0])}."
            )

        idx = 0
        for _ in range(n_layers):
            for a in range(0, n_qubits - 1, 2):
                block_params = params[idx: idx + p_per_pair]
                idx += p_per_pair
                _two_qubit_block_paper_layer(block_params, [a, a + 1], layer_type=layer_type, noise_model=noise_model)

        # Readout noise
        if noise_model is not None:
            noise_module.apply_measurement_noise(noise_model, result_wire)

        return qml.expval(qml.PauliX(result_wire))

    return qnode


def build_ttn_qnn(n_qubits: int, dev, shots: Optional[int] = None, noise_model=None) -> Callable:
    """
    TTN-like pairing reduction. Output wire measured in Z.
    Uses generic Rot+Rot+CNOT blocks (as in your original).
    """

    @qml.qnode(dev, interface="torch", diff_method="best")
    def qnode(x, params):
        for i in range(n_qubits):
            qml.RX(x[i], wires=i)
            _apply_noise(noise_model, [i])

        idx = 0
        active = list(range(n_qubits))
        while len(active) > 1:
            new_active = []
            for i in range(0, len(active) - 1, 2):
                a, b = active[i], active[i + 1]
                block_params = params[idx:idx + 6]
                idx += 6
                _two_qubit_block_rotcnot(block_params, [a, b], noise_model=noise_model)
                new_active.append(a)
            if len(active) % 2 == 1:
                new_active.append(active[-1])
            active = new_active

        out_wire = active[0]
        if noise_model is not None:
            noise_module.apply_measurement_noise(noise_model, out_wire)
        return qml.expval(qml.PauliZ(out_wire))

    return qnode


def build_mera_qnn(n_qubits: int, n_layers: int, dev, shots: Optional[int] = None, noise_model=None) -> Callable:
    """
    MERA-style (simplified). Output wire measured in Z.
    Uses Rot+Rot+CNOT blocks.
    """

    @qml.qnode(dev, interface="torch", diff_method="best")
    def qnode(x, params):
        for i in range(n_qubits):
            qml.RX(x[i], wires=i)
            _apply_noise(noise_model, [i])

        idx = 0
        for _ in range(n_layers):
            for a in range(0, n_qubits - 1, 2):
                block_params = params[idx:idx + 6]
                idx += 6
                _two_qubit_block_rotcnot(block_params, [a, a + 1], noise_model=noise_model)
            for a in range(1, n_qubits - 1, 2):
                block_params = params[idx:idx + 6]
                idx += 6
                _two_qubit_block_rotcnot(block_params, [a, a + 1], noise_model=noise_model)

        out_wire = 0
        if noise_model is not None:
            noise_module.apply_measurement_noise(noise_model, out_wire)
        return qml.expval(qml.PauliZ(out_wire))

    return qnode


def build_qcnn_qnn(n_qubits: int, dev, shots: Optional[int] = None, noise_model=None) -> Callable:
    """
    QCNN-style pooling (simplified). Output wire measured in Z.
    Uses Rot+Rot+CNOT blocks; 2 blocks per pair as in your code.
    """

    @qml.qnode(dev, interface="torch", diff_method="best")
    def qnode(x, params):
        for i in range(n_qubits):
            qml.RX(x[i], wires=i)
            _apply_noise(noise_model, [i])

        idx = 0
        active = list(range(n_qubits))
        while len(active) > 1:
            for i in range(0, len(active) - 1, 2):
                a, b = active[i], active[i + 1]
                block_params = params[idx:idx + 6]
                idx += 6
                _two_qubit_block_rotcnot(block_params, [a, b], noise_model=noise_model)

                block_params2 = params[idx:idx + 6]
                idx += 6
                _two_qubit_block_rotcnot(block_params2, [a, b], noise_model=noise_model)

            active = active[::2]

        out_wire = active[0]
        if noise_model is not None:
            noise_module.apply_measurement_noise(noise_model, out_wire)
        return qml.expval(qml.PauliZ(out_wire))

    return qnode