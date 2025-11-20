"""
QNN architectures:
 - Simple (stacked two-qubit blocks)
 - TTN (binary tree)
 - MERA-like (disentangler + isometry layers)
 - QCNN (conv + pooling, shared params per conv index)

Key design decisions:
 - Generic two-qubit gate = two single-qubit gates (qml.Rot) + single CNOT
 - Prefer single CNOT per two-qubit block to be NISQ-friendly
 - Single-qubit output measured; Simple uses |-> init and X-basis measurement,
   other architectures use Z-basis measurement.
 - When running with shots, it returns majority-vote class (sampling).
 - QNodes use interface="torch" so integration with PyTorch training is straightforward.
 - All architectures accept `shots` argument: if shots is None or 0, it returns
   analytic expectation (float). If shots>0, it returns sampled measurements and a
   majority-vote bit (0/1) via helper functions.
"""

from typing import List, Tuple, Optional
import math
import torch
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


# Generic two-qubit block
def two_qubit_block(params: pnp.ndarray, wires: List[int]):
    """
    Generic two-qubit block used by all architectures, implemented as:
      U(params[0:3]) on wires[0]  (qml.Rot)
      U(params[3:6]) on wires[1]
      CNOT(wires[0], wires[1])

    params shape: (6,)  (3 params per single-qubit Rot)
    wires: [a, b]
    """
    a, b = wires[0], wires[1]
    # first single qubit gate (Rot)
    qml.Rot(params[0], params[1], params[2], wires=a)
    # second single qubit gate
    qml.Rot(params[3], params[4], params[5], wires=b)
    # single CNOT entangler (NISQ-friendly)
    qml.CNOT(wires=[a, b])


# Helpers: measurement
def _sample_majority_result(samples: pnp.ndarray, measure_basis: str = "Z") -> int:
    """
    samples: output of qnode.samples() in computational basis (shape: (shots, n_wires))
    measure_basis: "Z" or "X"
    For X-basis majority we assume H was applied before measuring Z (so samples already in Z basis),
    and we interpret 0 as |0>, 1 as |1>. Majority vote returns 0 or 1.
    """
    # collapse each shot to the measured output bit of the result qubit (we assume qnode returns all wires)
    # For convenience, user QNodes below will return only the result-qubit samples.
    if samples.ndim == 1:
        # shape (shots,)
        bits = np.array(samples)
    else:
        # if samples is (shots, 1)
        bits = np.array(samples[:, 0])
    counts = (bits == 0).sum(), (bits == 1).sum()
    # majority: if ties, prefer 0 (arbitrary but deterministic)
    return 0 if counts[0] >= counts[1] else 1



# SIMPLE architecture
def build_simple_qnn(n_qubits: int, n_layers: int, dev, result_wire: Optional[int] = None, shots: Optional[int] = None):
    """
    Simple architecture: stack of n_layers; each layer applies two-qubit blocks
    across nearest neighbors (0-1,1-2,... optionally ring to connect last-first).
    The paper states the result qubit is initialized in |-> and measured in X-basis.

    params shape (n_layers, n_qubits-1, 6) if pairing non-overlapping, but for simplicity
    we will pair (0,1), (2,3), ... and if odd-qubit, last pair uses (n-2,n-1).
    We'll accept params as a flattened tensor and reshape inside qnode.
    """

    if result_wire is None:
        result_wire = n_qubits - 1  # paper: last qubit as result

    # QNode
    @qml.qnode(dev, interface="torch", diff_method="backprop" if shots in (None, 0) else None, shots=shots)
    def circuit(x: torch.Tensor, params: torch.Tensor):
        """
        x: angles array of length n_qubits (in radians) for RX encoding (paper uses RX)
        params: flattened parameters tensor; expected shape (n_layers, n_pairs, 6)
        """
        # 1) Encode: RX on each qubit
        for i in range(n_qubits):
            qml.RX(x[i].item(), wires=i)

        # Initialize result qubit in |-> : X then H (|0> -> |1> -> H|1> = |->)
        qml.PauliX(wires=result_wire)
        qml.Hadamard(wires=result_wire)

        # 2) Layers: apply two-qubit blocks
        # reshape params to (n_layers, n_pairs, 6)
        # we choose pairs (0,1),(2,3),...
        n_pairs = max(1, n_qubits // 2)
        p_arr = params.reshape((n_layers, n_pairs, 6))
        for l in range(n_layers):
            for p_idx in range(n_pairs):
                a = 2 * p_idx
                b = min(2 * p_idx + 1, n_qubits - 1)
                two_qubit_block(p_arr[l, p_idx, :], wires=[a, b])
            # optional ring entanglement: connect last with first
            if n_qubits > 2:
                # small extra entangler to increase connectivity (paper's Simple stacks two-qubit blocks)
                qml.CNOT(wires=[n_qubits - 1, 0])

        # 3) Measurement in X-basis for Simple: measure PauliX on result qubit
        if shots and shots > 0:
            # sample in computational basis after rotating basis: to measure X, apply H then sample Z
            qml.Hadamard(wires=result_wire)
            return qml.sample(qml.PauliZ(wires=result_wire))
        else:
            return qml.expval(qml.PauliX(result_wire))

    return circuit


# TTN architecture
def build_ttn_qnn(n_qubits: int, dev, shots: Optional[int] = None):
    """
    Binary Tree Tensor Network (TTN):
    - At each layer, pair qubits (0,1),(2,3),... apply two_qubit_block.
    - Keep the 'left' qubit of each pair as representative and proceed to next layer.
    - Repeat until single qubit remains.
    - Final measurement in Z basis (paper).
    """

    # Compute number of layers needed: ceil(log2(n_qubits))
    import math
    n_layers = math.ceil(math.log2(n_qubits))

    @qml.qnode(dev, interface="torch", diff_method="backprop" if shots in (None, 0) else None, shots=shots)
    def circuit(x: torch.Tensor, params: torch.Tensor):
        """
        params expected flattened shape: sum over layers of (n_pairs_layer * 6)
        For simplicity, we pass params as shape (n_total_pairs, 6) flattened and consume sequentially.
        """
        # Encode (RX)
        for i in range(n_qubits):
            qml.RX(x[i].item(), wires=i)

        # We'll perform in-place pairwise two-qubit blocks. We'll track active qubits indices.
        active = list(range(n_qubits))
        p_idx = 0
        params_np = params.reshape(-1, 6)
        # perform layers
        while len(active) > 1:
            next_active = []
            for i in range(0, len(active), 2):
                if i + 1 < len(active):
                    a = active[i]
                    b = active[i + 1]
                    two_qubit_block(params_np[p_idx], wires=[a, b])
                    p_idx += 1
                    # keep a as representative (we choose left)
                    next_active.append(a)
                else:
                    # odd last qubit remains for next layer
                    next_active.append(active[i])
            active = next_active

        # measure final remaining qubit (active[0]) in Z basis
        out_wire = active[0]
        if shots and shots > 0:
            return qml.sample(qml.PauliZ(wires=out_wire))
        else:
            return qml.expval(qml.PauliZ(out_wire))

    return circuit


# MERA architecture
def build_mera_qnn(n_qubits: int, n_layers: int, dev, shots: Optional[int] = None):
    """
    MERA: alternating disentangler and isometry layers.
    Implementation pattern per layer:
      - Disentanglers: apply two_qubit_block on pairs (1,2),(3,4),... (shifted)
      - Isometries: apply two_qubit_block on pairs (0,1),(2,3),...
    Repeat for n_layers. Final measurement in Z basis.
    """

    @qml.qnode(dev, interface="torch", diff_method="backprop" if shots in (None, 0) else None, shots=shots)
    def circuit(x: torch.Tensor, params: torch.Tensor):
        # params shape: (n_layers, 2 * n_pairs_layer, 6) flattened; for simplicity accept reshape
        for i in range(n_qubits):
            qml.RX(x[i].item(), wires=i)

        # reshape param array heuristically:
        # We'll assume params shape (n_layers, n_qubits, 6) and use a sliding window
        p_arr = params.reshape((n_layers, n_qubits, 6))
        for l in range(n_layers):
            # disentanglers on (1,2),(3,4),...
            for i in range(1, n_qubits - 1, 2):
                two_qubit_block(p_arr[l, i], wires=[i, i + 1])
            # isometries on (0,1),(2,3),...
            for i in range(0, n_qubits - 1, 2):
                two_qubit_block(p_arr[l, i], wires=[i, i + 1])

        # final measurement on qubit 0 (choose representative) in Z basis
        out_wire = 0
        if shots and shots > 0:
            return qml.sample(qml.PauliZ(wires=out_wire))
        else:
            return qml.expval(qml.PauliZ(out_wire))

    return circuit


# QCNN architecture
def build_qcnn_qnn(n_qubits: int, dev, shots: Optional[int] = None):
    """
    QCNN: convolution (two-qubit blocks applied to adjacent pairs with shared params per index)
    followed by pooling (we compress pair into first qubit using a small circuit).
    Repeat until 1 qubit remains. Measure in Z basis.

    Parameter sharing:
      - conv_params: per conv-index (shared across repetitions)
      - pool_params: per pooling gate
    For simplicity of API we accept params flattened as needed and interpret them.
    """

    @qml.qnode(dev, interface="torch", diff_method="backprop" if shots in (None, 0) else None, shots=shots)
    def circuit(x: torch.Tensor, params: torch.Tensor):
        # For QCNN we expect params shaped (levels, max_pairs_per_level, 6) ideally.
        # For simplicity, reshape to (L, floor(n_qubits/2), 6) where L = floor(log2(n_qubits))
        for i in range(n_qubits):
            qml.RX(x[i].item(), wires=i)

        # We'll compute levels until we have 1 qubit
        active = list(range(n_qubits))
        params_np = params.reshape(-1, 6)
        p_idx = 0

        # While reduce until single qubit remains
        while len(active) > 1:
            # convolution: apply two-qubit blocks to adjacent pairs (0,1),(2,3),...
            next_active = []
            for i in range(0, len(active), 2):
                a = active[i]
                if i + 1 < len(active):
                    b = active[i + 1]
                    # Use shared parameters per conv index: here we pull next params entry
                    two_qubit_block(params_np[p_idx], wires=[a, b])
                    p_idx += 1
                    # pooling: compress into 'a' by applying small compressing gate (CNOT + Rot)
                    qml.CNOT(wires=[a, b])
                    # apply single qubit rot on 'a' using next params (if available)
                    # fallback to identity if params exhausted
                    if p_idx < len(params_np):
                        p = params_np[p_idx]
                        # use first 3 entries for single-qubit rot
                        qml.Rot(p[0], p[1], p[2], wires=a)
                        p_idx += 1
                    next_active.append(a)
                else:
                    # odd leftover
                    next_active.append(a)
            active = next_active

        # final measurement on remaining qubit in Z basis
        out_wire = active[0]
        if shots and shots > 0:
            return qml.sample(qml.PauliZ(wires=out_wire))
        else:
            return qml.expval(qml.PauliZ(out_wire))

    return circuit


# -----------------------
# Utility wrappers for usage
# -----------------------
def run_with_sampling(qnode, x, params, shots: int = 100):
    """
    Run a qnode with sampling and perform majority vote to return a single bit (0/1).
    qnode should be constructed with shots=shots.
    This helper handles the basis-change required for X-basis measurement if user expects it.
    """
    res = qnode(x, params)
    # qnode sample return type may be an array of shape (shots,) or (shots, 1)
    samps = pnp.array(res)
    # majority
    zeros = (samps == 0).sum()
    ones = (samps == 1).sum()
    return 0 if zeros >= ones else 1


# -----------------------
# Example of how to build devices and circuits
# -----------------------
if __name__ == "__main__":
    # quick smoke test (analytic)
    dev = qml.device("default.qubit", wires=8, shots=None)

    n_qubits = 8
    n_layers = 2

    # Simple
    simple = build_simple_qnn(n_qubits, n_layers, dev, result_wire=7, shots=None)
    x = torch.rand(n_qubits) * math.pi
    # params shape for simple: (n_layers * n_pairs * 6,)
    n_pairs = n_qubits // 2
    params = torch.randn(n_layers * n_pairs * 6, requires_grad=True)
    out = simple(x, params)
    print("Simple (analytic) output:", out)

    # TTN
    ttn = build_ttn_qnn(n_qubits, dev, shots=None)
    # need params: sum of pairs per layer * 6 -> for 8 qubits, pairs=4+2+1 = 7 pairs -> 7*6 params
    params_ttn = torch.randn(7 * 6, requires_grad=True)
    out2 = ttn(x, params_ttn)
    print("TTN (analytic) output:", out2)

    # MERA
    dev2 = qml.device("default.qubit", wires=8, shots=None)
    mera = build_mera_qnn(n_qubits, n_layers=2, dev=dev2, shots=None)
    params_mera = torch.randn(n_layers * n_qubits * 6, requires_grad=True)
    out3 = mera(x, params_mera)
    print("MERA (analytic) output:", out3)

    # QCNN
    dev3 = qml.device("default.qubit", wires=8, shots=None)
    qcnn = build_qcnn_qnn(n_qubits, dev3, shots=None)
    params_qcnn = torch.randn(20 * 6, requires_grad=True)  # arbitrary size
    out4 = qcnn(x, params_qcnn)
    print("QCNN (analytic) output:", out4)
