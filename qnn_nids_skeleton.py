"""qnn_nids_skeleton.py

Implements the 'Simple' QNN architecture described in the paper:

- Each classical feature → single qubit via Rx(angle) encoding.
- Eight feature qubits + one result qubit (result measured in X-basis).
- Variational layers built from two-qubit Pauli rotations (Rxx / Ryy).
- Certainty factor computation helper.

This file focuses on building circuits and a TFQ Keras model. It does
not train a full model by default; it provides helpers to construct the
input circuits (state preparation), the parameterized variational
circuit, a Keras model using tfq.layers.PQC, and a small demo in
``__main__`` that verifies the shapes.

Note: This implementation uses Cirq/TensorFlow Quantum (TFQ). The
runtime must have those packages installed to actually run the
demo/train the model.
"""

import numpy as np
import cirq
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq
from typing import List, Tuple


# --- Configuration ---
N_FEATURE_QUBITS = 8
# feature qubits are indexed 0..7, result qubit is index 8
FEATURE_QUBITS = [cirq.GridQubit(0, i) for i in range(N_FEATURE_QUBITS)]
RESULT_QUBIT = cirq.GridQubit(0, N_FEATURE_QUBITS)


def _rx_encode_circuit_row(angles: List[float]) -> cirq.Circuit:
    """Create a Cirq circuit that prepares the input state for one sample.

    - Applies Rx(theta) on each feature qubit.
    - Prepares the result qubit in the |-> state (X then H on |0>).

    Args:
        angles: list/array of length N_FEATURE_QUBITS containing angles in
                radians to apply with Rx on feature qubits.

    Returns:
        cirq.Circuit preparing the encoded state.
    """
    if len(angles) != N_FEATURE_QUBITS:
        raise ValueError(f"angles length must be {N_FEATURE_QUBITS}")

    c = cirq.Circuit()
    for q, theta in zip(FEATURE_QUBITS, angles):
        # Rx rotation for feature encoding
        c.append(cirq.rx(float(theta)).on(q))

    # Prepare result qubit in |-> = H X |0> (X then H)
    c.append(cirq.X(RESULT_QUBIT))
    c.append(cirq.H(RESULT_QUBIT))
    return c


def build_variational_block(n_layers: int = 2, layer_type: str = "XY") -> Tuple[cirq.Circuit, List[sympy.Symbol]]:
    """Build the parameterized variational circuit (to be used with TFQ PQC).

    Layer types supported:
      - 'XY' : apply Rxx(theta) then Ryy(phi) on neighboring qubit pairs
      - 'XX' : apply only Rxx on neighboring pairs
      - 'YY' : apply only Ryy on neighboring pairs

    The circuit returns a Cirq circuit with sympy symbols representing
    the trainable parameters and the corresponding symbol list.
    """
    cir = cirq.Circuit()
    symbols = []

    # For convenience create a list of all data qubits (feature qubits)
    fq = FEATURE_QUBITS
    n = len(fq)

    for layer in range(n_layers):
        # Optionally add single-qubit trainable rotations (small expressive boost)
        for i, q in enumerate(fq):
            s = sympy.Symbol(f"w_l{layer}_q{i}")
            cir.append(cirq.rx(s).on(q))
            symbols.append(s)

        # Entangling two-qubit Pauli rotations along a chain (i,i+1)
        for i in range(n - 1):
            a = sympy.Symbol(f"r_l{layer}_p{i}")
            if layer_type in ("XY", "XX"):
                # Rxx: use XXPowGate with exponent = angle / π
                cir.append(cirq.XXPowGate(exponent=a / sympy.pi).on(fq[i], fq[i + 1]))
                symbols.append(a)

            if layer_type in ("XY", "YY"):
                b = sympy.Symbol(f"s_l{layer}_p{i}")
                # Ryy
                cir.append(cirq.YYPowGate(exponent=b / sympy.pi).on(fq[i], fq[i + 1]))
                symbols.append(b)

        # Optionally close the ring between last and first qubit (makes entanglement richer)
        # here we add a final XX/YY between last and first
        a_last = sympy.Symbol(f"r_l{layer}_p{n-1}")
        if layer_type in ("XY", "XX"):
            cir.append(cirq.XXPowGate(exponent=a_last / sympy.pi).on(fq[-1], fq[0]))
            symbols.append(a_last)
        if layer_type in ("XY", "YY"):
            b_last = sympy.Symbol(f"s_l{layer}_p{n-1}")
            cir.append(cirq.YYPowGate(exponent=b_last / sympy.pi).on(fq[-1], fq[0]))
            symbols.append(b_last)

    return cir, symbols


def build_pqc_model(n_layers: int = 2, layer_type: str = "XY") -> Tuple[tf.keras.Model, List[sympy.Symbol]]:
    """Build a TFQ Keras model using the Simple QNN architecture.

    - Input: circuits (string tensor) that prepare the feature+result initial
      state (only state preparation, no variational parameters).
    - Variational circuit (returned from build_variational_block) is appended
      by TFQ's PQC layer and provides trainable parameters.
    - Readout: measure X on the result qubit (expectation of Pauli-X).
    - Output: scalar in [0,1] (maps expectation -> probability via (exp+1)/2)

    Returns:
        (compiled Keras model, list of sympy.Symbol parameters in the variational block)
    """
    var_circuit, param_symbols = build_variational_block(n_layers=n_layers, layer_type=layer_type)

    # Readout operator: X on result qubit (we prepared result in |->, measure in X basis)
    readout_op = cirq.X(RESULT_QUBIT)

    # TFQ Keras model
    circuit_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="circuit_input")
    pqc_layer = tfq.layers.PQC(var_circuit, readout_op)(circuit_input)

    # PQC returns expectation in [-1,1]. Map to probability [0,1] representing P(+1)
    prob = tf.keras.layers.Lambda(lambda x: (x + 1.0) / 2.0, name="prob_from_expectation")(pqc_layer)

    # Optionally a small dense head (paper uses direct readout majority vote, but for training a head can help)
    out = tf.keras.layers.Dense(1, activation="tanh", name="output_tanh")(prob)
    # Map tanh [-1,1] to sigmoid-like for binary crossentropy if desired; keep as-is to let caller
    model = tf.keras.Model(inputs=circuit_input, outputs=out)

    # Paper used hinge loss with labels in {-1,1}. We'll compile with hinge here by default.
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.02, decay=1e-3, momentum=0.0),
                  loss=tf.keras.losses.Hinge(),
                  metrics=[tf.keras.metrics.BinaryAccuracy()])

    return model, param_symbols


def circuits_from_angle_matrix(X_angles: np.ndarray) -> List[cirq.Circuit]:
    """Convert an (n_samples, n_features) matrix of angles (radians) into a
    list of Cirq circuits that perform the feature Rx encoding and prepare the
    result qubit.
    """
    circuits = []
    for row in X_angles:
        c = _rx_encode_circuit_row(list(row))
        circuits.append(c)
    return circuits


def compute_certainty_from_expectation(expectations: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Compute the certainty factor C in [-1,1] per the paper's definition.

    For the Simple architecture (result qubit measured in X basis):
      - PQC returns expectation <X> in [-1,1]. The probability of +1 is p = (1+<X>)/2.
      - If labels are encoded as y_true in {1,-1} where 1==benign (expected +1),
        we define C = <X> * y_true (so positive when prediction aligns with label).

    Args:
        expectations: array of shape (n_samples,) with values in [-1,1] (the raw PQC expectations)
        y_true: array of shape (n_samples,) with values in {1,-1} where 1 denotes the 'positive' class

    Returns:
        certainty array in [-1,1]
    """
    expectations = np.asarray(expectations).flatten()
    y = np.asarray(y_true).flatten()
    if expectations.shape[0] != y.shape[0]:
        raise ValueError("expectations and y_true must have same length")

    # Certainty positive when expectation sign matches label sign
    certainty = expectations * y
    # Clip to [-1,1] just in case of numeric issues
    return np.clip(certainty, -1.0, 1.0)


if __name__ == "__main__":
    # Small demo that builds the model and runs a forward pass with dummy data.
    print("Building Simple QNN model (demo)...")
    demo_layers = 2
    model, params = build_pqc_model(n_layers=demo_layers, layer_type="XY")
    print("Model built. Variational params count:", len(params))

    # Create two dummy samples with random angles in [0, pi]
    X_demo = np.random.rand(2, N_FEATURE_QUBITS) * np.pi
    circuits = circuits_from_angle_matrix(X_demo)
    circuit_tensor = tfq.convert_to_tensor(circuits)

    # Run a forward pass (will require TFQ and Cirq installed)
    try:
        out = model.predict(circuit_tensor)
        print("Forward pass output shape:", out.shape)
    except Exception as e:
        print("Could not execute forward pass — ensure TFQ & Cirq are installed.")
        print("Exception:", e)

    print("Demo complete. Use `build_pqc_model`, `circuits_from_angle_matrix`, and `compute_certainty_from_expectation` in training pipelines.")
