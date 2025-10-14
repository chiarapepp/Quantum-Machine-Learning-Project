# qnn_nids_skeleton.py
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np

# Number of qubits = number of features
n_qubits = 10
qubits = [cirq.GridQubit(0, i) for i in range(n_qubits)]

# --- Define the quantum circuit ---
def create_qnn_circuit():
    circuit = cirq.Circuit()
    
    # 1. Feature encoding (angle encoding with Ry rotations)
    input_symbols = [sympy.Symbol(f"x{i}") for i in range(n_qubits)]
    for q, s in zip(qubits, input_symbols):
        circuit.append(cirq.ry(s).on(q))
    
    # 2. Entanglement layer
    for i in range(n_qubits - 1):
        circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))
    
    # 3. Trainable rotation layer
    weights = [sympy.Symbol(f"w{i}") for i in range(n_qubits)]
    for q, w in zip(qubits, weights):
        circuit.append(cirq.ry(w).on(q))
    
    return circuit, input_symbols, weights

# Build the circuit
qnn_circuit, input_symbols, weight_symbols = create_qnn_circuit()

# Create a readout operator (Z measurement on last qubit)
readout = cirq.Z(qubits[-1])

# Define the Keras model
qnn_layer = tfq.layers.PQC(qnn_circuit, readout)

# --- Define the hybrid model ---
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(n_qubits,)),
    qnn_layer,
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss="binary_crossentropy",
              metrics=["accuracy"])

print("✅ Quantum Neural Network model built successfully!")
model.summary()
