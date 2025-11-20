import pennylane as qml
from pennylane import numpy as pnp
from .encoding import encode_inputs

def xy_layer(params, wires):
    n = len(wires)
    for i in range(n - 1):
        qml.IsingXX(params[i], wires=[wires[i], wires[i+1]])
        qml.IsingYY(params[i], wires=[wires[i], wires[i+1]])
    # connect last→first
    qml.IsingXX(params[-1], wires=[wires[-1], wires[0]])
    qml.IsingYY(params[-1], wires=[wires[-1], wires[0]])

def build_simple_qnn(n_qubits, n_layers, dev):
    @qml.qnode(dev, interface="torch")
    def circuit(x, params):
        encode_inputs(x)
        for l in range(n_layers):
            xy_layer(params[l], range(n_qubits))
        return qml.expval(qml.PauliZ(n_qubits - 1))
    return circuit
