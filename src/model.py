# src/model.py
"""
PyTorch wrapper around PennyLane QNodes.

Separates model construction from training logic.

Key correctness points vs. the paper:
  - Simple arch: result qubit = wire n_feature_qubits  (separate from feature wires 0..n-1)
  - Device must expose n_feature_qubits + 1 wires for Simple, n_feature_qubits for others.
  - Parameter counts come exclusively from architectures.py helper functions
    (no duplicate, possibly-wrong formulas here).
  - `noise_model` is forwarded to arch builders (they accept and ignore it;
    noisy simulation is done by swapping in a `default.mixed` device externally).
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import pennylane as qml

import architectures as archs


# ---------------------------------------------------------------------------
# Device factory
# ---------------------------------------------------------------------------

def make_device(
    n_feature_qubits: int,
    arch: str,
    shots: Optional[int] = None,
    noisy: bool = False,
) -> qml.Device:
    """
    Create a PennyLane device with the correct wire count.

    Simple needs n_feature_qubits + 1 wires (extra result qubit).
    All others need n_feature_qubits wires.
    """
    n_wires = n_feature_qubits + (1 if arch == "simple" else 0)
    backend = "default.mixed" if noisy else "default.qubit"
    return qml.device(backend, wires=n_wires, shots=shots)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class QNNModel(nn.Module):
    """
    Thin torch.nn.Module wrapper around a PennyLane QNode.

    Parameters
    ----------
    arch : str
        One of "simple", "ttn", "mera", "qcnn".
    n_feature_qubits : int
        Number of encoded feature qubits (8 in the paper).
    n_layers : int
        Number of variational layers / scales.
    layer_type : str
        Pauli sweep type for Simple arch. Ignored by TTN/MERA/QCNN.
    dev : qml.Device
        Pre-built PennyLane device (use make_device() above).
    noise_model : optional
        Passed through to arch builders (currently unused inside them;
        reserved for future gate-level noise injection).
    """

    def __init__(
        self,
        arch: str,
        n_feature_qubits: int,
        n_layers: int,
        dev: qml.Device,
        layer_type: str = "XXYY",
        noise_model=None,
    ):
        super().__init__()
        self.arch             = arch
        self.n_feature_qubits = n_feature_qubits
        self.n_layers         = n_layers
        self.layer_type       = layer_type

        # Build QNode
        if arch == "simple":
            self.qnode = archs.build_simple_qnn(
                n_feature_qubits=n_feature_qubits,
                n_layers=n_layers,
                dev=dev,
                layer_type=layer_type,
                noise_model=noise_model,
            )
            n_params = archs.simple_num_params(n_feature_qubits, n_layers, layer_type)

        elif arch == "ttn":
            self.qnode = archs.build_ttn_qnn(
                n_qubits=n_feature_qubits,
                dev=dev,
                noise_model=noise_model,
            )
            n_params = archs.ttn_num_params(n_feature_qubits)

        elif arch == "mera":
            self.qnode = archs.build_mera_qnn(
                n_qubits=n_feature_qubits,
                n_scales=n_layers,
                dev=dev,
                noise_model=noise_model,
            )
            n_params = archs.mera_num_params(n_feature_qubits, n_layers)

        elif arch == "qcnn":
            self.qnode = archs.build_qcnn_qnn(
                n_qubits=n_feature_qubits,
                dev=dev,
                noise_model=noise_model,
            )
            n_params = archs.qcnn_num_params(n_feature_qubits)

        else:
            raise ValueError(f"Unknown arch '{arch}'. Choose: simple, ttn, mera, qcnn.")

        # Small random initialisation (avoids barren plateau at zero)
        self.qparams = nn.Parameter(0.01 * torch.randn(n_params))

    def forward(self, x_batch: torch.Tensor) -> torch.Tensor:
        """
        Run one forward pass for a batch.

        Returns a 1-D tensor of certainty factors C in [-1, 1],
        one value per sample.
        """
        outs = []
        for i in range(x_batch.shape[0]):
            val = self.qnode(x_batch[i], self.qparams)
            if not isinstance(val, torch.Tensor):
                val = torch.tensor(float(val), dtype=torch.float32)
            outs.append(val.reshape(()))
        return torch.stack(outs)