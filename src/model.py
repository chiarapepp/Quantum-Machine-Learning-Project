from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import pennylane as qml

import architectures as archs


def make_device(
    n_feature_qubits: int,
    arch: str,
    shots: Optional[int] = None,
    noisy: bool = False,
):
    n_wires = n_feature_qubits + (1 if arch == "simple" else 0)
    backend = "default.mixed" if noisy else "default.qubit"
    return qml.device(backend, wires=n_wires, shots=shots)


class QNNModel(nn.Module):
    def __init__(
        self,
        arch: str,
        n_feature_qubits: int,
        n_layers: int,
        dev,
        layer_type: str = "XXYY",
    ) -> None:
        super().__init__()
        self.arch = arch
        self.n_feature_qubits = n_feature_qubits
        self.n_layers = n_layers
        self.layer_type = layer_type

        if arch == "simple":
            self.qnode = archs.build_simple_qnn(
                n_feature_qubits=n_feature_qubits,
                n_layers=n_layers,
                dev=dev,
                layer_type=layer_type,
            )
            n_params = archs.simple_num_params(n_feature_qubits, n_layers, layer_type)

        elif arch == "ttn":
            self.qnode = archs.build_ttn_qnn(
                n_qubits=n_feature_qubits,
                dev=dev,
            )
            n_params = archs.ttn_num_params(n_feature_qubits)

        elif arch == "mera":
            self.qnode = archs.build_mera_qnn(
                n_qubits=n_feature_qubits,
                n_scales=n_layers,
                dev=dev,
            )
            n_params = archs.mera_num_params(n_feature_qubits, n_layers)

        elif arch == "qcnn":
            self.qnode = archs.build_qcnn_qnn(
                n_qubits=n_feature_qubits,
                dev=dev,
            )
            n_params = archs.qcnn_num_params(n_feature_qubits)

        else:
            raise ValueError(f"Unknown arch '{arch}'.")

        # Initialize parameters with small random values.
        self.qparams = nn.Parameter(0.01 * torch.randn(n_params))

    def forward(self, x_batch: torch.Tensor) -> torch.Tensor:
        outs = []
        for i in range(x_batch.shape[0]):
            val = self.qnode(x_batch[i], self.qparams)
            outs.append(val.reshape(()))
        return torch.stack(outs)