"""
Noise simulation for NISQ quantum computing environments.

Implements noise models used in "Network Anomaly Detection Using Quantum Neural Networks 
on Noisy Quantum Computers":
- Depolarizing noise (primary focus in paper: 0.1%, 0.5%, 1%, 5%, 10%)
- Amplitude damping
- Phase damping
- Bit flip errors
- Measurement errors

The paper tests QNN resilience to noise by adding depolarizing channels after each gate.
"""

import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
from typing import Optional, Dict, List, Callable
import warnings


class NoiseModel:
    """
    Container for noise parameters and application methods.
    """
    def __init__(self, 
                 noise_type: str = "depolarizing",
                 noise_prob: float = 0.01,
                 gate_noise_prob: Optional[float] = None,
                 measurement_noise_prob: Optional[float] = None,
                 t1: Optional[float] = None,
                 t2: Optional[float] = None):
        """
        Args:
            noise_type: Type of noise ("depolarizing", "amplitude_damping", "phase_damping", "bit_flip", "mixed")
            noise_prob: Default noise probability (0.0 to 1.0)
            gate_noise_prob: Specific noise for gate operations (overrides noise_prob if set)
            measurement_noise_prob: Specific noise for measurements
            t1: Amplitude damping time constant (for amplitude_damping)
            t2: Phase damping time constant (for phase_damping)
        """
        self.noise_type = noise_type
        self.noise_prob = noise_prob
        self.gate_noise_prob = gate_noise_prob if gate_noise_prob is not None else noise_prob
        self.measurement_noise_prob = measurement_noise_prob if measurement_noise_prob is not None else noise_prob
        self.t1 = t1
        self.t2 = t2
        
        # Validate
        if not 0 <= self.gate_noise_prob <= 1:
            raise ValueError(f"Gate noise probability must be in [0,1], got {self.gate_noise_prob}")
        if not 0 <= self.measurement_noise_prob <= 1:
            raise ValueError(f"Measurement noise probability must be in [0,1], got {self.measurement_noise_prob}")


def create_noisy_device(n_qubits: int, 
                        noise_model: NoiseModel,
                        shots: Optional[int] = None):
    """
    Create a PennyLane device with noise simulation.
    
    For PennyLane, we use 'default.mixed' device which supports density matrix simulation
    and noise channels.
    
    Args:
        n_qubits: Number of qubits
        noise_model: NoiseModel instance with noise parameters
        shots: Number of shots for sampling (None = analytic)
    
    Returns:
        PennyLane device configured for noisy simulation
    """
    if noise_model.gate_noise_prob > 0 or noise_model.measurement_noise_prob > 0:
        # Use mixed state simulator for noise
        dev = qml.device("default.mixed", wires=n_qubits, shots=shots)
        print(f"[noise] Created noisy device: {noise_model.noise_type} "
              f"(gate_p={noise_model.gate_noise_prob:.4f}, meas_p={noise_model.measurement_noise_prob:.4f})")
    else:
        # Clean simulation
        dev = qml.device("default.qubit", wires=n_qubits, shots=shots)
        print(f"[noise] Created clean device (no noise)")
    
    return dev


def apply_gate_noise(noise_model: NoiseModel, wires: List[int]):
    """
    Apply noise channel after a gate operation.
    
    This should be called after each gate in the circuit to simulate NISQ hardware.
    
    Args:
        noise_model: NoiseModel instance
        wires: Qubits to apply noise to
    """
    if noise_model.gate_noise_prob <= 0:
        return
    
    for wire in wires:
        if noise_model.noise_type == "depolarizing":
            qml.DepolarizingChannel(noise_model.gate_noise_prob, wires=wire)
        
        elif noise_model.noise_type == "amplitude_damping":
            gamma = noise_model.gate_noise_prob if noise_model.t1 is None else 1 - np.exp(-1/noise_model.t1)
            qml.AmplitudeDamping(gamma, wires=wire)
        
        elif noise_model.noise_type == "phase_damping":
            gamma = noise_model.gate_noise_prob if noise_model.t2 is None else 1 - np.exp(-1/noise_model.t2)
            qml.PhaseDamping(gamma, wires=wire)
        
        elif noise_model.noise_type == "bit_flip":
            qml.BitFlip(noise_model.gate_noise_prob, wires=wire)
        
        elif noise_model.noise_type == "phase_flip":
            qml.PhaseFlip(noise_model.gate_noise_prob, wires=wire)
        
        elif noise_model.noise_type == "mixed":
            # Apply combination of noise channels (realistic NISQ model)
            # Depolarizing + amplitude damping
            qml.DepolarizingChannel(noise_model.gate_noise_prob * 0.7, wires=wire)
            qml.AmplitudeDamping(noise_model.gate_noise_prob * 0.3, wires=wire)
        
        else:
            warnings.warn(f"Unknown noise type: {noise_model.noise_type}")


def apply_measurement_noise(noise_model: NoiseModel, wire: int):
    """
    Apply noise before measurement (readout errors).
    
    Args:
        noise_model: NoiseModel instance
        wire: Qubit to apply measurement noise to
    """
    if noise_model.measurement_noise_prob <= 0:
        return
    
    # Measurement noise typically modeled as bit flip before readout
    qml.BitFlip(noise_model.measurement_noise_prob, wires=wire)


# Paper-specific noise levels for experiments
PAPER_NOISE_LEVELS = {
    "clean": 0.000,
    "very_low": 0.001,   # 0.1%
    "low": 0.005,        # 0.5%
    "medium": 0.01,      # 1%
    "high": 0.05,        # 5%
    "very_high": 0.10    # 10%
}


def get_paper_noise_model(noise_level: str = "medium") -> NoiseModel:
    """
    Get pre-configured noise model matching paper experiments.
    
    Args:
        noise_level: One of "clean", "very_low", "low", "medium", "high", "very_high"
    
    Returns:
        NoiseModel instance
    """
    if noise_level not in PAPER_NOISE_LEVELS:
        raise ValueError(f"Unknown noise level: {noise_level}. "
                        f"Choose from {list(PAPER_NOISE_LEVELS.keys())}")
    
    prob = PAPER_NOISE_LEVELS[noise_level]
    return NoiseModel(
        noise_type="depolarizing",
        noise_prob=prob,
        gate_noise_prob=prob,
        measurement_noise_prob=prob * 0.5  # Typically lower measurement error
    )


def noise_aware_two_qubit_block(params: pnp.ndarray, 
                                wires: List[int],
                                noise_model: Optional[NoiseModel] = None):
    """
    Generic two-qubit block with noise injection.
    
    This replaces the clean two_qubit_block from architectures.py when noise is desired.
    
    Args:
        params: shape (6,) - 3 params per Rot gate
        wires: [a, b]
        noise_model: NoiseModel instance (None = no noise)
    """
    a, b = wires[0], wires[1]
    
    # First single qubit gate
    qml.Rot(params[0], params[1], params[2], wires=a)
    if noise_model:
        apply_gate_noise(noise_model, [a])
    
    # Second single qubit gate
    qml.Rot(params[3], params[4], params[5], wires=b)
    if noise_model:
        apply_gate_noise(noise_model, [b])
    
    # CNOT entangler
    qml.CNOT(wires=[a, b])
    if noise_model:
        apply_gate_noise(noise_model, [a, b])


def wrap_circuit_with_noise(circuit_builder: Callable,
                            noise_model: NoiseModel) -> Callable:
    """
    Decorator to wrap a circuit builder function with noise injection.
    
    This is a more automated approach - wraps existing circuit builders.
    
    Args:
        circuit_builder: Function that builds a QNN circuit
        noise_model: NoiseModel instance
    
    Returns:
        Modified circuit builder with noise
    """
    def noisy_circuit_builder(*args, **kwargs):
        # Override device with noisy version if needed
        if 'dev' in kwargs:
            n_qubits = kwargs.get('n_qubits', args[0] if args else 8)
            shots = kwargs.get('shots', None)
            kwargs['dev'] = create_noisy_device(n_qubits, noise_model, shots)
        
        # Build circuit (this returns a qnode)
        qnode = circuit_builder(*args, **kwargs)
        
        # Note: Noise is applied within the circuit using noise_aware_two_qubit_block
        # or by using default.mixed device which supports noise channels
        
        return qnode
    
    return noisy_circuit_builder


# Convenience function for benchmarking
def run_noise_sweep(model, X_test, y_test, noise_levels: List[str], 
                   eval_fn: Callable) -> Dict[str, Dict]:
    """
    Evaluate model across different noise levels.
    
    Args:
        model: QNN model instance
        X_test: Test features
        y_test: Test labels
        noise_levels: List of noise level names (e.g., ["clean", "low", "medium", "high"])
        eval_fn: Function(model, X, y, noise_model) -> metrics dict
    
    Returns:
        Dictionary mapping noise_level -> metrics
    """
    results = {}
    
    for level in noise_levels:
        print(f"\n[noise_sweep] Evaluating at noise level: {level}")
        noise_model = get_paper_noise_model(level)
        metrics = eval_fn(model, X_test, y_test, noise_model)
        results[level] = metrics
        print(f"  F1: {metrics.get('f1', 0):.4f}, Acc: {metrics.get('accuracy', 0):.4f}")
    
    return results


# IBM hardware noise profiles (approximate)
IBM_HARDWARE_NOISE = {
    "ibmq_lima": NoiseModel(
        noise_type="mixed",
        gate_noise_prob=0.005,  # ~0.5% gate error
        measurement_noise_prob=0.015,  # ~1.5% readout error
        t1=100e-6,  # 100 microseconds
        t2=50e-6    # 50 microseconds
    ),
    "ibmq_belem": NoiseModel(
        noise_type="mixed",
        gate_noise_prob=0.008,
        measurement_noise_prob=0.02,
        t1=80e-6,
        t2=40e-6
    ),
    "ibmq_quito": NoiseModel(
        noise_type="mixed",
        gate_noise_prob=0.006,
        measurement_noise_prob=0.018,
        t1=90e-6,
        t2=45e-6
    )
}


def get_hardware_noise_model(hardware_name: str) -> NoiseModel:
    """
    Get noise model approximating real IBM hardware.
    
    Args:
        hardware_name: Name of IBM hardware ("ibmq_lima", "ibmq_belem", "ibmq_quito")
    
    Returns:
        NoiseModel instance
    """
    if hardware_name not in IBM_HARDWARE_NOISE:
        raise ValueError(f"Unknown hardware: {hardware_name}. "
                        f"Choose from {list(IBM_HARDWARE_NOISE.keys())}")
    
    return IBM_HARDWARE_NOISE[hardware_name]


if __name__ == "__main__":
    # Test noise models
    print("Testing noise models...")
    
    # Create clean device
    clean_model = NoiseModel(noise_type="depolarizing", noise_prob=0.0)
    dev_clean = create_noisy_device(4, clean_model, shots=None)
    
    # Create noisy device (1% depolarizing)
    noisy_model = get_paper_noise_model("medium")
    dev_noisy = create_noisy_device(4, noisy_model, shots=1000)
    
    # Simple test circuit
    @qml.qnode(dev_noisy, interface="torch")
    def test_circuit(x):
        for i in range(4):
            qml.RX(x[i], wires=i)
            apply_gate_noise(noisy_model, [i])
        
        qml.CNOT(wires=[0, 1])
        apply_gate_noise(noisy_model, [0, 1])
        
        apply_measurement_noise(noisy_model, 0)
        return qml.expval(qml.PauliZ(0))
    
    import torch
    x = torch.tensor([0.5, 0.3, 0.2, 0.1])
    result = test_circuit(x)
    print(f"Noisy circuit result: {result}")
    
    # Print paper noise levels
    print("\nPaper noise levels:")
    for level, prob in PAPER_NOISE_LEVELS.items():
        print(f"  {level:12s}: {prob*100:5.1f}%")