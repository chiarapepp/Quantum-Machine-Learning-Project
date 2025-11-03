"""consolidate_gates.py

Conservative utilities to consolidate consecutive single-qubit rotations in a
Cirq circuit. This helps reduce the number of unique rotation gates, which
can be useful when targeting devices with per-channel unique-gate limits.

This implementation only merges consecutive Rx rotations on the same qubit
when they are adjacent in the circuit; it does NOT attempt to reorder gates or
merge rotations separated by entangling operations (safer approach).
"""
from typing import Iterable
import cirq
import math


def merge_consecutive_rx(circuit: cirq.Circuit, tolerance: float = 1e-9) -> cirq.Circuit:
    """Return a new circuit where consecutive Rx operations on the same
    qubit are merged into a single Rx with summed angle.

    Args:
        circuit: input Cirq circuit
        tolerance: small value below which rotations are considered zero and removed
    """
    new_moments = []

    for moment in circuit:
        # We'll accumulate replacements within the moment
        ops = list(moment.operations)
        merged_ops = []

        i = 0
        while i < len(ops):
            op = ops[i]
            # Check if op is RxPowGate or XX/YYPow handled elsewhere; we only merge Rx
            if isinstance(op.gate, cirq.ops.common_gates.Rx):
                # Shouldn't happen with Rx objects; Cirq represents rx via XPowGate with exponent
                # instead inspect gate type name
                pass

            # Accommodate XPowGate which is e^{-i X pi exponent / 2} such that Rx(angle) == XPowGate(exponent=angle/pi)
            if hasattr(op.gate, 'exponent') and op.gate.__class__.__name__ in ('XPowGate', 'XPowGate'):
                # get angle
                angle = float(op.gate._rads) if hasattr(op.gate, '_rads') else float(op.gate.exponent * math.pi)
                q = op.qubits[0]
                # accumulate consecutive XPowGate on same qubit within this moment
                j = i + 1
                total_angle = angle
                while j < len(ops):
                    next_op = ops[j]
                    if hasattr(next_op.gate, 'exponent') and next_op.gate.__class__.__name__ in ('XPowGate', 'XPowGate') and next_op.qubits[0] == q:
                        a2 = float(next_op.gate._rads) if hasattr(next_op.gate, '_rads') else float(next_op.gate.exponent * math.pi)
                        total_angle += a2
                        j += 1
                    else:
                        break

                # create replacement
                if abs(total_angle) > tolerance:
                    new_op = cirq.rx(total_angle).on(q)
                    merged_ops.append(new_op)
                # advance
                i = j
            else:
                merged_ops.append(op)
                i += 1

        new_moments.append(cirq.Moment(merged_ops))

    return cirq.Circuit(new_moments)


def count_unique_single_qubit_rotations(circuit: cirq.Circuit) -> int:
    """Count unique rotation angles applied in Rx/Ry/Rz gates across the circuit.
    Conservative counting — compares rounded angles.
    """
    uniq = set()
    for op in circuit.all_operations():
        gname = op.gate.__class__.__name__
        if gname in ('XPowGate', 'YPowGate', 'ZPowGate'):
            # map exponent -> angle
            exp = getattr(op.gate, 'exponent', None)
            if exp is None:
                continue
            angle = float(exp) * math.pi
            uniq.add(round(angle, 8))
    return len(uniq)


if __name__ == '__main__':
    # Basic smoke test
    q = cirq.GridQubit(0, 0)
    c = cirq.Circuit()
    c.append(cirq.rx(0.1).on(q))
    c.append(cirq.rx(0.2).on(q))
    c.append(cirq.H(q))
    c.append(cirq.rx(-0.05).on(q))

    print('Original circuit:')
    print(c)
    merged = merge_consecutive_rx(c)
    print('\nMerged circuit:')
    print(merged)
    print('Unique rot count:', count_unique_single_qubit_rotations(merged))
