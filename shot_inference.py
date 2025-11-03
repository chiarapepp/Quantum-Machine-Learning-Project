"""shot_inference.py

Utilities to convert shot counts (bitstrings) into probabilities, expectations,
majority-vote class predictions, and certainty factors as defined in the paper.

Expected input for counts: a dict mapping bitstring (e.g., '0' or '1' or '010') to
an integer count. For single-qubit readout, bitstrings are typically '0'/'1'.
"""
from typing import Dict, Tuple
import numpy as np


def counts_to_prob(counts: Dict[str, int], target_outcome: str = '0') -> float:
    """Compute empirical probability of target_outcome given shot counts.

    Args:
        counts: mapping bitstring -> count
        target_outcome: the bitstring representing the +1 outcome in the chosen basis

    Returns:
        probability in [0,1]
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    target_count = counts.get(target_outcome, 0)
    return float(target_count) / float(total)


def counts_to_expectation(counts: Dict[str, int], plus_outcome: str = '0', minus_outcome: str = '1') -> float:
    """Convert shot counts to expectation value in [-1,1].

    expectation = P(plus_outcome) - P(minus_outcome)
    For multi-bit results, caller should provide canonical strings.
    """
    p_plus = counts_to_prob(counts, plus_outcome)
    p_minus = counts_to_prob(counts, minus_outcome)
    return p_plus - p_minus


def majority_vote(counts: Dict[str, int], positive_label: int = -1, negative_label: int = 1, plus_outcome: str = '0') -> int:
    """Return class label by majority vote over shots.

    Args:
        counts: bitstring->count
        positive_label: label to return when outcome indicates malicious (paper uses -1 for malicious)
        negative_label: label to return when outcome indicates benign (paper uses 1 for benign)
        plus_outcome: which bitstring corresponds to 'positive measurement' (e.g., '0')
    """
    p = counts_to_prob(counts, plus_outcome)
    # Convention: if majority measures plus_outcome, return negative_label (the paper maps +1 to benign)
    if p >= 0.5:
        return negative_label
    else:
        return positive_label


def certainty_from_counts(counts: Dict[str, int], y_true: int, plus_outcome: str = '0') -> float:
    """Compute certainty factor C from shot counts and true label.

    The paper defines C = |alpha_0|^2 - |alpha_1|^2 where for the Simple
    architecture probabilities map to measuring |+> vs |-> (or |0> vs |1> depending).

    Here we compute p_plus - p_minus and then map depending on y_true:
      if y_true==1 (benign expected +), C = p_plus - p_minus
      if y_true==-1 (malicious expected -), C = p_minus - p_plus

    Args:
        counts: shot counts
        y_true: 1 for benign (expected plus_outcome), -1 for malicious
        plus_outcome: string representing plus outcome

    Returns:
        certainty in [-1,1]
    """
    p_plus = counts_to_prob(counts, plus_outcome)
    p_minus = 1.0 - p_plus
    if y_true == 1:
        return p_plus - p_minus
    else:
        return p_minus - p_plus


def aggregate_counts_from_list(counts_list: Tuple[Dict[str, int], ...]) -> Dict[str, int]:
    """Aggregate multiple counts dicts (e.g., repeated tasks) into one counts dict."""
    agg = {}
    for c in counts_list:
        for k, v in c.items():
            agg[k] = agg.get(k, 0) + v
    return agg


if __name__ == '__main__':
    # Tiny self-check
    sample = {'0': 70, '1': 30}
    print('P(0)=', counts_to_prob(sample, '0'))
    print('expect=', counts_to_expectation(sample))
    print('majority_label=', majority_vote(sample))
    print('certainty (y=1)=', certainty_from_counts(sample, 1))
    print('certainty (y=-1)=', certainty_from_counts(sample, -1))
