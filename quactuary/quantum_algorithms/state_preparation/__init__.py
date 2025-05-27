"""
Quantum state preparation utilities for quactuary.

This module provides core utilities for preparing quantum states from classical data,
particularly probability distributions used in actuarial modeling.

Key Components:
    - amplitude_encode: Encode probability distributions as quantum amplitudes
    - uniform_superposition: Create uniform superposition states
    - controlled_rotation_encoding: Encode using controlled rotations
    - validate_quantum_state: Validate quantum state properties
    - prepare_distribution_state: Prepare states from scipy distributions

Examples:
    Basic amplitude encoding:
        >>> import numpy as np
        >>> from quactuary.quantum.state_preparation import amplitude_encode
        >>> 
        >>> probabilities = [0.25, 0.25, 0.25, 0.25]
        >>> qc = amplitude_encode(probabilities)
        >>> print(qc.num_qubits)
        2
        
    Lognormal distribution encoding:
        >>> from quactuary.quantum.state_preparation import prepare_lognormal_state
        >>> 
        >>> probs, x_vals = prepare_lognormal_state(mu=0.0, sigma=1.0, num_qubits=6)
        >>> qc = amplitude_encode(probs)
"""

from .amplitude_encoding import (
    amplitude_encode,
    uniform_superposition,
    controlled_rotation_encoding,
)
from .probability_loader import (
    prepare_lognormal_state,
    prepare_distribution_state,
    discretize_distribution,
)
from .validation import (
    validate_quantum_state,
    normalize_probabilities,
    check_normalization,
)

__all__ = [
    # Core encoding functions
    'amplitude_encode',
    'uniform_superposition',
    'controlled_rotation_encoding',
    
    # Distribution preparation
    'prepare_lognormal_state',
    'prepare_distribution_state',
    'discretize_distribution',
    
    # Validation utilities
    'validate_quantum_state',
    'normalize_probabilities',
    'check_normalization',
]