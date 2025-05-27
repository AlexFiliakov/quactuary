"""
Validation utilities for quantum state preparation.

This module provides functions to validate quantum states and ensure
they meet the requirements for quantum computation.
"""

import numpy as np
from typing import Union, Tuple, List


def validate_quantum_state(
    amplitudes: Union[np.ndarray, List[complex]], 
    tolerance: float = 1e-10
) -> bool:
    """
    Validate that amplitudes form a valid quantum state.
    
    A valid quantum state must:
    1. Be normalized (sum of squared magnitudes equals 1)
    2. Have a dimension that is a power of 2
    
    Args:
        amplitudes: Complex amplitudes representing the quantum state.
        tolerance: Numerical tolerance for normalization check.
        
    Returns:
        bool: True if valid quantum state.
        
    Raises:
        ValueError: If state is invalid with details about the issue.
        
    Examples:
        >>> # Valid 2-qubit state
        >>> amps = [0.5, 0.5, 0.5, 0.5]  # |++⟩ state
        >>> validate_quantum_state(amps)
        True
        
        >>> # Invalid: not normalized
        >>> amps = [1, 1, 1, 1]
        >>> validate_quantum_state(amps)  # Raises ValueError
        
        >>> # Invalid: not power of 2
        >>> amps = [0.577, 0.577, 0.577]  # 3 amplitudes
        >>> validate_quantum_state(amps)  # Raises ValueError
    """
    amplitudes = np.array(amplitudes, dtype=complex)
    
    # Check dimension is power of 2
    n = len(amplitudes)
    if n == 0:
        raise ValueError("State vector cannot be empty")
    
    if n & (n - 1) != 0:
        raise ValueError(
            f"State dimension must be a power of 2, got {n}. "
            f"Nearest valid dimensions: {2**int(np.log2(n))}, {2**int(np.log2(n)+1)}"
        )
    
    # Check normalization
    norm_squared = np.sum(np.abs(amplitudes)**2)
    if not np.isclose(norm_squared, 1.0, rtol=tolerance):
        raise ValueError(
            f"State not normalized: ||ψ||² = {norm_squared:.6e} "
            f"(should be 1.0 within tolerance {tolerance})"
        )
    
    return True


def normalize_probabilities(
    probabilities: Union[np.ndarray, List[float]], 
    epsilon: float = 1e-12
) -> np.ndarray:
    """
    Normalize probability distribution to sum to 1.
    
    Handles edge cases like all-zero distributions gracefully.
    
    Args:
        probabilities: Unnormalized probability values.
        epsilon: Small value to avoid division by zero.
        
    Returns:
        np.ndarray: Normalized probabilities that sum to 1.
        
    Examples:
        >>> probs = [1, 2, 3, 4]
        >>> norm_probs = normalize_probabilities(probs)
        >>> print(f"Sum: {np.sum(norm_probs):.6f}")
        Sum: 1.000000
        
        >>> # Handle all-zero case
        >>> probs = [0, 0, 0, 0]
        >>> norm_probs = normalize_probabilities(probs)
        >>> print(norm_probs)  # Uniform distribution
        [0.25 0.25 0.25 0.25]
    """
    probabilities = np.array(probabilities, dtype=float)
    
    # Check for negative values
    if np.any(probabilities < 0):
        raise ValueError("Probabilities cannot be negative")
    
    # Calculate sum
    total = np.sum(probabilities)
    
    # Handle edge cases
    if total < epsilon:
        # All zeros or very small - return uniform distribution
        n = len(probabilities)
        return np.ones(n) / n
    else:
        # Normal normalization
        return probabilities / total


def check_normalization(
    values: Union[np.ndarray, List[float]], 
    mode: str = 'probability',
    tolerance: float = 1e-10
) -> Tuple[bool, float]:
    """
    Check if values are properly normalized.
    
    Args:
        values: Values to check.
        mode: Either 'probability' (sum to 1) or 'amplitude' (sum of squares to 1).
        tolerance: Numerical tolerance for the check.
        
    Returns:
        Tuple[bool, float]: (is_normalized, actual_norm)
        
    Examples:
        >>> # Check probability normalization
        >>> probs = [0.25, 0.25, 0.25, 0.25]
        >>> is_norm, norm = check_normalization(probs, mode='probability')
        >>> print(f"Normalized: {is_norm}, Sum: {norm}")
        Normalized: True, Sum: 1.0
        
        >>> # Check amplitude normalization
        >>> amps = [0.5, 0.5, 0.5, 0.5]
        >>> is_norm, norm = check_normalization(amps, mode='amplitude')
        >>> print(f"Normalized: {is_norm}, ||ψ||²: {norm}")
        Normalized: True, ||ψ||²: 1.0
    """
    values = np.array(values)
    
    if mode == 'probability':
        norm = np.sum(values)
        target = 1.0
    elif mode == 'amplitude':
        norm = np.sum(np.abs(values)**2)
        target = 1.0
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'probability' or 'amplitude'")
    
    is_normalized = np.isclose(norm, target, rtol=tolerance)
    
    return is_normalized, norm


def validate_probability_distribution(
    probabilities: Union[np.ndarray, List[float]],
    tolerance: float = 1e-10
) -> bool:
    """
    Validate that values form a valid probability distribution.
    
    Requirements:
    1. All values must be non-negative
    2. Values must sum to 1 (within tolerance)
    
    Args:
        probabilities: Probability values to validate.
        tolerance: Numerical tolerance for normalization.
        
    Returns:
        bool: True if valid probability distribution.
        
    Raises:
        ValueError: If distribution is invalid with details.
        
    Examples:
        >>> # Valid distribution
        >>> probs = [0.1, 0.2, 0.3, 0.4]
        >>> validate_probability_distribution(probs)
        True
        
        >>> # Invalid: negative value
        >>> probs = [0.5, -0.1, 0.6]
        >>> validate_probability_distribution(probs)  # Raises ValueError
    """
    probabilities = np.array(probabilities, dtype=float)
    
    # Check for negative values
    if np.any(probabilities < 0):
        negative_indices = np.where(probabilities < 0)[0]
        raise ValueError(
            f"Probabilities cannot be negative. "
            f"Found negative values at indices: {negative_indices.tolist()}"
        )
    
    # Check normalization
    total = np.sum(probabilities)
    if not np.isclose(total, 1.0, rtol=tolerance):
        raise ValueError(
            f"Probabilities must sum to 1. "
            f"Current sum: {total:.6e} (tolerance: {tolerance})"
        )
    
    return True


def suggest_qubit_number(data_size: int) -> int:
    """
    Suggest appropriate number of qubits for given data size.
    
    Args:
        data_size: Number of data points to encode.
        
    Returns:
        int: Suggested number of qubits.
        
    Examples:
        >>> suggest_qubit_number(100)  # Need at least 7 qubits
        7
        >>> suggest_qubit_number(256)  # Exactly 8 qubits
        8
    """
    if data_size <= 0:
        raise ValueError("Data size must be positive")
    
    # Calculate minimum qubits needed
    min_qubits = int(np.ceil(np.log2(data_size)))
    
    # Ensure at least 1 qubit
    return max(1, min_qubits)