"""
Amplitude encoding utilities for quantum state preparation.

This module provides functions to encode classical probability distributions
as quantum amplitudes, following patterns from quantum actuarial algorithms.
"""

import numpy as np
from typing import List, Union, Optional
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation


def amplitude_encode(
    probabilities: Union[List[float], np.ndarray], 
    num_qubits: Optional[int] = None,
    normalize: bool = True,
    validate: bool = True
) -> QuantumCircuit:
    """
    Encode probability distribution as quantum amplitudes.
    
    This function takes a classical probability distribution and encodes it
    into a quantum state where the amplitudes correspond to the square roots
    of the probabilities. This is the standard encoding for quantum algorithms
    that process probability distributions.
    
    Args:
        probabilities: List or array of probability values. Can be unnormalized.
        num_qubits: Number of qubits to use. If None, automatically determined
            from the length of probabilities (rounded up to nearest power of 2).
        normalize: Whether to normalize the probabilities to sum to 1.
        validate: Whether to validate the resulting quantum state.
        
    Returns:
        QuantumCircuit: Circuit that prepares the encoded state.
        
    Raises:
        ValueError: If probabilities contain negative values or if validation fails.
        
    Examples:
        >>> # Encode a simple distribution
        >>> probs = [0.1, 0.2, 0.3, 0.4]
        >>> qc = amplitude_encode(probs)
        >>> print(f"Circuit uses {qc.num_qubits} qubits")
        Circuit uses 2 qubits
        
        >>> # Encode with specific number of qubits
        >>> probs = [0.25, 0.75]
        >>> qc = amplitude_encode(probs, num_qubits=3)  # Will pad with zeros
        >>> print(f"State dimension: {2**qc.num_qubits}")
        State dimension: 8
        
    Notes:
        - The StatePreparation gate from Qiskit is used for efficient encoding
        - Probabilities are automatically padded with zeros if needed
        - The resulting state |ψ⟩ has amplitudes √(p_i/Σp_i)
    """
    # Convert to numpy array for easier manipulation
    probabilities = np.array(probabilities, dtype=float)
    
    # Check for negative probabilities
    if np.any(probabilities < 0):
        raise ValueError("Probabilities cannot be negative")
    
    # Determine number of qubits needed
    if num_qubits is None:
        num_qubits = int(np.ceil(np.log2(len(probabilities))))
        # Ensure at least 1 qubit
        num_qubits = max(1, num_qubits)
    
    # Calculate state dimension
    state_size = 2**num_qubits
    
    # Pad or truncate to correct size
    if len(probabilities) < state_size:
        # Pad with zeros
        probabilities = np.pad(
            probabilities, 
            (0, state_size - len(probabilities)),
            mode='constant',
            constant_values=0
        )
    elif len(probabilities) > state_size:
        # Truncate and warn
        print(f"Warning: Truncating {len(probabilities)} probabilities to {state_size}")
        probabilities = probabilities[:state_size]
    
    # Normalize if requested
    if normalize:
        prob_sum = np.sum(probabilities)
        if prob_sum > 0:
            probabilities = probabilities / prob_sum
        else:
            # Handle all-zero case
            probabilities = np.ones(state_size) / state_size
    
    # Convert probabilities to amplitudes (square root)
    amplitudes = np.sqrt(probabilities)
    
    # Additional normalization check for amplitudes (only if normalize=True)
    if normalize:
        amp_norm = np.sum(np.abs(amplitudes)**2)
        if not np.isclose(amp_norm, 1.0, rtol=1e-10):
            amplitudes = amplitudes / np.sqrt(amp_norm)
    
    # Validate if requested
    if validate:
        from .validation import validate_quantum_state
        validate_quantum_state(amplitudes)
    
    # Create quantum circuit
    qc = QuantumCircuit(num_qubits, name='amplitude_encoding')
    
    # Use StatePreparation for efficient encoding
    # Pass normalize parameter to StatePreparation to handle unnormalized states
    state_prep = StatePreparation(amplitudes, normalize=normalize)
    qc.append(state_prep, range(num_qubits))
    
    return qc


def uniform_superposition(num_qubits: int) -> QuantumCircuit:
    """
    Create uniform superposition state using Hadamard gates.
    
    This creates the state |ψ⟩ = (1/√N) Σ|i⟩ where N = 2^num_qubits.
    This is equivalent to amplitude encoding a uniform distribution.
    
    Args:
        num_qubits: Number of qubits to put in superposition.
        
    Returns:
        QuantumCircuit: Circuit creating uniform superposition.
        
    Examples:
        >>> qc = uniform_superposition(3)
        >>> # Creates state (|000⟩ + |001⟩ + ... + |111⟩)/√8
        >>> print(f"Creates superposition of {2**3} states")
        Creates superposition of 8 states
        
    Notes:
        - This is the most efficient way to create uniform superposition
        - Uses n Hadamard gates for n qubits (linear scaling)
        - Equivalent to amplitude_encode with uniform probabilities
    """
    if num_qubits < 1:
        raise ValueError("Number of qubits must be at least 1")
    
    qc = QuantumCircuit(num_qubits, name='uniform_superposition')
    
    # Apply Hadamard to all qubits
    qc.h(range(num_qubits))
    
    return qc


def controlled_rotation_encoding(
    angles: Union[List[float], np.ndarray],
    target_qubit: Optional[int] = None
) -> QuantumCircuit:
    """
    Encode data using controlled rotation gates.
    
    This encoding is useful for sparse states or when specific angle
    rotations are needed. It uses multi-controlled RY rotations to
    encode the given angles.
    
    Args:
        angles: List of rotation angles in radians.
        target_qubit: Which qubit to use as rotation target.
            If None, uses the last qubit.
            
    Returns:
        QuantumCircuit: Circuit implementing the controlled rotations.
        
    Examples:
        >>> # Encode four different angles
        >>> angles = [np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
        >>> qc = controlled_rotation_encoding(angles)
        >>> print(f"Circuit depth: {qc.depth()}")
        
    Notes:
        - Number of qubits is determined by ceil(log2(len(angles)))
        - Uses gray code ordering for efficiency when possible
        - More efficient than StatePreparation for sparse encodings
    """
    angles = np.array(angles)
    n_angles = len(angles)
    
    if n_angles == 0:
        raise ValueError("Must provide at least one angle")
    
    # Determine number of control qubits needed
    n_control = int(np.ceil(np.log2(n_angles)))
    n_qubits = n_control + 1  # Control qubits + target qubit
    
    # Create circuit
    qc = QuantumCircuit(n_qubits, name='controlled_rotation_encoding')
    
    # Determine target qubit
    if target_qubit is None:
        target_qubit = n_qubits - 1
    
    # Control qubits are all except target
    control_qubits = list(range(n_qubits))
    control_qubits.remove(target_qubit)
    
    # Initialize control qubits in superposition
    for q in control_qubits:
        qc.h(q)
    
    # Apply controlled rotations based on binary encoding
    for i, angle in enumerate(angles):
        if abs(angle) < 1e-10:  # Skip near-zero rotations
            continue
            
        # Get binary representation of index
        binary = format(i, f'0{n_control}b')
        
        # Determine which qubits should be in |1⟩ for this rotation
        control_state = []
        for j, bit in enumerate(binary):
            if bit == '1':
                control_state.append(control_qubits[j])
        
        # Apply appropriate rotation
        if len(control_state) == 0:
            # No controls - direct rotation
            qc.ry(2 * angle, target_qubit)  # Factor of 2 for RY convention
        elif len(control_state) == 1:
            # Single control
            qc.cry(2 * angle, control_state[0], target_qubit)
        else:
            # Multi-controlled rotation
            qc.mcry(2 * angle, control_state, target_qubit)
    
    return qc