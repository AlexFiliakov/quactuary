"""
Validation utilities for quantum computations.

This module provides functions to validate quantum circuits, states,
and algorithm inputs.
"""

from typing import Union, List, Tuple, Optional
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix

from quactuary.quantum.quantum_types import (
    MAX_QUBITS_SIMULATOR,
    CircuitConstructionError,
    StatePreparationError
)


def validate_circuit_size(num_qubits: int, backend_type: str = "simulator") -> None:
    """
    Validate that circuit size is within backend limits.
    
    Args:
        num_qubits: Number of qubits in circuit
        backend_type: Type of backend ("simulator" or "hardware")
        
    Raises:
        CircuitConstructionError: If circuit is too large
    """
    if backend_type == "simulator" and num_qubits > MAX_QUBITS_SIMULATOR:
        raise CircuitConstructionError(
            f"Circuit with {num_qubits} qubits exceeds simulator limit "
            f"of {MAX_QUBITS_SIMULATOR} qubits"
        )
    elif backend_type == "hardware" and num_qubits > 127:  # Current IBM limit
        raise CircuitConstructionError(
            f"Circuit with {num_qubits} qubits exceeds current "
            f"hardware limit of 127 qubits"
        )


def validate_probability_distribution(probabilities: Union[List[float], np.ndarray],
                                    tolerance: float = 1e-10) -> np.ndarray:
    """
    Validate and normalize a probability distribution.
    
    Args:
        probabilities: Probability values
        tolerance: Tolerance for normalization check
        
    Returns:
        Normalized probability distribution
        
    Raises:
        ValueError: If probabilities are invalid
    """
    probs = np.asarray(probabilities)
    
    # Check for negative values
    if np.any(probs < 0):
        raise ValueError("Probabilities cannot be negative")
        
    # Check for all zeros
    total = np.sum(probs)
    if total == 0:
        raise ValueError("All probabilities are zero")
        
    # Normalize if needed
    if abs(total - 1.0) > tolerance:
        probs = probs / total
        
    return probs


def validate_quantum_state(state: Union[Statevector, DensityMatrix, np.ndarray],
                          num_qubits: Optional[int] = None) -> None:
    """
    Validate a quantum state.
    
    Args:
        state: Quantum state to validate
        num_qubits: Expected number of qubits
        
    Raises:
        StatePreparationError: If state is invalid
    """
    if isinstance(state, np.ndarray):
        # Check if it's a valid state vector
        if len(state.shape) != 1:
            raise StatePreparationError(
                "State vector must be a 1D array"
            )
            
        # Check if dimension is power of 2
        n = len(state)
        if n & (n - 1) != 0:
            raise StatePreparationError(
                f"State vector dimension {n} is not a power of 2"
            )
            
        # Check normalization
        norm = np.linalg.norm(state)
        if abs(norm - 1.0) > 1e-10:
            raise StatePreparationError(
                f"State vector is not normalized (norm={norm})"
            )
            
        # Check number of qubits if specified
        if num_qubits is not None:
            expected_dim = 2**num_qubits
            if n != expected_dim:
                raise StatePreparationError(
                    f"State vector dimension {n} does not match "
                    f"expected dimension {expected_dim} for {num_qubits} qubits"
                )
                
    elif isinstance(state, (Statevector, DensityMatrix)):
        # Qiskit objects have built-in validation
        if not state.is_valid():
            raise StatePreparationError("Invalid quantum state")
            
        if num_qubits is not None and state.num_qubits != num_qubits:
            raise StatePreparationError(
                f"State has {state.num_qubits} qubits, "
                f"expected {num_qubits}"
            )
    else:
        raise StatePreparationError(
            f"Unknown state type: {type(state)}"
        )


def validate_circuit_compatibility(circuit1: QuantumCircuit,
                                 circuit2: QuantumCircuit) -> None:
    """
    Validate that two circuits are compatible for composition.
    
    Args:
        circuit1: First circuit
        circuit2: Second circuit
        
    Raises:
        CircuitConstructionError: If circuits are incompatible
    """
    if circuit1.num_qubits != circuit2.num_qubits:
        raise CircuitConstructionError(
            f"Circuits have different qubit counts: "
            f"{circuit1.num_qubits} vs {circuit2.num_qubits}"
        )
        
    # Check for naming conflicts in registers
    c1_qregs = {reg.name for reg in circuit1.qregs}
    c2_qregs = {reg.name for reg in circuit2.qregs}
    
    conflicts = c1_qregs.intersection(c2_qregs)
    if conflicts and len(conflicts) < len(c1_qregs):
        # Partial overlap is problematic
        raise CircuitConstructionError(
            f"Register name conflicts: {conflicts}"
        )


def validate_parameter_values(circuit: QuantumCircuit,
                            param_values: Union[List[float], np.ndarray]) -> None:
    """
    Validate parameter values for a parameterized circuit.
    
    Args:
        circuit: Parameterized quantum circuit
        param_values: Parameter values to validate
        
    Raises:
        ValueError: If parameter values are invalid
    """
    num_params = circuit.num_parameters
    num_values = len(param_values)
    
    if num_params != num_values:
        raise ValueError(
            f"Circuit expects {num_params} parameters, "
            f"got {num_values} values"
        )
        
    # Check for NaN or infinite values
    param_array = np.asarray(param_values)
    if np.any(np.isnan(param_array)):
        raise ValueError("Parameter values contain NaN")
        
    if np.any(np.isinf(param_array)):
        raise ValueError("Parameter values contain infinity")


def check_circuit_depth(circuit: QuantumCircuit,
                       max_depth: Optional[int] = None) -> Tuple[bool, int]:
    """
    Check if circuit depth is within limits.
    
    Args:
        circuit: Circuit to check
        max_depth: Maximum allowed depth
        
    Returns:
        Tuple of (is_within_limit, actual_depth)
    """
    depth = circuit.depth()
    
    if max_depth is None:
        return True, depth
        
    return depth <= max_depth, depth


def estimate_circuit_runtime(circuit: QuantumCircuit,
                           backend_type: str = "simulator") -> float:
    """
    Estimate circuit execution time.
    
    Args:
        circuit: Circuit to analyze
        backend_type: Type of backend
        
    Returns:
        Estimated runtime in seconds
    """
    depth = circuit.depth()
    num_qubits = circuit.num_qubits
    gate_count = len(circuit.data)
    
    if backend_type == "simulator":
        # Rough estimate for statevector simulation
        # Scales exponentially with qubits, linearly with gates
        base_time = 1e-6  # 1 microsecond per gate per state
        state_size = 2**num_qubits
        return base_time * gate_count * state_size
        
    else:  # hardware
        # Rough estimate based on gate times
        single_qubit_time = 50e-9  # 50 ns
        two_qubit_time = 300e-9    # 300 ns
        
        # Count gate types
        single_qubit_gates = sum(1 for inst in circuit.data 
                                if len(inst.qubits) == 1)
        two_qubit_gates = sum(1 for inst in circuit.data 
                             if len(inst.qubits) == 2)
        
        # Estimate based on circuit depth and parallelism
        return depth * (single_qubit_time + two_qubit_time)