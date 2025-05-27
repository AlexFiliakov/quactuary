"""
Common circuit templates for quantum algorithms.

This module provides pre-built circuit templates for common quantum
computing patterns used in actuarial applications.
"""

from typing import List, Optional, Union, Callable
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import (
    EfficientSU2, TwoLocal, RealAmplitudes, 
    UniformDistribution, NormalDistribution
)
import numpy as np


def create_uniform_superposition(num_qubits: int) -> QuantumCircuit:
    """
    Create a circuit that prepares uniform superposition state.
    
    Args:
        num_qubits: Number of qubits
        
    Returns:
        QuantumCircuit preparing |+>^n state
    """
    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))
    return qc


def create_ghz_state(num_qubits: int) -> QuantumCircuit:
    """
    Create a circuit that prepares GHZ state.
    
    The GHZ state is (|00...0> + |11...1>) / sqrt(2).
    
    Args:
        num_qubits: Number of qubits (minimum 2)
        
    Returns:
        QuantumCircuit preparing GHZ state
    """
    if num_qubits < 2:
        raise ValueError("GHZ state requires at least 2 qubits")
        
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    for i in range(1, num_qubits):
        qc.cx(0, i)
    
    return qc


def create_qft_circuit(num_qubits: int, inverse: bool = False,
                      insert_barriers: bool = True) -> QuantumCircuit:
    """
    Create Quantum Fourier Transform circuit.
    
    Args:
        num_qubits: Number of qubits
        inverse: If True, create inverse QFT
        insert_barriers: If True, insert barriers between stages
        
    Returns:
        QFT or inverse QFT circuit
    """
    qc = QuantumCircuit(num_qubits)
    
    def qft_rotations(circuit, n):
        """Apply QFT rotations."""
        if n == 0:
            return
        n -= 1
        circuit.h(n)
        for qubit in range(n):
            circuit.cp(np.pi/2**(n-qubit), qubit, n)
        if insert_barriers:
            circuit.barrier()
        qft_rotations(circuit, n)
    
    def swap_qubits(circuit, n):
        """Swap qubits for proper QFT output."""
        for qubit in range(n//2):
            circuit.swap(qubit, n-qubit-1)
        if insert_barriers:
            circuit.barrier()
    
    # Build QFT
    qft_rotations(qc, num_qubits)
    swap_qubits(qc, num_qubits)
    
    if inverse:
        qc = qc.inverse()
        
    return qc


def create_amplitude_encoding_circuit(amplitudes: Union[List[float], np.ndarray],
                                    normalize: bool = True) -> QuantumCircuit:
    """
    Create circuit for amplitude encoding of classical data.
    
    Args:
        amplitudes: Classical data to encode (length must be power of 2)
        normalize: Whether to normalize the amplitudes
        
    Returns:
        Circuit that prepares state with given amplitudes
    """
    amplitudes = np.asarray(amplitudes)
    
    # Check if length is power of 2
    n = len(amplitudes)
    if n & (n - 1) != 0:
        raise ValueError(f"Number of amplitudes ({n}) must be a power of 2")
        
    # Normalize if requested
    if normalize:
        norm = np.linalg.norm(amplitudes)
        if norm > 0:
            amplitudes = amplitudes / norm
            
    # Calculate required qubits
    num_qubits = int(np.log2(n))
    
    # Create circuit
    qc = QuantumCircuit(num_qubits)
    qc.initialize(amplitudes, range(num_qubits))
    
    return qc


def create_probability_distribution_loader(probabilities: Union[List[float], np.ndarray],
                                         num_qubits: Optional[int] = None) -> QuantumCircuit:
    """
    Create circuit that loads a probability distribution.
    
    Args:
        probabilities: Probability values (will be normalized)
        num_qubits: Number of qubits (auto-calculated if None)
        
    Returns:
        Circuit that prepares state encoding the distribution
    """
    probabilities = np.asarray(probabilities)
    
    # Normalize to ensure valid probability distribution
    probabilities = probabilities / np.sum(probabilities)
    
    # Calculate amplitudes (square root of probabilities)
    amplitudes = np.sqrt(probabilities)
    
    # Pad with zeros if needed
    if num_qubits is not None:
        target_length = 2**num_qubits
        if len(amplitudes) < target_length:
            amplitudes = np.pad(amplitudes, (0, target_length - len(amplitudes)))
        elif len(amplitudes) > target_length:
            raise ValueError(f"Too many probabilities ({len(amplitudes)}) "
                           f"for {num_qubits} qubits")
    
    return create_amplitude_encoding_circuit(amplitudes, normalize=False)


def create_grover_oracle(marked_states: List[int], num_qubits: int) -> QuantumCircuit:
    """
    Create Grover oracle for marked states.
    
    Args:
        marked_states: List of computational basis states to mark
        num_qubits: Total number of qubits
        
    Returns:
        Oracle circuit that flips phase of marked states
    """
    qc = QuantumCircuit(num_qubits)
    
    for state in marked_states:
        # Convert state number to binary string
        state_str = format(state, f'0{num_qubits}b')
        
        # Add X gates to flip qubits where bit is 0
        for i, bit in enumerate(state_str):
            if bit == '0':
                qc.x(i)
                
        # Multi-controlled Z gate
        if num_qubits > 1:
            qc.h(num_qubits - 1)
            qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
            qc.h(num_qubits - 1)
        else:
            qc.z(0)
            
        # Undo X gates
        for i, bit in enumerate(state_str):
            if bit == '0':
                qc.x(i)
                
    return qc


def create_diffusion_operator(num_qubits: int) -> QuantumCircuit:
    """
    Create Grover diffusion operator.
    
    Args:
        num_qubits: Number of qubits
        
    Returns:
        Diffusion operator circuit
    """
    qc = QuantumCircuit(num_qubits)
    
    # Apply Hadamard gates
    qc.h(range(num_qubits))
    
    # Apply X gates
    qc.x(range(num_qubits))
    
    # Multi-controlled Z gate
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    
    # Apply X gates
    qc.x(range(num_qubits))
    
    # Apply Hadamard gates
    qc.h(range(num_qubits))
    
    return qc


def create_variational_ansatz(num_qubits: int, depth: int = 3,
                            entanglement: str = 'linear',
                            rotation_blocks: Optional[List[str]] = None) -> QuantumCircuit:
    """
    Create a variational ansatz circuit.
    
    Args:
        num_qubits: Number of qubits
        depth: Number of repetitions
        entanglement: Entanglement pattern ('linear', 'circular', 'full')
        rotation_blocks: List of rotation gates (default: ['ry', 'rz'])
        
    Returns:
        Parameterized variational circuit
    """
    if rotation_blocks is None:
        rotation_blocks = ['ry', 'rz']
        
    # Use TwoLocal from Qiskit
    ansatz = TwoLocal(
        num_qubits=num_qubits,
        rotation_blocks=rotation_blocks,
        entanglement_blocks='cx',
        entanglement=entanglement,
        reps=depth,
        insert_barriers=True
    )
    
    return ansatz


def create_hardware_efficient_ansatz(num_qubits: int, depth: int = 3) -> QuantumCircuit:
    """
    Create hardware-efficient ansatz optimized for NISQ devices.
    
    Args:
        num_qubits: Number of qubits
        depth: Circuit depth
        
    Returns:
        Hardware-efficient parameterized circuit
    """
    return EfficientSU2(
        num_qubits=num_qubits,
        reps=depth,
        entanglement='linear',
        insert_barriers=True
    )


def create_phase_estimation_circuit(unitary: QuantumCircuit,
                                  num_counting_qubits: int) -> QuantumCircuit:
    """
    Create quantum phase estimation circuit.
    
    Args:
        unitary: Unitary operator as a circuit
        num_counting_qubits: Number of counting qubits (precision)
        
    Returns:
        Phase estimation circuit
    """
    # Get number of qubits in unitary
    num_state_qubits = unitary.num_qubits
    
    # Create registers
    qr_counting = QuantumRegister(num_counting_qubits, 'counting')
    qr_state = QuantumRegister(num_state_qubits, 'state')
    
    qc = QuantumCircuit(qr_counting, qr_state)
    
    # Initialize counting qubits in superposition
    qc.h(qr_counting)
    
    # Controlled unitary operations
    for i in range(num_counting_qubits):
        controlled_unitary = unitary.control(1)
        qc.append(
            controlled_unitary.power(2**i),
            [qr_counting[i]] + list(qr_state)
        )
        
    # Apply inverse QFT
    qft_inv = create_qft_circuit(num_counting_qubits, inverse=True)
    qc.append(qft_inv, qr_counting)
    
    return qc