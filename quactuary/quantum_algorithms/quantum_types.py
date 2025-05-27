"""
Type definitions and protocols for quantum computing in quactuary.

This module provides a comprehensive type system for quantum algorithms, ensuring
type safety and clarity throughout the quantum components. It defines type aliases
for common quantum concepts, protocols for backend interfaces, and custom types
for algorithm results.

The type system helps:
- Ensure compatibility between different quantum components
- Provide clear interfaces for quantum algorithm implementations
- Enable static type checking with mypy
- Document expected data structures

Examples:
    Using type aliases for clarity:
        >>> from quactuary.quantum_algorithms.quantum_types import (
        ...     ProbabilityDistribution, StateVector, OptimizationLevel
        ... )
        >>> 
        >>> def prepare_state(
        ...     distribution: ProbabilityDistribution,
        ...     optimization: OptimizationLevel = 2
        ... ) -> StateVector:
        ...     # Type hints make the interface clear
        ...     pass
    
    Implementing a protocol:
        >>> from quactuary.quantum_algorithms.quantum_types import (
        ...     QuantumBackendProtocol, BackendResult
        ... )
        >>> 
        >>> class MyBackend:
        ...     def run(self, circuit: QuantumCircuit, shots: int = 1024) -> BackendResult:
        ...         # Implementation
        ...         return {'counts': {}, 'metadata': {}}
        ...     
        ...     @property
        ...     def configuration(self) -> Dict[str, Any]:
        ...         return {'n_qubits': 20, 'basis_gates': ['cx', 'u3']}
        ...     
        ...     @property 
        ...     def properties(self) -> Optional[Any]:
        ...         return None
        >>> 
        >>> # Type checker knows this satisfies QuantumBackendProtocol
        >>> backend: QuantumBackendProtocol = MyBackend()

Notes:
    - Type aliases use TypeAlias for Python 3.10+ compatibility
    - Protocols enable structural subtyping (duck typing with types)
    - Custom result classes provide structured data with validation

See Also:
    typing: Python's type hinting module
    qiskit.quantum_info: Quantum state representations
    numpy.typing: NumPy type annotations
"""

from typing import Union, List, Dict, TypeVar, Protocol, Optional, Tuple, Any
from typing_extensions import TypeAlias
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, DensityMatrix


# Type for probability distributions
ProbabilityDistribution: TypeAlias = Union[np.ndarray, List[float], Dict[int, float]]

# Quantum state vector type
StateVector: TypeAlias = np.ndarray

# Circuit optimization level (0-3 as per Qiskit standards)
OptimizationLevel: TypeAlias = int

# Measurement results type
MeasurementResults: TypeAlias = Dict[str, int]

# Parameter values for variational circuits
ParameterValues: TypeAlias = Union[np.ndarray, List[float], Dict[Parameter, float]]

# Quantum state representation
QuantumState: TypeAlias = Union[Statevector, DensityMatrix, np.ndarray]

# Backend result type
BackendResult: TypeAlias = Dict[str, Any]

# Quantum register size
QubitCount: TypeAlias = int

# Circuit depth
CircuitDepth: TypeAlias = int

# Fidelity value (0.0 to 1.0)
Fidelity: TypeAlias = float

# Error/noise probability
ErrorRate: TypeAlias = float

# Shot count for measurements
Shots: TypeAlias = int

# Basis gates for transpilation
BasisGates: TypeAlias = List[str]


class QuantumBackendProtocol(Protocol):
    """
    Protocol defining the interface for quantum backend implementations.
    
    This protocol ensures compatibility with different quantum backends (simulators,
    real hardware, cloud services) by defining the minimal interface required. Any
    class implementing these methods can be used as a quantum backend.
    
    Examples:
        Checking if an object satisfies the protocol:
            >>> from qiskit_aer import AerSimulator
            >>> backend = AerSimulator()
            >>> # Type checkers recognize this satisfies QuantumBackendProtocol
            >>> isinstance(backend, QuantumBackendProtocol)  # Runtime check
            
        Creating a mock backend:
            >>> class MockBackend:
            ...     def run(self, circuit, shots=1024):
            ...         return {'counts': {'00': shots//2, '11': shots//2}}
            ...     @property
            ...     def configuration(self):
            ...         return {'n_qubits': 5, 'simulator': True}
            ...     @property
            ...     def properties(self):
            ...         return None
    """
    
    def run(self, circuit: QuantumCircuit, shots: int = 1024) -> BackendResult:
        """
        Execute a quantum circuit on the backend.
        
        Args:
            circuit: The quantum circuit to execute. Should be transpiled for
                the target backend.
            shots: Number of measurement shots. More shots give better statistics
                but take longer. Default 1024.
                
        Returns:
            BackendResult: Dictionary containing at minimum:
                - 'counts': Dict[str, int] of measurement outcomes
                - 'metadata': Dict with execution details
                
        Notes:
            - Real backends may have shot limits (e.g., max 8192)
            - Some backends support multiple circuits in one call
        """
        ...
    
    @property
    def configuration(self) -> Dict[str, Any]:
        """
        Get backend configuration information.
        
        Returns:
            Dict containing:
                - 'n_qubits': Maximum number of qubits
                - 'basis_gates': List of supported gates
                - 'simulator': Whether this is a simulator
                - Other backend-specific properties
        """
        ...
    
    @property
    def properties(self) -> Optional[Any]:
        """
        Get backend properties like gate errors and connectivity.
        
        Returns:
            Backend properties object if available (real hardware),
            None for ideal simulators.
            
        Notes:
            Properties include gate error rates, readout errors, T1/T2 times
        """
        ...


class QuantumPrimitiveProtocol(Protocol):
    """Protocol for Qiskit primitives (Estimator, Sampler)."""
    
    def run(self, circuits: Union[QuantumCircuit, List[QuantumCircuit]], 
            **kwargs) -> Any:
        """Run the primitive on circuits."""
        ...


class OptimizationResult:
    """
    Container for results from variational quantum algorithm optimization.
    
    This class encapsulates all information from a classical optimization run
    in variational algorithms like VQE and QAOA. It provides a standardized
    way to access optimization outcomes, convergence information, and metadata.
    
    Attributes:
        optimal_value (float): The minimum (or maximum) function value found
        optimal_params (np.ndarray): Parameter values at the optimum
        num_iterations (int): Total iterations performed
        converged (bool): Whether optimization met convergence criteria
        metadata (Dict[str, Any]): Additional information like:
            - 'cost_history': List of cost values per iteration
            - 'grad_norm_history': Gradient norms (if applicable)
            - 'optimizer_name': Name of optimizer used
            - 'convergence_reason': Why optimization stopped
    
    Examples:
        Accessing optimization results:
            >>> result = OptimizationResult(
            ...     optimal_value=-1.274,
            ...     optimal_params=np.array([0.523, 1.103, -0.331]),
            ...     num_iterations=150,
            ...     converged=True,
            ...     metadata={
            ...         'cost_history': [-0.5, -0.8, -1.1, -1.274],
            ...         'optimizer_name': 'COBYLA',
            ...         'convergence_reason': 'ftol_reached'
            ...     }
            ... )
            >>> 
            >>> print(f"Found minimum: {result.optimal_value:.3f}")
            Found minimum: -1.274
            >>> print(f"Converged after {result.num_iterations} iterations")
            Converged after 150 iterations
            >>> 
            >>> # Plot convergence
            >>> import matplotlib.pyplot as plt
            >>> plt.plot(result.metadata['cost_history'])
            >>> plt.xlabel('Iteration')
            >>> plt.ylabel('Cost')
            >>> plt.title('Optimization Convergence')
        
        Checking convergence:
            >>> if not result.converged:
            ...     print(f"Warning: Did not converge!")
            ...     print(f"Reason: {result.metadata.get('convergence_reason', 'Unknown')}")
    """
    
    def __init__(self, 
                 optimal_value: float,
                 optimal_params: np.ndarray,
                 num_iterations: int,
                 converged: bool,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize optimization result container.
        
        Args:
            optimal_value: The optimal function value found. For minimization,
                this is the minimum; for maximization, the maximum.
            optimal_params: Array of parameter values that produced the optimal
                value. Length should match the number of variational parameters.
            num_iterations: Total number of optimizer iterations performed. This
                includes all function evaluations, not just improvements.
            converged: Whether the optimizer met its convergence criteria. True
                if converged, False if stopped due to iteration limit or other issue.
            metadata: Optional dictionary containing additional information:
                - 'cost_history': List[float] of cost values
                - 'time_elapsed': Total optimization time in seconds
                - 'function_evals': Number of function evaluations
                - Algorithm-specific data
        
        Examples:
            >>> # From a VQE run
            >>> result = OptimizationResult(
            ...     optimal_value=-1.85727,  # Ground state energy
            ...     optimal_params=np.array([3.14, 0.0, 1.57]),
            ...     num_iterations=200,
            ...     converged=True,
            ...     metadata={'variance': 0.0001}
            ... )
        """
        self.optimal_value = optimal_value
        self.optimal_params = optimal_params
        self.num_iterations = num_iterations
        self.converged = converged
        self.metadata = metadata or {}
    
    def __repr__(self) -> str:
        return (f"OptimizationResult(value={self.optimal_value:.6f}, "
                f"converged={self.converged}, iterations={self.num_iterations})")


class CircuitMetrics:
    """Container for quantum circuit metrics."""
    
    def __init__(self,
                 num_qubits: int,
                 depth: int,
                 gate_count: int,
                 cnot_count: int = 0,
                 single_qubit_gates: int = 0,
                 two_qubit_gates: int = 0):
        """
        Initialize circuit metrics.
        
        Args:
            num_qubits: Number of qubits in the circuit
            depth: Circuit depth
            gate_count: Total number of gates
            cnot_count: Number of CNOT gates
            single_qubit_gates: Number of single-qubit gates
            two_qubit_gates: Number of two-qubit gates
        """
        self.num_qubits = num_qubits
        self.depth = depth
        self.gate_count = gate_count
        self.cnot_count = cnot_count
        self.single_qubit_gates = single_qubit_gates
        self.two_qubit_gates = two_qubit_gates
    
    @classmethod
    def from_circuit(cls, circuit: QuantumCircuit) -> "CircuitMetrics":
        """
        Create CircuitMetrics from a QuantumCircuit.
        
        Args:
            circuit: The quantum circuit to analyze
            
        Returns:
            CircuitMetrics instance
        """
        cnot_count = sum(1 for inst in circuit.data 
                        if inst.operation.name in ['cx', 'cnot'])
        
        single_qubit_count = sum(1 for inst in circuit.data 
                                if len(inst.qubits) == 1)
        
        two_qubit_count = sum(1 for inst in circuit.data 
                             if len(inst.qubits) == 2)
        
        return cls(
            num_qubits=circuit.num_qubits,
            depth=circuit.depth(),
            gate_count=len(circuit.data),
            cnot_count=cnot_count,
            single_qubit_gates=single_qubit_count,
            two_qubit_gates=two_qubit_count
        )
    
    def __repr__(self) -> str:
        return (f"CircuitMetrics(qubits={self.num_qubits}, depth={self.depth}, "
                f"gates={self.gate_count}, cnots={self.cnot_count})")


# Type variable for generic quantum algorithms
T = TypeVar('T')

# Result type for quantum algorithms
QuantumResult: TypeAlias = Union[float, np.ndarray, Dict[str, Any]]

# Hamiltonian representation
Hamiltonian: TypeAlias = Any  # Will be SparsePauliOp in practice

# Quantum error/exception types
class QuantumError(Exception):
    """Base exception for quantum-related errors."""
    pass


class CircuitConstructionError(QuantumError):
    """Error in quantum circuit construction."""
    pass


class StatePreparationError(QuantumError):
    """Error in quantum state preparation."""
    pass


class BackendError(QuantumError):
    """Error related to quantum backend execution."""
    pass


class OptimizationError(QuantumError):
    """Error in quantum algorithm optimization."""
    pass


# Constants
MAX_QUBITS_SIMULATOR = 30  # Maximum qubits for statevector simulator
DEFAULT_SHOTS = 1024       # Default number of measurement shots
DEFAULT_OPTIMIZATION_LEVEL = 3  # Default transpiler optimization level

# Basis gates for common backends
IBM_BASIS_GATES = ['id', 'rz', 'sx', 'x', 'cx']
SIMULATOR_BASIS_GATES = ['u1', 'u2', 'u3', 'cx', 'id']