"""
Base algorithm implementations for quantum actuarial computations.

This module extends the base quantum classes with specific algorithm
implementations common to actuarial applications.
"""

from typing import Dict, Any, Optional, Union, List
import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.primitives import Estimator, Sampler
from qiskit.transpiler import generate_preset_pass_manager

from quactuary.quantum.base_quantum import (
    QuantumAlgorithm, 
    StatePreparationAlgorithm,
    VariationalQuantumAlgorithm
)
from quactuary.quantum.quantum_types import (
    OptimizationLevel, 
    CircuitMetrics,
    DEFAULT_OPTIMIZATION_LEVEL
)


class ActuarialQuantumAlgorithm(QuantumAlgorithm):
    """
    Base class for actuarial-specific quantum algorithms.
    
    This class extends QuantumAlgorithm with actuarial-specific
    functionality and default implementations.
    """
    
    def __init__(self, backend: Optional[Backend] = None,
                 optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL):
        """
        Initialize actuarial quantum algorithm.
        
        Args:
            backend: Optional quantum backend
            optimization_level: Transpiler optimization level (0-3)
        """
        super().__init__(backend)
        self.optimization_level = optimization_level
        self._estimator: Optional[Estimator] = None
        self._sampler: Optional[Sampler] = None
        
    @property
    def estimator(self) -> Estimator:
        """Get or create Estimator primitive."""
        if self._estimator is None:
            self._estimator = Estimator()
        return self._estimator
        
    @property
    def sampler(self) -> Sampler:
        """Get or create Sampler primitive."""
        if self._sampler is None:
            self._sampler = Sampler()
        return self._sampler
    
    def optimize_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Default implementation of circuit optimization.
        
        Args:
            circuit: Circuit to optimize
            
        Returns:
            Optimized circuit
        """
        if self.backend is None:
            # Return circuit as-is if no backend specified
            return circuit
            
        # Use Qiskit's preset pass manager
        pm = generate_preset_pass_manager(
            backend=self.backend,
            optimization_level=self.optimization_level
        )
        
        return pm.run(circuit)
    
    def validate_input(self, **params) -> None:
        """
        Validate input parameters for the algorithm.
        
        This method should be overridden by subclasses to implement
        specific validation logic.
        
        Args:
            **params: Algorithm-specific parameters
            
        Raises:
            ValueError: If parameters are invalid
        """
        pass
    
    def get_resource_requirements(self, **params) -> Dict[str, Any]:
        """
        Estimate resource requirements for the algorithm.
        
        Args:
            **params: Algorithm-specific parameters
            
        Returns:
            Dictionary with resource estimates
        """
        return {
            "num_qubits": self.required_qubits,
            "circuit_depth": None,  # To be calculated after circuit building
            "estimated_time": None,  # Backend-specific
            "memory_requirements": None,  # Implementation-specific
        }


class ProbabilityDistributionLoader(StatePreparationAlgorithm):
    """
    Quantum algorithm for loading probability distributions.
    
    This algorithm prepares quantum states that encode classical
    probability distributions for use in quantum algorithms.
    """
    
    def __init__(self, probabilities: Union[List[float], np.ndarray],
                 **kwargs):
        """
        Initialize probability distribution loader.
        
        Args:
            probabilities: Classical probability distribution
            **kwargs: Additional arguments for base class
        """
        super().__init__(**kwargs)
        self.probabilities = np.asarray(probabilities)
        self._normalize_probabilities()
        
    def _normalize_probabilities(self):
        """Normalize probabilities to ensure valid distribution."""
        total = np.sum(self.probabilities)
        if total > 0:
            self.probabilities = self.probabilities / total
        else:
            raise ValueError("Probabilities sum to zero")
            
    @property
    def required_qubits(self) -> int:
        """Calculate required number of qubits."""
        n = len(self.probabilities)
        return int(np.ceil(np.log2(n)))
    
    def build_circuit(self, **params) -> QuantumCircuit:
        """
        Build circuit for probability distribution loading.
        
        Returns:
            Quantum circuit that prepares the distribution
        """
        from quactuary.quantum.circuits.templates import (
            create_probability_distribution_loader
        )
        
        self._circuit = create_probability_distribution_loader(
            self.probabilities,
            num_qubits=self.required_qubits
        )
        
        return self._circuit
    
    def execute(self, circuit: QuantumCircuit) -> Dict[str, Any]:
        """
        Execute the state preparation circuit.
        
        Args:
            circuit: The circuit to execute
            
        Returns:
            Execution results including statevector
        """
        from qiskit.quantum_info import Statevector
        
        # For state preparation, we typically want the statevector
        statevector = Statevector.from_instruction(circuit)
        
        return {
            "statevector": statevector,
            "probabilities": np.abs(statevector.data)**2
        }
    
    def analyze_results(self, results: Dict[str, Any]) -> np.ndarray:
        """
        Extract prepared probability distribution.
        
        Args:
            results: Execution results
            
        Returns:
            Prepared probability distribution
        """
        probs = results["probabilities"]
        # Truncate to original distribution size
        return probs[:len(self.probabilities)]
    
    def get_fidelity(self, target_state: np.ndarray) -> float:
        """
        Calculate fidelity with target distribution.
        
        Args:
            target_state: Target probability distribution
            
        Returns:
            Fidelity value
        """
        if self._circuit is None:
            raise ValueError("Circuit not built yet")
            
        results = self.execute(self._circuit)
        prepared = self.analyze_results(results)
        
        # Calculate fidelity as overlap
        return np.sum(np.sqrt(prepared * target_state))
    
    def get_prepared_state(self) -> np.ndarray:
        """Get the prepared quantum state."""
        if self._circuit is None:
            raise ValueError("Circuit not built yet")
            
        results = self.execute(self._circuit)
        return results["statevector"].data
    
    def classical_equivalent(self, **params) -> np.ndarray:
        """Return the classical probability distribution."""
        return self.probabilities.copy()


class MonteCarloQuantumAlgorithm(ActuarialQuantumAlgorithm):
    """
    Base class for quantum Monte Carlo algorithms.
    
    This class provides common functionality for quantum algorithms
    that perform Monte Carlo-style sampling or estimation.
    """
    
    def __init__(self, num_samples: int = 1000, **kwargs):
        """
        Initialize Monte Carlo quantum algorithm.
        
        Args:
            num_samples: Number of samples for estimation
            **kwargs: Additional arguments for base class
        """
        super().__init__(**kwargs)
        self.num_samples = num_samples
        
    def execute(self, circuit: QuantumCircuit) -> Dict[str, Any]:
        """
        Execute circuit with sampling.
        
        Args:
            circuit: Circuit to execute
            
        Returns:
            Sampling results
        """
        # Add measurements if not present
        if circuit.num_clbits == 0:
            measured_circuit = circuit.copy()
            measured_circuit.add_register(
                circuit.num_qubits * [0]  # Classical register
            )
            measured_circuit.measure_all()
        else:
            measured_circuit = circuit
            
        # Run with sampler
        job = self.sampler.run(measured_circuit, shots=self.num_samples)
        result = job.result()
        
        return {
            "counts": result.quasi_dists[0],
            "metadata": result.metadata
        }
    
    def estimate_expectation(self, observable: Any,
                           circuit: Optional[QuantumCircuit] = None) -> float:
        """
        Estimate expectation value of an observable.
        
        Args:
            observable: Quantum observable
            circuit: Circuit to use (or use internal circuit)
            
        Returns:
            Estimated expectation value
        """
        if circuit is None:
            circuit = self._circuit
            
        if circuit is None:
            raise ValueError("No circuit available")
            
        job = self.estimator.run(circuit, observable)
        result = job.result()
        
        return result.values[0]