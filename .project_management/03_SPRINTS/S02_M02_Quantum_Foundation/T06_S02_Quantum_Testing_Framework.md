---
task_id: T06_S02
sprint_sequence_id: S02
status: open
complexity: Medium
last_updated: 2025-01-25T00:00:00Z
---

# Task: Quantum Testing Framework

## Description
Establish a testing framework specifically for quantum algorithms and circuits. This includes utilities for quantum state verification, circuit equivalence testing, and quantum-specific assertions.

## Goal / Objectives
- Create quantum-specific testing utilities
- Implement state fidelity checking
- Build circuit equivalence testing
- Establish quantum test fixtures and helpers

## Acceptance Criteria
- [ ] Quantum test base class created
- [ ] State comparison utilities implemented
- [ ] Circuit equivalence testing functional
- [ ] Quantum assertions library available
- [ ] Integration with pytest framework

## Subtasks
- [ ] Create QuantumTestCase base class
- [ ] Implement quantum state comparison functions
- [ ] Add fidelity and distance metrics
- [ ] Create circuit equivalence checker
- [ ] Build quantum test fixtures (common states, circuits)
- [ ] Add quantum-specific pytest markers
- [ ] Write example tests demonstrating usage

## Implementation Guidelines

### Quantum Test Base Class
```python
# quactuary/quantum/testing/base_test.py
import unittest
import numpy as np
from typing import Optional, Tuple
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit.quantum_info import process_fidelity, Operator

class QuantumTestCase(unittest.TestCase):
    """
    Base class for quantum algorithm testing.
    Provides quantum-specific assertions and utilities.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.tolerance = 1e-10
        self.fidelity_threshold = 0.99
        
    def assertStateEqual(self, state1: Statevector, state2: Statevector, 
                        tolerance: float = None):
        """Assert two quantum states are equal."""
        tolerance = tolerance or self.tolerance
        fidelity = state_fidelity(state1, state2)
        self.assertAlmostEqual(fidelity, 1.0, delta=tolerance,
                              msg=f"States not equal: fidelity={fidelity}")
    
    def assertStateFidelity(self, state1: Statevector, state2: Statevector,
                           min_fidelity: float = None):
        """Assert states have minimum fidelity."""
        min_fidelity = min_fidelity or self.fidelity_threshold
        fidelity = state_fidelity(state1, state2)
        self.assertGreaterEqual(fidelity, min_fidelity,
                               msg=f"Fidelity {fidelity} below threshold {min_fidelity}")
    
    def assertCircuitEquivalent(self, circuit1: QuantumCircuit, 
                               circuit2: QuantumCircuit):
        """Assert two circuits are functionally equivalent."""
        # Convert to operators
        op1 = Operator(circuit1)
        op2 = Operator(circuit2)
        
        # Check process fidelity
        fidelity = process_fidelity(op1, op2)
        self.assertAlmostEqual(fidelity, 1.0, delta=self.tolerance,
                              msg=f"Circuits not equivalent: fidelity={fidelity}")
    
    def assertProbabilityDistribution(self, probs: np.ndarray, 
                                     expected: np.ndarray,
                                     tolerance: float = 0.01):
        """Assert probability distributions match within tolerance."""
        # Check normalization
        self.assertAlmostEqual(np.sum(probs), 1.0, delta=1e-10)
        self.assertAlmostEqual(np.sum(expected), 1.0, delta=1e-10)
        
        # Check individual probabilities
        for i, (p, e) in enumerate(zip(probs, expected)):
            self.assertAlmostEqual(p, e, delta=tolerance,
                                  msg=f"Probability mismatch at index {i}: {p} != {e}")
```

### Quantum State Comparison Utilities
```python
# quactuary/quantum/testing/state_utils.py
import numpy as np
from qiskit.quantum_info import Statevector, DensityMatrix
from typing import Union, Dict

def compare_quantum_states(state1: Union[Statevector, np.ndarray],
                          state2: Union[Statevector, np.ndarray]) -> Dict[str, float]:
    """
    Compare two quantum states using multiple metrics.
    
    Returns:
        Dictionary with comparison metrics:
        - fidelity: State fidelity (0-1)
        - trace_distance: Trace distance (0-1)
        - hellinger_distance: Hellinger distance
    """
    # Convert to Statevector if needed
    if isinstance(state1, np.ndarray):
        state1 = Statevector(state1)
    if isinstance(state2, np.ndarray):
        state2 = Statevector(state2)
    
    # Calculate metrics
    fidelity = state_fidelity(state1, state2)
    
    # Convert to density matrices for trace distance
    rho1 = DensityMatrix(state1)
    rho2 = DensityMatrix(state2)
    trace_distance = 0.5 * np.trace(np.abs((rho1 - rho2).data))
    
    # Hellinger distance
    hellinger = np.sqrt(1 - np.sqrt(fidelity))
    
    return {
        'fidelity': fidelity,
        'trace_distance': trace_distance,
        'hellinger_distance': hellinger
    }

def validate_probability_encoding(circuit: QuantumCircuit, 
                                 target_probs: np.ndarray) -> bool:
    """
    Validate that a circuit correctly encodes a probability distribution.
    """
    # Get statevector from circuit
    statevector = Statevector.from_instruction(circuit)
    
    # Extract probabilities
    circuit_probs = statevector.probabilities()
    
    # Compare distributions
    return np.allclose(circuit_probs, target_probs, rtol=1e-5, atol=1e-8)
```

### Circuit Equivalence Testing
```python
# quactuary/quantum/testing/circuit_utils.py
from qiskit import transpile
from qiskit.quantum_info import Operator, average_gate_fidelity

class CircuitEquivalenceChecker:
    """Check equivalence between quantum circuits."""
    
    @staticmethod
    def are_equivalent(circuit1: QuantumCircuit, 
                      circuit2: QuantumCircuit,
                      check_phase: bool = True) -> Tuple[bool, float]:
        """
        Check if two circuits are equivalent.
        
        Returns:
            (is_equivalent, fidelity)
        """
        # Ensure same number of qubits
        if circuit1.num_qubits != circuit2.num_qubits:
            return False, 0.0
        
        # Convert to operators
        op1 = Operator(circuit1)
        op2 = Operator(circuit2)
        
        if check_phase:
            # Check exact equivalence
            fidelity = process_fidelity(op1, op2)
        else:
            # Check up to global phase
            fidelity = average_gate_fidelity(op1, op2)
        
        is_equivalent = np.isclose(fidelity, 1.0, rtol=1e-10)
        return is_equivalent, fidelity
    
    @staticmethod
    def verify_optimization(original: QuantumCircuit, 
                           optimized: QuantumCircuit) -> Dict[str, Any]:
        """
        Verify that circuit optimization preserves functionality.
        """
        # Check equivalence
        is_equiv, fidelity = CircuitEquivalenceChecker.are_equivalent(
            original, optimized
        )
        
        # Compare circuit metrics
        return {
            'equivalent': is_equiv,
            'fidelity': fidelity,
            'depth_reduction': original.depth() - optimized.depth(),
            'gate_reduction': original.size() - optimized.size(),
            'original_depth': original.depth(),
            'optimized_depth': optimized.depth()
        }
```

### Quantum Test Fixtures
```python
# quactuary/quantum/testing/fixtures.py
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy as np

class QuantumTestFixtures:
    """Common quantum states and circuits for testing."""
    
    @staticmethod
    def bell_states() -> Dict[str, Statevector]:
        """Standard Bell states."""
        return {
            'phi_plus': Statevector([1, 0, 0, 1]) / np.sqrt(2),
            'phi_minus': Statevector([1, 0, 0, -1]) / np.sqrt(2),
            'psi_plus': Statevector([0, 1, 1, 0]) / np.sqrt(2),
            'psi_minus': Statevector([0, 1, -1, 0]) / np.sqrt(2)
        }
    
    @staticmethod
    def ghz_state(n_qubits: int) -> Statevector:
        """N-qubit GHZ state."""
        state = np.zeros(2**n_qubits)
        state[0] = 1/np.sqrt(2)
        state[-1] = 1/np.sqrt(2)
        return Statevector(state)
    
    @staticmethod
    def random_statevector(n_qubits: int, seed: Optional[int] = None) -> Statevector:
        """Generate random quantum state."""
        if seed is not None:
            np.random.seed(seed)
        
        # Random complex amplitudes
        real = np.random.randn(2**n_qubits)
        imag = np.random.randn(2**n_qubits)
        state = real + 1j * imag
        
        # Normalize
        state = state / np.linalg.norm(state)
        
        return Statevector(state)
    
    @staticmethod
    def test_distributions() -> Dict[str, np.ndarray]:
        """Common probability distributions for testing."""
        return {
            'uniform': np.ones(8) / 8,
            'binomial': np.array([1, 7, 21, 35, 35, 21, 7, 1]) / 128,
            'exponential': np.exp(-np.arange(8)) / np.sum(np.exp(-np.arange(8))),
            'sparse': np.array([0.9, 0.05, 0.03, 0.02, 0, 0, 0, 0])
        }
```

### Integration with pytest
```python
# quactuary/quantum/testing/pytest_quantum.py
import pytest
from typing import Any

# Custom pytest markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "quantum: mark test as quantum algorithm test"
    )
    config.addinivalue_line(
        "markers", "slow_quantum: mark test as slow quantum test"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: mark test as requiring GPU"
    )

# Quantum-specific fixtures
@pytest.fixture
def quantum_backend():
    """Provide default quantum backend."""
    from quactuary.quantum.backends import BackendFactory
    return BackendFactory.create_backend('statevector')

@pytest.fixture
def bell_state():
    """Provide Bell state for testing."""
    return QuantumTestFixtures.bell_states()['phi_plus']

@pytest.fixture
def random_circuit(n_qubits: int = 4):
    """Generate random quantum circuit."""
    from qiskit.circuit.random import random_circuit
    return random_circuit(n_qubits, depth=10, seed=42)
```

### Example Test Usage
```python
# Example test file using the framework
class TestQuantumStatePreparation(QuantumTestCase):
    """Test quantum state preparation utilities."""
    
    def test_amplitude_encoding(self):
        """Test amplitude encoding of probability distribution."""
        # Test distribution
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        
        # Encode
        circuit = amplitude_encode(probs)
        
        # Verify
        statevector = Statevector.from_instruction(circuit)
        encoded_probs = statevector.probabilities()
        
        # Assert using custom assertion
        self.assertProbabilityDistribution(encoded_probs, probs)
    
    @pytest.mark.quantum
    def test_circuit_optimization_preserves_state(self):
        """Test that optimization doesn't change circuit behavior."""
        # Create test circuit
        original = create_test_circuit()
        
        # Optimize
        optimized = optimize_circuit(original)
        
        # Verify equivalence
        self.assertCircuitEquivalent(original, optimized)
```

## Output Log
*(This section is populated as work progresses on the task)*