# Quantum Module Structure

This document describes the structure and organization of the `quactuary.quantum` module.

## Overview

The quantum module provides quantum computing capabilities for actuarial calculations, leveraging Qiskit for quantum algorithm implementations. It follows a modular architecture with clear separation between different functional areas.

## Directory Structure

```
quantum/
├── __init__.py                 # Module initialization and exports
├── base_quantum.py             # Abstract base classes for quantum algorithms
├── quantum_types.py            # Type definitions and protocols
├── state_preparation/          # Quantum state preparation utilities
│   ├── __init__.py
│   ├── amplitude_encoding.py   # Amplitude encoding implementations
│   ├── probability_loader.py   # Distribution loading utilities
│   └── validation.py          # State validation functions
├── circuits/                   # Circuit construction utilities
│   ├── __init__.py
│   ├── builders.py            # Circuit builder patterns
│   └── templates.py           # Common circuit templates
├── algorithms/                 # Quantum algorithm implementations
│   ├── __init__.py
│   └── base_algorithm.py      # Base algorithm implementations
└── utils/                      # Utility functions
    ├── __init__.py
    ├── validation.py          # Circuit and parameter validation
    └── optimization.py        # Circuit optimization utilities
```

## Key Components

### Base Classes (`base_quantum.py`)

- **QuantumAlgorithm**: Abstract base class following Qiskit's 4-step workflow
  - `build_circuit()`: Map problem to quantum-native format
  - `optimize_circuit()`: Optimize circuits using transpile
  - `execute()`: Execute using quantum primitives
  - `analyze_results()`: Analyze quantum results

- **StatePreparationAlgorithm**: Specialized for state preparation tasks
- **VariationalQuantumAlgorithm**: Base for variational algorithms (VQE, QAOA)

### Type System (`quantum_types.py`)

Comprehensive type definitions including:
- `ProbabilityDistribution`: Type for probability distributions
- `StateVector`: Quantum state vector type
- `OptimizationLevel`: Circuit optimization level (0-3)
- Custom exceptions: `QuantumError`, `CircuitConstructionError`, etc.
- Protocol definitions for backends and primitives

### Circuit Building (`circuits/`)

- **CircuitBuilder**: Fluent interface for circuit construction
- **ParameterizedCircuitBuilder**: Specialized for variational circuits
- **Templates**: Pre-built circuits for common patterns:
  - Uniform superposition
  - GHZ states
  - QFT circuits
  - Amplitude encoding
  - Grover operators
  - Variational ansätze

### Algorithms (`algorithms/`)

- **ActuarialQuantumAlgorithm**: Base class with actuarial-specific functionality
- **ProbabilityDistributionLoader**: Load classical distributions into quantum states
- **MonteCarloQuantumAlgorithm**: Base for quantum Monte Carlo methods

### Utilities (`utils/`)

- **Validation**: Circuit size, probability distributions, quantum states
- **Optimization**: Circuit depth reduction, parameter optimization, measurement ordering

## Usage Examples

### Basic Circuit Construction

```python
from quactuary.quantum.circuits.builders import CircuitBuilder

# Build a simple quantum circuit
builder = CircuitBuilder(num_qubits=4)
circuit = (builder
    .add_hadamard_layer()
    .add_entangling_layer('linear')
    .add_measurement()
    .build())
```

### State Preparation

```python
from quactuary.quantum.algorithms.base_algorithm import ProbabilityDistributionLoader
import numpy as np

# Load a probability distribution
probs = np.array([0.1, 0.2, 0.3, 0.4])
loader = ProbabilityDistributionLoader(probs)
circuit = loader.build_circuit()
```

### Custom Algorithm Implementation

```python
from quactuary.quantum.base_quantum import QuantumAlgorithm
from qiskit import QuantumCircuit

class MyQuantumAlgorithm(QuantumAlgorithm):
    @property
    def required_qubits(self) -> int:
        return 5
    
    def build_circuit(self, **params) -> QuantumCircuit:
        # Implementation here
        pass
    
    def optimize_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
        # Use default optimization
        return super().optimize_circuit(circuit)
    
    def execute(self, circuit: QuantumCircuit) -> Dict[str, Any]:
        # Execute with sampler
        return self.sampler.run(circuit, shots=1000).result()
    
    def analyze_results(self, results: Dict[str, Any]) -> Any:
        # Process results
        pass
    
    def classical_equivalent(self, **params) -> Any:
        # Classical fallback
        pass
```

## Integration with Quactuary

The quantum module integrates seamlessly with the rest of quactuary:

1. **Backend System**: Works with `quactuary.backend` for quantum/classical switching
2. **Pricing Models**: Can be used within `PricingModel` for quantum-accelerated calculations
3. **Distributions**: Interfaces with `quactuary.distributions` for state preparation

## Best Practices

1. **Always provide classical fallback**: Implement `classical_equivalent()` method
2. **Validate inputs**: Use utilities in `utils.validation` 
3. **Optimize circuits**: Use `optimize_circuit()` before execution
4. **Handle errors gracefully**: Catch quantum-specific exceptions
5. **Document qubit requirements**: Clearly specify `required_qubits`

## Dependencies

- **Qiskit >= 1.4.2**: Core quantum computing framework
- **NumPy**: Numerical computations
- **SciPy**: Classical optimization (for variational algorithms)

## Future Extensions

The module is designed to be extensible. Future additions may include:
- Specific actuarial quantum algorithms (QAE for risk measures)
- Hardware backend integration
- Noise modeling and error mitigation
- Quantum machine learning interfaces