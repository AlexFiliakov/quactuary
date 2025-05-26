---
task_id: T04_S02
sprint_sequence_id: S02
status: open
complexity: Medium
last_updated: 2025-01-25T00:00:00Z
---

# Task: Quantum Circuit Builders

## Description
Create reusable quantum circuit building blocks and utilities that will be used across different quantum algorithms. This includes common gates, controlled operations, and circuit composition utilities.

## Goal / Objectives
- Implement quantum circuit builder patterns
- Create reusable quantum gate combinations
- Build utilities for circuit optimization
- Establish circuit validation and debugging tools

## Acceptance Criteria
- [ ] Circuit builder class implemented with fluent interface
- [ ] Common multi-qubit gate patterns available
- [ ] Circuit composition utilities functional
- [ ] Circuit depth optimization utilities included
- [ ] Debugging and visualization helpers implemented

## Subtasks
- [ ] Create QuantumCircuitBuilder class
- [ ] Implement common gate patterns (QFT, controlled rotations, etc.)
- [ ] Add circuit composition methods (append, prepend, insert)
- [ ] Create circuit optimization utilities (gate fusion, cancellation)
- [ ] Implement circuit visualization helpers
- [ ] Add circuit validation methods
- [ ] Write comprehensive tests for circuit builders

## Implementation Guidelines

### QuantumCircuitBuilder Class
Following Qiskit best practices for efficient circuit construction:

```python
# quactuary/quantum/circuits/builders.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from typing import List, Optional, Union

class QuantumCircuitBuilder:
    """
    Fluent interface for building quantum circuits.
    Implements Qiskit best practices for efficiency.
    """
    
    def __init__(self, num_qubits: int, num_classical: int = 0, 
                 name: str = "circuit"):
        self.qreg = QuantumRegister(num_qubits, name=f"{name}_q")
        self.creg = None
        if num_classical > 0:
            self.creg = ClassicalRegister(num_classical, name=f"{name}_c")
            self.circuit = QuantumCircuit(self.qreg, self.creg, name=name)
        else:
            self.circuit = QuantumCircuit(self.qreg, name=name)
    
    def add_gate(self, gate, qubits: Union[int, List[int]], 
                 copy: bool = False):
        """Add gate with copy=False by default for efficiency."""
        self.circuit.append(gate, qubits, copy=copy)
        return self
    
    def compose_with(self, other: QuantumCircuit, qubits: List[int] = None,
                     inplace: bool = True):
        """Compose with inplace=True by default."""
        self.circuit.compose(other, qubits=qubits, inplace=inplace)
        return self
    
    def add_qft(self, qubits: List[int], inverse: bool = False):
        """Add Quantum Fourier Transform."""
        from qiskit.circuit.library import QFT
        qft = QFT(len(qubits), inverse=inverse)
        self.circuit.append(qft, qubits, copy=False)
        return self
    
    def add_controlled_rotation(self, angle: float, 
                                control: Union[int, List[int]], 
                                target: int, axis: str = 'y'):
        """Add controlled rotation gates."""
        if isinstance(control, int):
            control = [control]
        
        if axis == 'x':
            self.circuit.mcrx(angle, control, target)
        elif axis == 'y':
            self.circuit.mcry(angle, control, target)
        elif axis == 'z':
            self.circuit.mcrz(angle, control, target)
        
        return self
    
    def build(self) -> QuantumCircuit:
        """Return the built circuit."""
        return self.circuit
```

### Common Circuit Templates
```python
# quactuary/quantum/circuits/templates.py
from qiskit import QuantumCircuit
import numpy as np

def create_bell_state(theta: float = 0) -> QuantumCircuit:
    """Create parameterized Bell state."""
    qc = QuantumCircuit(2)
    qc.ry(theta, 0)
    qc.h(0)
    qc.cx(0, 1)
    return qc

def create_ghz_state(num_qubits: int) -> QuantumCircuit:
    """Create GHZ state efficiently."""
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
    return qc

def create_amplitude_estimation_circuit(oracle, num_iterations: int) -> QuantumCircuit:
    """Template for amplitude estimation."""
    # Following Qiskit patterns for AE
    num_qubits = oracle.num_qubits
    qc = QuantumCircuit(num_qubits + 1)  # +1 for ancilla
    
    # Initialize superposition
    qc.h(range(num_qubits))
    
    # Apply Grover operator iterations
    for _ in range(num_iterations):
        qc.append(oracle, range(num_qubits), copy=False)
        # Apply diffusion operator
        qc.h(range(num_qubits))
        qc.x(range(num_qubits))
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
        qc.x(range(num_qubits))
        qc.h(range(num_qubits))
    
    return qc
```

### Circuit Optimization Utilities
```python
# quactuary/quantum/utils/optimization.py
from qiskit import transpile
from qiskit.circuit import QuantumCircuit

def optimize_circuit(circuit: QuantumCircuit, 
                    optimization_level: int = 3,
                    basis_gates: List[str] = None) -> QuantumCircuit:
    """
    Optimize circuit following Qiskit best practices.
    
    Optimization levels:
    - 0: No optimization
    - 1: Light optimization
    - 2: Medium optimization (default in Qiskit)
    - 3: Heavy optimization (best for depth reduction)
    """
    if basis_gates is None:
        # Default gates for simulators
        basis_gates = ['id', 'rz', 'sx', 'cx', 'reset']
    
    return transpile(
        circuit,
        basis_gates=basis_gates,
        optimization_level=optimization_level,
        layout_method='dense',  # Good for small circuits
        routing_method='stochastic'  # Good general purpose
    )

def reduce_circuit_depth(circuit: QuantumCircuit) -> QuantumCircuit:
    """Specific optimizations for depth reduction."""
    # Use commutation analysis
    from qiskit.transpiler.passes import CommutationAnalysis, CommutativeCancellation
    
    # Apply specific passes for depth reduction
    optimized = transpile(
        circuit,
        optimization_level=3,
        basis_gates=['u1', 'u2', 'u3', 'cx'],
        coupling_map=None,  # All-to-all connectivity for simulators
    )
    
    return optimized
```

### Circuit Validation and Debugging
```python
def validate_circuit(circuit: QuantumCircuit) -> Dict[str, Any]:
    """Validate and analyze circuit properties."""
    return {
        'num_qubits': circuit.num_qubits,
        'depth': circuit.depth(),
        'size': circuit.size(),
        'num_parameters': circuit.num_parameters,
        'gate_counts': circuit.count_ops(),
        'is_valid': circuit.num_qubits > 0 and circuit.size() > 0
    }

def debug_circuit(circuit: QuantumCircuit, verbose: bool = True):
    """Debug helper for circuit inspection."""
    if verbose:
        print(f"Circuit: {circuit.name}")
        print(f"Qubits: {circuit.num_qubits}")
        print(f"Classical bits: {circuit.num_clbits}")
        print(f"Depth: {circuit.depth()}")
        print(f"Gates: {circuit.count_ops()}")
        print("\nCircuit diagram:")
        print(circuit.draw(output='text'))
```

### Key Implementation Notes
1. **Always use copy=False** when appending gates that won't be reused
2. **Use inplace=True** for compose operations
3. **Prefer transpile with optimization_level=3** for production circuits
4. **Use meaningful register names** for debugging
5. **Leverage circuit.append() over individual gate methods** for bulk operations

## Output Log
*(This section is populated as work progresses on the task)*