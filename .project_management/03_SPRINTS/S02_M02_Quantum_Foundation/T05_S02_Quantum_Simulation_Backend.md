---
task_id: T05_S02
sprint_sequence_id: S02
status: open
complexity: Medium
last_updated: 2025-01-25T00:00:00Z
---

# Task: Quantum Simulation Backend Integration

## Description
Set up and configure quantum simulation backends, including both local simulators and potential cloud backend connections. Create an abstraction layer that allows easy switching between different quantum backends.

## Goal / Objectives
- Configure qiskit Aer simulators for local execution
- Create backend abstraction for future extensibility
- Implement backend selection logic
- Set up noise models for realistic simulations

## Acceptance Criteria
- [ ] Aer simulators configured and functional
- [ ] Backend abstraction layer implemented
- [ ] Backend selection based on circuit requirements
- [ ] Basic noise models available
- [ ] Performance profiling for different backends

## Subtasks
- [ ] Configure Aer statevector simulator
- [ ] Set up Aer QASM simulator
- [ ] Create QuantumBackend abstract class
- [ ] Implement backend factory pattern
- [ ] Add noise model configurations
- [ ] Create backend performance profiler
- [ ] Document backend usage and selection criteria

## Implementation Guidelines

### Backend Abstraction Layer
Based on Excess Loss algorithm and Qiskit patterns:

```python
# quactuary/quantum/backends/base_backend.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from qiskit import QuantumCircuit
from qiskit.providers import Backend, Job

class QuantumBackend(ABC):
    """Abstract base class for quantum backends."""
    
    def __init__(self, backend: Backend, shots: int = 1024):
        self.backend = backend
        self.shots = shots
        self._last_job: Optional[Job] = None
    
    @abstractmethod
    def execute(self, circuit: QuantumCircuit, **kwargs) -> Job:
        """Execute circuit on backend."""
        pass
    
    @abstractmethod
    def get_results(self, job: Job) -> Dict[str, Any]:
        """Extract results from job."""
        pass
    
    def validate_circuit(self, circuit: QuantumCircuit) -> bool:
        """Validate circuit compatibility with backend."""
        config = self.backend.configuration()
        return (circuit.num_qubits <= config.n_qubits and
                all(inst.operation.name in config.basis_gates 
                    for inst in circuit.data))
```

### Simulator Backend Implementation
```python
# quactuary/quantum/backends/simulator_backend.py
from qiskit_aer import AerSimulator
from qiskit import transpile
import numpy as np

class SimulatorBackend(QuantumBackend):
    """
    Aer simulator backend implementation.
    Following Excess Loss example patterns.
    """
    
    def __init__(self, method: str = 'statevector', shots: int = 1024):
        """
        Initialize simulator backend.
        
        Methods:
        - 'statevector': Exact statevector simulation
        - 'qasm': Shot-based simulation
        - 'density_matrix': Full density matrix
        """
        backend = AerSimulator(method=method)
        super().__init__(backend, shots)
        self.method = method
    
    def execute(self, circuit: QuantumCircuit, 
                optimization_level: int = 3, **kwargs) -> Job:
        """Execute with optimization."""
        # Transpile for optimization (as in Excess Loss example)
        transpiled = transpile(
            circuit, 
            self.backend,
            optimization_level=optimization_level
        )
        
        # Execute based on method
        if self.method == 'statevector':
            job = self.backend.run(transpiled, shots=1)
        else:
            job = self.backend.run(transpiled, shots=self.shots)
        
        self._last_job = job
        return job
    
    def get_results(self, job: Job = None) -> Dict[str, Any]:
        """Extract results based on simulation method."""
        if job is None:
            job = self._last_job
        
        result = job.result()
        
        if self.method == 'statevector':
            statevector = result.get_statevector()
            probabilities = np.abs(statevector) ** 2
            return {
                'statevector': statevector,
                'probabilities': probabilities,
                'counts': self._probs_to_counts(probabilities)
            }
        else:
            return {
                'counts': result.get_counts(),
                'memory': result.get_memory() if hasattr(result, 'get_memory') else None
            }
    
    def _probs_to_counts(self, probabilities: np.ndarray) -> Dict[str, int]:
        """Convert probabilities to simulated counts."""
        n_qubits = int(np.log2(len(probabilities)))
        counts = {}
        
        for i, prob in enumerate(probabilities):
            if prob > 1e-10:  # Threshold for numerical stability
                bitstring = format(i, f'0{n_qubits}b')
                counts[bitstring] = int(prob * self.shots)
        
        return counts
```

### Backend Factory
```python
# quactuary/quantum/backends/factory.py
from typing import Literal, Optional

class BackendFactory:
    """Factory for creating appropriate backends."""
    
    @staticmethod
    def create_backend(
        backend_type: Literal['statevector', 'qasm', 'density_matrix', 'gpu'],
        shots: int = 1024,
        noise_model: Optional[Any] = None
    ) -> QuantumBackend:
        """
        Create backend based on requirements.
        
        Args:
            backend_type: Type of simulation backend
            shots: Number of shots for sampling
            noise_model: Optional noise model
        """
        if backend_type == 'gpu':
            # GPU backend for large circuits
            try:
                from qiskit_aer import AerSimulator
                backend = AerSimulator(
                    method='statevector',
                    device='GPU'
                )
                return SimulatorBackend(backend, shots)
            except:
                print("GPU not available, falling back to CPU")
                backend_type = 'statevector'
        
        if backend_type in ['statevector', 'qasm', 'density_matrix']:
            sim_backend = SimulatorBackend(method=backend_type, shots=shots)
            
            if noise_model:
                sim_backend.backend.set_noise_model(noise_model)
            
            return sim_backend
        
        raise ValueError(f"Unknown backend type: {backend_type}")
```

### Noise Model Configuration
```python
# quactuary/quantum/backends/noise_models.py
from qiskit_aer.noise import NoiseModel, QuantumError, depolarizing_error
from qiskit_aer.noise import amplitude_damping_error, phase_damping_error

def create_realistic_noise_model(
    single_qubit_error: float = 0.001,
    two_qubit_error: float = 0.01,
    readout_error: float = 0.01
) -> NoiseModel:
    """
    Create realistic noise model for simulations.
    Based on typical NISQ device characteristics.
    """
    noise_model = NoiseModel()
    
    # Single-qubit gate errors
    single_qubit_gates = ['u1', 'u2', 'u3', 'rx', 'ry', 'rz']
    for gate in single_qubit_gates:
        error = depolarizing_error(single_qubit_error, 1)
        noise_model.add_quantum_error(error, gate, [0])
    
    # Two-qubit gate errors
    two_qubit_gates = ['cx', 'cz']
    for gate in two_qubit_gates:
        error = depolarizing_error(two_qubit_error, 2)
        noise_model.add_all_qubit_quantum_error(error, gate)
    
    # Measurement errors
    prob_meas0_prep1 = readout_error
    prob_meas1_prep0 = readout_error
    noise_model.add_readout_error(
        [[1 - prob_meas1_prep0, prob_meas1_prep0],
         [prob_meas0_prep1, 1 - prob_meas0_prep1]],
        [0]
    )
    
    return noise_model

def create_amplitude_damping_model(t1: float = 50e-6, 
                                   gate_time: float = 0.1e-6) -> NoiseModel:
    """Create T1 relaxation noise model."""
    noise_model = NoiseModel()
    
    # Calculate error probability
    prob = 1 - np.exp(-gate_time / t1)
    error = amplitude_damping_error(prob)
    
    # Add to all single-qubit gates
    noise_model.add_all_qubit_quantum_error(error, ['id', 'u1', 'u2', 'u3'])
    
    return noise_model
```

### Performance Profiler
```python
# quactuary/quantum/backends/profiler.py
import time
from typing import List, Dict
import numpy as np

class BackendProfiler:
    """Profile backend performance for circuit execution."""
    
    @staticmethod
    def profile_backends(circuit: QuantumCircuit, 
                        backends: List[QuantumBackend],
                        num_runs: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Profile execution time and accuracy across backends.
        """
        results = {}
        
        for backend in backends:
            backend_name = backend.__class__.__name__
            times = []
            
            for _ in range(num_runs):
                start = time.time()
                job = backend.execute(circuit)
                result = backend.get_results(job)
                end = time.time()
                
                times.append(end - start)
            
            results[backend_name] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times)
            }
        
        return results
    
    @staticmethod
    def recommend_backend(circuit: QuantumCircuit) -> str:
        """Recommend backend based on circuit properties."""
        num_qubits = circuit.num_qubits
        depth = circuit.depth()
        
        if num_qubits <= 20 and depth < 100:
            return 'statevector'  # Exact simulation feasible
        elif num_qubits <= 30:
            return 'qasm'  # Shot-based simulation
        else:
            return 'gpu'  # Need GPU acceleration
```

### Usage Example
```python
# Example matching Excess Loss pattern
from quactuary.quantum.backends import BackendFactory
from quactuary.quantum.state_preparation import amplitude_encode

# Create probability distribution
probabilities = prepare_lognormal_state(mu=1.0, sigma=0.5, num_qubits=6)

# Encode in quantum circuit
circuit = amplitude_encode(probabilities)

# Create backend (matching Excess Loss example)
backend = BackendFactory.create_backend('statevector')

# Execute
job = backend.execute(circuit, optimization_level=3)
results = backend.get_results(job)

# Extract quantum probabilities
quantum_probs = results['probabilities']
```

## Output Log
*(This section is populated as work progresses on the task)*