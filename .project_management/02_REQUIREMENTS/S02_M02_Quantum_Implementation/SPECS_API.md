# API Specifications
## S02_M02_Quantum_Implementation

### Overview
This document defines the API specifications for quantum algorithms in the quactuary package. The APIs are designed to maintain backward compatibility while seamlessly integrating quantum capabilities into the existing framework.

### Core Quantum APIs

#### 1. Quantum Backend Extension
```python
# Extend existing Backend enum
class Backend(Enum):
    CLASSICAL = "classical"
    QUANTUM = "quantum"
    AUTO = "auto"  # Intelligent selection

# Quantum backend implementation
class QuantumBackend:
    """Quantum computation backend for actuarial calculations."""
    
    def __init__(self, simulator_type: str = "statevector", 
                 optimization_level: int = 3):
        """
        Initialize quantum backend.
        
        Args:
            simulator_type: Type of quantum simulator 
                          ("statevector", "qasm", "density_matrix")
            optimization_level: Circuit optimization level (0-3)
        """
        
    def execute(self, algorithm: QuantumAlgorithm, 
                parameters: Dict[str, Any]) -> Result:
        """Execute quantum algorithm with given parameters."""
        
    def estimate_resources(self, algorithm: QuantumAlgorithm,
                          parameters: Dict[str, Any]) -> ResourceEstimate:
        """Estimate quantum resources required."""
```

#### 2. Quantum Algorithm Base Class
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from qiskit import QuantumCircuit

class QuantumAlgorithm(ABC):
    """Base class for all quantum algorithms."""
    
    @abstractmethod
    def build_circuit(self, **params) -> QuantumCircuit:
        """Build quantum circuit for the algorithm."""
        pass
        
    @abstractmethod
    def post_process(self, counts: Dict[str, int]) -> Any:
        """Post-process quantum measurement results."""
        pass
        
    @property
    @abstractmethod
    def required_qubits(self) -> int:
        """Number of qubits required."""
        pass
        
    @property
    @abstractmethod
    def classical_equivalent(self) -> str:
        """Name of equivalent classical algorithm."""
        pass
```

#### 3. Quantum State Preparation APIs
```python
class QuantumStatePreparation:
    """APIs for quantum state preparation."""
    
    @staticmethod
    def amplitude_encode(distribution: np.ndarray, 
                        num_qubits: Optional[int] = None) -> QuantumCircuit:
        """
        Encode probability distribution as quantum amplitudes.
        
        Args:
            distribution: Probability distribution to encode
            num_qubits: Number of qubits (auto-determined if None)
            
        Returns:
            QuantumCircuit with encoded distribution
        """
        
    @staticmethod
    def prepare_lognormal(mu: float, sigma: float, 
                         num_qubits: int = 6) -> QuantumCircuit:
        """Prepare lognormal distribution in quantum state."""
        
    @staticmethod
    def prepare_poisson(lambda_param: float, 
                       num_qubits: int = 6) -> QuantumCircuit:
        """Prepare Poisson distribution in quantum state."""
```

#### 4. Quantum Excess Loss Algorithm
```python
class QuantumExcessLoss(QuantumAlgorithm):
    """Quantum algorithm for excess loss calculation."""
    
    def __init__(self, threshold: float, distribution_params: Dict[str, float]):
        """
        Initialize quantum excess loss algorithm.
        
        Args:
            threshold: Excess threshold value
            distribution_params: Parameters for loss distribution
        """
        
    def calculate_excess_loss(self) -> float:
        """Calculate expected excess loss using quantum algorithm."""
        
    def calculate_excess_probability(self) -> float:
        """Calculate probability of exceeding threshold."""
```

#### 5. Quantum Amplitude Estimation APIs
```python
class QuantumAmplitudeEstimation:
    """Quantum amplitude estimation for risk measures."""
    
    def __init__(self, oracle: QuantumCircuit, num_evaluation_qubits: int = 5):
        """
        Initialize amplitude estimation.
        
        Args:
            oracle: Quantum oracle marking target states
            num_evaluation_qubits: Precision of estimation
        """
        
    def estimate(self, confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Estimate amplitude with confidence interval.
        
        Returns:
            Tuple of (estimate, error_bound)
        """
```

### Classical-Quantum Bridge APIs

#### 1. Unified Pricing Model Interface
```python
class PricingModel:
    """Extended pricing model with quantum support."""
    
    def __init__(self, backend: Backend = Backend.AUTO):
        """Initialize with specified backend."""
        self.backend = backend
        self._quantum_backend = None
        self._classical_backend = None
        
    def calculate_premium(self, policy: Policy, 
                         portfolio: Portfolio) -> float:
        """
        Calculate premium using optimal backend.
        
        Automatically selects quantum or classical based on:
        - Problem size
        - Available resources
        - Expected performance gain
        """
        
    def calculate_var(self, portfolio: Portfolio, 
                     confidence: float = 0.95) -> float:
        """Calculate VaR using optimal algorithm."""
        
    def calculate_tvar(self, portfolio: Portfolio,
                      confidence: float = 0.95) -> float:
        """Calculate TVaR using optimal algorithm."""
```

#### 2. Backend Selection API
```python
class BackendSelector:
    """Intelligent backend selection."""
    
    @staticmethod
    def select_backend(algorithm: str, problem_size: Dict[str, int],
                      available_resources: Dict[str, Any]) -> Backend:
        """
        Select optimal backend for given problem.
        
        Args:
            algorithm: Name of algorithm to execute
            problem_size: Problem dimensions (e.g., num_samples, dimension)
            available_resources: Available computational resources
            
        Returns:
            Recommended backend
        """
        
    @staticmethod
    def estimate_speedup(algorithm: str, problem_size: Dict[str, int]) -> float:
        """Estimate quantum speedup factor."""
```

#### 3. Distribution Bridge API
```python
class Distribution:
    """Extended distribution class with quantum support."""
    
    def to_quantum_state(self, num_qubits: int = None) -> QuantumCircuit:
        """Convert distribution to quantum state."""
        
    def sample_quantum(self, num_samples: int, 
                      backend: QuantumBackend) -> np.ndarray:
        """Sample from distribution using quantum algorithm."""
```

### Algorithm Selection APIs

#### 1. Algorithm Registry
```python
class QuantumAlgorithmRegistry:
    """Registry of available quantum algorithms."""
    
    @classmethod
    def register(cls, name: str, algorithm_class: Type[QuantumAlgorithm],
                 metadata: Dict[str, Any]):
        """Register new quantum algorithm."""
        
    @classmethod
    def get_algorithm(cls, name: str) -> Type[QuantumAlgorithm]:
        """Get quantum algorithm by name."""
        
    @classmethod
    def list_algorithms(cls) -> List[str]:
        """List all available quantum algorithms."""
        
    @classmethod
    def get_metadata(cls, name: str) -> Dict[str, Any]:
        """Get algorithm metadata (speedup, requirements, etc.)."""
```

#### 2. Decision Engine API
```python
class QuantumDecisionEngine:
    """Decision engine for algorithm selection."""
    
    def __init__(self, preferences: Dict[str, Any] = None):
        """
        Initialize with user preferences.
        
        Args:
            preferences: User preferences (e.g., min_speedup_factor)
        """
        
    def should_use_quantum(self, algorithm: str, 
                          problem_params: Dict[str, Any]) -> bool:
        """Decide whether to use quantum algorithm."""
        
    def get_recommendation(self, task: str, 
                          problem_params: Dict[str, Any]) -> AlgorithmRecommendation:
        """Get detailed recommendation with reasoning."""
```

### Error Handling

#### 1. Quantum-Specific Exceptions
```python
class QuantumError(Exception):
    """Base class for quantum-related errors."""
    pass

class QuantumResourceError(QuantumError):
    """Insufficient quantum resources."""
    pass

class QuantumCircuitError(QuantumError):
    """Error in quantum circuit construction."""
    pass

class QuantumSimulationError(QuantumError):
    """Error during quantum simulation."""
    pass

class QuantumConvergenceError(QuantumError):
    """Quantum algorithm failed to converge."""
    pass
```

#### 2. Error Recovery API
```python
class QuantumErrorHandler:
    """Handle quantum-specific errors gracefully."""
    
    @staticmethod
    def with_classical_fallback(quantum_func: Callable, 
                               classical_func: Callable,
                               *args, **kwargs) -> Any:
        """Execute quantum function with classical fallback."""
        
    @staticmethod
    def validate_circuit(circuit: QuantumCircuit) -> List[str]:
        """Validate quantum circuit and return warnings."""
```

### Performance Considerations

#### 1. Performance Profiling API
```python
class QuantumPerformanceProfiler:
    """Profile quantum algorithm performance."""
    
    def profile_circuit(self, circuit: QuantumCircuit) -> CircuitMetrics:
        """
        Profile quantum circuit.
        
        Returns metrics:
        - Circuit depth
        - Gate count by type
        - Estimated execution time
        - Memory requirements
        """
        
    def compare_backends(self, algorithm: QuantumAlgorithm,
                        params: Dict[str, Any]) -> ComparisonReport:
        """Compare quantum vs classical performance."""
        
    def benchmark_suite(self, problem_sizes: List[int]) -> BenchmarkResults:
        """Run comprehensive benchmarks across problem sizes."""
```

#### 2. Resource Estimation API
```python
class QuantumResourceEstimator:
    """Estimate quantum resource requirements."""
    
    @staticmethod
    def estimate_qubits(algorithm: str, problem_size: int) -> int:
        """Estimate number of qubits required."""
        
    @staticmethod
    def estimate_circuit_depth(algorithm: str, problem_size: int) -> int:
        """Estimate circuit depth."""
        
    @staticmethod
    def estimate_execution_time(circuit: QuantumCircuit, 
                               backend: str) -> float:
        """Estimate execution time in seconds."""
```

#### 3. Optimization API
```python
class QuantumOptimizer:
    """Optimize quantum circuits for performance."""
    
    @staticmethod
    def optimize_for_depth(circuit: QuantumCircuit) -> QuantumCircuit:
        """Optimize circuit to minimize depth."""
        
    @staticmethod
    def optimize_for_gates(circuit: QuantumCircuit, 
                          target_gates: List[str]) -> QuantumCircuit:
        """Optimize circuit for specific gate set."""
        
    @staticmethod
    def auto_optimize(circuit: QuantumCircuit, 
                     backend: QuantumBackend) -> QuantumCircuit:
        """Automatically optimize for target backend."""
```

### Usage Examples

#### Example 1: Calculate Excess Loss with Auto Backend
```python
# Existing API works transparently
model = PricingModel(backend=Backend.AUTO)
portfolio = Portfolio([policy1, policy2, policy3])

# Automatically uses quantum if beneficial
excess_loss = model.calculate_excess_loss(
    portfolio, 
    threshold=1000000,
    confidence=0.95
)
```

#### Example 2: Force Quantum Backend
```python
# Explicitly use quantum
model = PricingModel(backend=Backend.QUANTUM)

# Will use quantum algorithm
var = model.calculate_var(portfolio, confidence=0.99)
```

#### Example 3: Custom Quantum Algorithm
```python
# Implement custom quantum algorithm
class MyQuantumAlgorithm(QuantumAlgorithm):
    def build_circuit(self, **params):
        # Custom circuit construction
        pass
        
    def post_process(self, counts):
        # Custom post-processing
        pass

# Register algorithm
QuantumAlgorithmRegistry.register(
    "my_algorithm",
    MyQuantumAlgorithm,
    {"speedup": "quadratic", "min_qubits": 10}
)
```

### API Versioning and Deprecation

- All quantum APIs will be marked as `@experimental` in v1.0
- Stable API guarantee after v2.0
- Deprecation warnings for 2 minor versions before removal
- Backward compatibility maintained for all existing APIs