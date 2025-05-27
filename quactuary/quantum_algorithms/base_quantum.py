"""
Base classes for quantum algorithms in quactuary.

This module provides abstract base classes and interfaces for implementing
quantum algorithms following Qiskit best practices and the four-step workflow.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.transpiler import generate_preset_pass_manager


class QuantumAlgorithm(ABC):
    """
    Abstract base class for quantum algorithms following Qiskit patterns.
    
    This class provides a standardized interface for implementing quantum algorithms
    in the quactuary framework. It enforces the four-step Qiskit workflow pattern,
    ensuring consistent implementation across all quantum algorithms.
    
    The four-step workflow ensures proper separation of concerns:
    1. **Problem Mapping**: Convert classical problems to quantum circuits
    2. **Circuit Optimization**: Optimize for target quantum hardware
    3. **Quantum Execution**: Run on simulators or real quantum devices
    4. **Result Analysis**: Extract meaningful classical results
    
    Attributes:
        backend (Backend, optional): Qiskit backend for circuit execution.
            Can be a simulator (AerSimulator) or real quantum hardware.
        _circuit (QuantumCircuit): Internal storage for the constructed circuit.
        _transpiled_circuit (QuantumCircuit): Optimized circuit after transpilation.
    
    Examples:
        Creating a custom quantum algorithm:
            >>> from quactuary.quantum_algorithms.base_quantum import QuantumAlgorithm
            >>> from qiskit import QuantumCircuit
            >>> from qiskit_aer import AerSimulator
            >>> 
            >>> class MyQuantumAlgorithm(QuantumAlgorithm):
            ...     def build_circuit(self, n_qubits=3):
            ...         qc = QuantumCircuit(n_qubits)
            ...         qc.h(range(n_qubits))  # Superposition
            ...         qc.measure_all()
            ...         return qc
            ...     
            ...     def optimize_circuit(self, circuit):
            ...         # Use default optimization level 1
            ...         from qiskit.transpiler import generate_preset_pass_manager
            ...         pm = generate_preset_pass_manager(1, self.backend)
            ...         return pm.run(circuit)
            ...     
            ...     def execute(self, circuit):
            ...         from qiskit.primitives import Sampler
            ...         sampler = Sampler()
            ...         job = sampler.run(circuit, shots=1000)
            ...         return {'counts': job.result().quasi_dists[0]}
            ...     
            ...     def analyze_results(self, results):
            ...         # Convert counts to probabilities
            ...         counts = results['counts']
            ...         total = sum(counts.values())
            ...         return {k: v/total for k, v in counts.items()}
            >>> 
            >>> # Use the algorithm
            >>> algo = MyQuantumAlgorithm(backend=AerSimulator())
            >>> circuit = algo.build_circuit(n_qubits=2)
            >>> opt_circuit = algo.optimize_circuit(circuit)
            >>> results = algo.execute(opt_circuit)
            >>> probabilities = algo.analyze_results(results)
            >>> print(probabilities)
            {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
    
        Using the run() convenience method:
            >>> # The base class provides a run() method that chains all steps
            >>> algo = MyQuantumAlgorithm()
            >>> final_results = algo.run(n_qubits=3)
            >>> print(f"Generated {len(final_results)} unique states")
            Generated 8 unique states
    
    Notes:
        - Subclasses must implement all four abstract methods
        - The backend parameter enables hardware-specific optimization
        - Use inplace=True and copy=False in circuit operations for efficiency
        - Consider caching transpiled circuits for repeated execution
    
    See Also:
        StatePreparationAlgorithm: For quantum state encoding algorithms
        VariationalQuantumAlgorithm: For VQE, QAOA, and similar algorithms
        qiskit.providers.Backend: Backend interface documentation
    """
    
    def __init__(self, backend: Optional[Backend] = None):
        """
        Initialize the quantum algorithm with optional backend.
        
        Args:
            backend: Qiskit backend for circuit execution. Can be:
                - None: Will use default simulator when execute() is called
                - AerSimulator(): High-performance local simulator
                - FakeBackend: Simulated quantum hardware with noise
                - IBMBackend: Real quantum hardware (requires credentials)
        
        Examples:
            Using default simulator:
                >>> algo = MyQuantumAlgorithm()  # Uses default backend
            
            Using specific simulator:
                >>> from qiskit_aer import AerSimulator
                >>> algo = MyQuantumAlgorithm(backend=AerSimulator(method='statevector'))
            
            Using fake hardware for testing:
                >>> from qiskit.providers.fake_provider import FakeMontreal
                >>> algo = MyQuantumAlgorithm(backend=FakeMontreal())
        """
        self.backend = backend
        self._circuit: Optional[QuantumCircuit] = None
        self._transpiled_circuit: Optional[QuantumCircuit] = None
        
    @abstractmethod
    def build_circuit(self, **params) -> QuantumCircuit:
        """
        Step 1: Map problem to quantum-native format.
        
        This method constructs the quantum circuit that encodes the problem. It should
        translate the classical problem specification into quantum gates and operations,
        handling state preparation, oracle construction, and measurement setup.
        
        Args:
            **params: Algorithm-specific parameters. Common parameters include:
                - n_qubits (int): Number of qubits to use
                - data (array-like): Classical data to encode
                - precision (float): Desired precision/accuracy
                - Any other algorithm-specific configuration
            
        Returns:
            QuantumCircuit: The constructed quantum circuit ready for optimization.
                Should include all necessary quantum registers, gates, and measurements.
        
        Examples:
            In a subclass implementation:
                >>> def build_circuit(self, data, n_qubits=5):
                ...     qc = QuantumCircuit(n_qubits, n_qubits)
                ...     # Encode data as amplitudes
                ...     qc.initialize(data / np.linalg.norm(data), range(n_qubits))
                ...     # Apply algorithm-specific operations
                ...     qc.h(0)
                ...     qc.measure_all()
                ...     return qc
        
        Notes:
            - Use meaningful register names for clarity
            - Consider using QuantumRegister and ClassicalRegister explicitly
            - Apply optimizations like inplace=True when composing circuits
        """
        pass
        
    @abstractmethod
    def optimize_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Step 2: Optimize circuit for target quantum backend.
        
        This method transpiles the circuit to match hardware constraints and optimize
        performance. It should handle basis gate translation, connectivity mapping,
        and circuit depth reduction based on the target backend's properties.
        
        Args:
            circuit: The quantum circuit to optimize. This is typically the output
                from build_circuit() but could be any QuantumCircuit instance.
            
        Returns:
            QuantumCircuit: Optimized circuit ready for execution. The circuit will:
                - Use only basis gates supported by the backend
                - Respect hardware connectivity constraints
                - Have reduced depth where possible
                - Include any necessary error mitigation
        
        Examples:
            Basic optimization with default settings:
                >>> def optimize_circuit(self, circuit):
                ...     from qiskit.transpiler import generate_preset_pass_manager
                ...     pm = generate_preset_pass_manager(
                ...         optimization_level=2,
                ...         backend=self.backend
                ...     )
                ...     return pm.run(circuit)
            
            Advanced optimization with custom settings:
                >>> def optimize_circuit(self, circuit):
                ...     from qiskit import transpile
                ...     return transpile(
                ...         circuit,
                ...         backend=self.backend,
                ...         optimization_level=3,
                ...         layout_method='sabre',
                ...         routing_method='sabre',
                ...         approximation_degree=0.99
                ...     )
        
        Notes:
            - Higher optimization levels (2-3) take more time but produce better circuits
            - Consider caching transpiled circuits for repeated execution
            - Some algorithms may benefit from custom transpiler passes
        """
        pass
        
    @abstractmethod
    def execute(self, circuit: QuantumCircuit) -> Dict[str, Any]:
        """
        Step 3: Execute circuit using quantum primitives.
        
        This method runs the optimized circuit on the quantum backend using appropriate
        Qiskit primitives. The choice of primitive depends on what the algorithm needs:
        Sampler for measurement outcomes, Estimator for expectation values.
        
        Args:
            circuit: Optimized quantum circuit ready for execution. Should be the
                output from optimize_circuit() for best performance.
            
        Returns:
            Dict[str, Any]: Execution results containing at minimum:
                - 'raw_counts' or 'expectation_values': Primary measurement data
                - 'metadata': Execution metadata (shots, backend, timestamp)
                - Algorithm-specific data (e.g., 'statevector', 'density_matrix')
        
        Raises:
            RuntimeError: If execution fails due to backend issues
            ValueError: If circuit is incompatible with the backend
        
        Examples:
            Using Sampler for measurement-based algorithms:
                >>> def execute(self, circuit):
                ...     from qiskit.primitives import Sampler
                ...     sampler = Sampler()
                ...     job = sampler.run(circuit, shots=10000)
                ...     result = job.result()
                ...     return {
                ...         'quasi_dists': result.quasi_dists[0],
                ...         'metadata': result.metadata[0]
                ...     }
            
            Using Estimator for expectation values:
                >>> def execute(self, circuit):
                ...     from qiskit.primitives import Estimator
                ...     from qiskit.quantum_info import SparsePauliOp
                ...     estimator = Estimator()
                ...     observable = SparsePauliOp(['ZZ', 'XI', 'IX'])
                ...     job = estimator.run(circuit, observable)
                ...     result = job.result()
                ...     return {
                ...         'expectation': result.values[0],
                ...         'std_error': result.metadata[0]['variance'] ** 0.5
                ...     }
        
        Notes:
            - Always use Qiskit primitives (Sampler/Estimator) not deprecated execute()
            - Consider error mitigation options for noisy backends
            - Set appropriate shot counts based on required precision
        """
        pass
        
    @abstractmethod
    def analyze_results(self, results: Dict[str, Any]) -> Any:
        """
        Step 4: Post-process quantum results into meaningful output.
        
        This method converts raw quantum measurement data into the final algorithm
        output. It handles statistical analysis, error estimation, and conversion
        to problem-specific formats.
        
        Args:
            results: Raw results dictionary from execute() containing measurement
                data, metadata, and any algorithm-specific information.
            
        Returns:
            Any: Processed results in algorithm-specific format. Common returns:
                - float: Single value (e.g., expectation value)
                - dict: Multiple values with error bars
                - np.ndarray: Vector/matrix results
                - Custom data class for complex results
        
        Examples:
            Extracting most likely outcome:
                >>> def analyze_results(self, results):
                ...     counts = results['quasi_dists']
                ...     # Find most probable state
                ...     best_state = max(counts, key=counts.get)
                ...     # Convert to problem solution
                ...     return self._decode_solution(best_state)
            
            Computing statistics with error bars:
                >>> def analyze_results(self, results):
                ...     expectation = results['expectation']
                ...     std_error = results['std_error']
                ...     return {
                ...         'value': expectation,
                ...         'confidence_interval': (
                ...             expectation - 1.96 * std_error,
                ...             expectation + 1.96 * std_error
                ...         ),
                ...         'relative_error': std_error / abs(expectation)
                ...     }
        
        Notes:
            - Include error estimates when possible
            - Handle edge cases (e.g., all-zero measurements)
            - Consider returning metadata for debugging/analysis
        """
        pass
    
    @property
    @abstractmethod
    def required_qubits(self) -> int:
        """
        Return the number of qubits required by this algorithm.
        
        This property helps with backend selection and resource planning.
        
        Returns:
            int: Number of qubits required
        """
        pass
    
    @abstractmethod
    def classical_equivalent(self, **params) -> Any:
        """
        Provide a classical implementation for comparison/fallback.
        
        This method should implement a classical version of the algorithm
        that produces mathematically equivalent results (within tolerance).
        
        Args:
            **params: Algorithm-specific parameters
            
        Returns:
            Results from classical computation
        """
        pass
    
    def run(self, **params) -> Any:
        """
        Execute the complete quantum algorithm workflow.
        
        This method orchestrates the four-step process and handles
        the complete execution flow.
        
        Args:
            **params: Algorithm-specific parameters
            
        Returns:
            Final processed results
        """
        # Step 1: Build circuit
        self._circuit = self.build_circuit(**params)
        
        # Step 2: Optimize circuit
        self._transpiled_circuit = self.optimize_circuit(self._circuit)
        
        # Step 3: Execute
        raw_results = self.execute(self._transpiled_circuit)
        
        # Step 4: Analyze results
        return self.analyze_results(raw_results)
    
    def get_circuit_info(self) -> Dict[str, Any]:
        """
        Get information about the current circuit.
        
        Returns:
            Dict containing circuit metrics like depth, gate count, etc.
        """
        if self._circuit is None:
            return {"error": "No circuit has been built yet"}
            
        info = {
            "num_qubits": self._circuit.num_qubits,
            "depth": self._circuit.depth(),
            "gate_count": len(self._circuit.data),
            "num_parameters": self._circuit.num_parameters,
        }
        
        if self._transpiled_circuit is not None:
            info["transpiled_depth"] = self._transpiled_circuit.depth()
            info["transpiled_gate_count"] = len(self._transpiled_circuit.data)
            
        return info


class StatePreparationAlgorithm(QuantumAlgorithm):
    """
    Abstract base class for quantum state preparation algorithms.
    
    This specialized class handles algorithms that encode classical data into quantum
    states, particularly probability distributions for actuarial applications. It extends
    QuantumAlgorithm with methods specific to state preparation and validation.
    
    State preparation is fundamental to quantum algorithms in insurance and finance,
    as it encodes loss distributions, portfolio data, and risk measures into quantum
    amplitudes that can be processed with quantum speedup.
    
    Attributes:
        target_distribution (np.ndarray): The probability distribution to encode
        prepared_state (np.ndarray): The actual quantum state prepared
        preparation_error (float): Error in state preparation
    
    Examples:
        Implementing a lognormal state preparation:
            >>> class LognormalStatePrep(StatePreparationAlgorithm):
            ...     def __init__(self, mu=0, sigma=1, n_qubits=8):
            ...         super().__init__()
            ...         self.mu = mu
            ...         self.sigma = sigma
            ...         self.n_qubits = n_qubits
            ...         
            ...     def build_circuit(self, **params):
            ...         # Generate lognormal probabilities
            ...         from scipy.stats import lognorm
            ...         x = np.linspace(0, 10, 2**self.n_qubits)
            ...         probs = lognorm.pdf(x, s=self.sigma, scale=np.exp(self.mu))
            ...         probs = probs / np.sum(probs)
            ...         
            ...         # Create circuit with amplitude encoding
            ...         from qiskit import QuantumCircuit
            ...         from qiskit.circuit.library import StatePreparation
            ...         qc = QuantumCircuit(self.n_qubits)
            ...         qc.append(StatePreparation(np.sqrt(probs)), range(self.n_qubits))
            ...         return qc
            ...         
            ...     def get_fidelity(self, target_state):
            ...         prepared = self.get_prepared_state()
            ...         return abs(np.vdot(prepared, target_state))**2
            >>> 
            >>> # Use the algorithm
            >>> prep = LognormalStatePrep(mu=1.0, sigma=0.5, n_qubits=6)
            >>> circuit = prep.build_circuit()
            >>> print(f"Circuit depth: {circuit.depth()}")
            Circuit depth: 1
    
    See Also:
        quactuary.quantum_algorithms.state_preparation: Concrete implementations
        qiskit.circuit.library.StatePreparation: Qiskit's state prep gate
    """
    
    @abstractmethod
    def get_fidelity(self, target_state: np.ndarray) -> float:
        """
        Calculate fidelity between prepared state and target state.
        
        Fidelity measures how close the prepared quantum state is to the desired
        target state. For pure states, F = |⟨ψ_target|ψ_prepared⟩|². A fidelity
        of 1.0 means perfect state preparation.
        
        Args:
            target_state: The ideal target state vector as a 1D numpy array of
                complex amplitudes. Must be normalized (sum of squared magnitudes = 1).
                Length must be 2^n where n is the number of qubits.
            
        Returns:
            float: Fidelity value between 0 and 1, where:
                - 1.0: Perfect state preparation
                - 0.99+: Excellent (suitable for most applications)  
                - 0.95+: Good (may need refinement for high precision)
                - <0.9: Poor (check implementation or increase resources)
        
        Raises:
            ValueError: If target_state dimensions don't match prepared state
            RuntimeError: If no state has been prepared yet
        
        Examples:
            >>> # In implementation
            >>> def get_fidelity(self, target_state):
            ...     if self.prepared_state is None:
            ...         raise RuntimeError("No state prepared yet")
            ...     # Calculate overlap
            ...     overlap = np.vdot(target_state, self.prepared_state)
            ...     return abs(overlap)**2
        
        Notes:
            - For mixed states, use state_fidelity from qiskit.quantum_info
            - Consider multiple fidelity calculations for statistical confidence
            - High fidelity may require more qubits or better encoding
        """
        pass
    
    @abstractmethod
    def get_prepared_state(self) -> np.ndarray:
        """
        Retrieve the quantum state prepared by this algorithm.
        
        Returns the state vector representation of the prepared quantum state.
        This is useful for verification, debugging, and comparison with theoretical
        expectations. For large systems, consider returning only relevant subsystems.
        
        Returns:
            np.ndarray: Complex state vector of length 2^n_qubits containing the
                quantum amplitudes. The state is normalized such that the sum of
                squared magnitudes equals 1.
        
        Raises:
            RuntimeError: If called before state preparation is complete
            MemoryError: If state vector is too large (>20 qubits typically)
        
        Examples:
            >>> # After running the algorithm
            >>> state = algo.get_prepared_state()
            >>> print(f"State dimension: {len(state)}")
            >>> print(f"First few amplitudes: {state[:4]}")
            >>> # Verify normalization
            >>> norm = np.sum(np.abs(state)**2)
            >>> print(f"State norm: {norm:.10f}")
            State dimension: 64
            First few amplitudes: [0.125+0j 0.125+0j 0.125+0j 0.125+0j]
            State norm: 1.0000000000
        
        Notes:
            - For >15 qubits, consider returning only probability distributions
            - Use statevector_simulator backend for exact state retrieval
            - Real hardware cannot directly access state vectors
        """
        pass


class VariationalQuantumAlgorithm(QuantumAlgorithm):
    """
    Abstract base class for variational quantum algorithms (VQE, QAOA, VQD).
    
    Variational algorithms combine parameterized quantum circuits with classical
    optimization to solve problems iteratively. They are particularly suited for
    NISQ devices as they can adapt to hardware limitations and noise.
    
    In actuarial applications, variational algorithms can optimize:
    - Portfolio allocations (QAOA for combinatorial optimization)
    - Risk measures (VQE for expectation values)
    - Model parameters (variational quantum regression)
    
    The algorithm alternates between:
    1. Quantum: Evaluate cost function with current parameters
    2. Classical: Update parameters to minimize cost
    
    Attributes:
        optimizer: Classical optimizer (SLSQP, COBYLA, SPSA, etc.)
        ansatz (QuantumCircuit): Parameterized quantum circuit template
        cost_history (List[float]): Cost function values during optimization
        _optimal_params (np.ndarray): Best parameters found
    
    Examples:
        Implementing a simple VQE for portfolio optimization:
            >>> from qiskit.circuit.library import TwoLocal
            >>> from qiskit_algorithms.optimizers import SPSA
            >>> 
            >>> class PortfolioVQE(VariationalQuantumAlgorithm):
            ...     def __init__(self, n_assets=4):
            ...         super().__init__(optimizer=SPSA(maxiter=100))
            ...         self.n_assets = n_assets
            ...         self.ansatz = TwoLocal(n_assets, 'ry', 'cz', reps=2)
            ...         
            ...     def cost_function(self, params):
            ...         # Build circuit with parameters
            ...         circuit = self.ansatz.bind_parameters(params)
            ...         # Execute and get expectation value
            ...         # (simplified - real implementation would use Estimator)
            ...         return np.random.random()  # Placeholder
            ...         
            ...     def get_initial_params(self):
            ...         # Random initialization
            ...         return np.random.uniform(-np.pi, np.pi, 
            ...                                 self.ansatz.num_parameters)
            >>> 
            >>> # Run optimization
            >>> vqe = PortfolioVQE(n_assets=3)
            >>> optimal_params = vqe.optimize_parameters()
            >>> print(f"Found {len(optimal_params)} optimal parameters")
            Found 12 optimal parameters
    
        Using pre-built optimizers:
            >>> from qiskit_algorithms.optimizers import COBYLA
            >>> 
            >>> # Noise-resilient optimizer for real hardware
            >>> algo = MyVQAlgorithm(
            ...     optimizer=COBYLA(maxiter=200, tol=0.001)
            ... )
            >>> 
            >>> # Gradient-based for simulators
            >>> from qiskit_algorithms.optimizers import L_BFGS_B
            >>> algo = MyVQAlgorithm(
            ...     optimizer=L_BFGS_B(maxfun=1000)
            ... )
    
    Notes:
        - Choice of optimizer impacts convergence and noise resilience
        - SPSA and COBYLA work well with noisy quantum hardware
        - Consider warm-starting with classically pre-optimized parameters
        - Monitor cost history to detect convergence issues
    
    See Also:
        qiskit_algorithms.VQE: Variational Quantum Eigensolver
        qiskit_algorithms.QAOA: Quantum Approximate Optimization Algorithm
        qiskit_algorithms.optimizers: Available classical optimizers
    """
    
    def __init__(self, backend: Optional[Backend] = None, 
                 optimizer: Optional[Any] = None):
        """
        Initialize variational quantum algorithm with optimizer.
        
        Args:
            backend: Quantum backend for circuit execution. If None, uses
                default simulator. Consider noise characteristics when choosing.
            optimizer: Classical optimizer instance from qiskit_algorithms.optimizers.
                Common choices:
                - COBYLA: Gradient-free, works well with noise
                - SPSA: Simultaneous perturbation, efficient for many parameters
                - SLSQP: Gradient-based, fast for noiseless simulation
                If None, must be set before calling optimize_parameters().
        
        Examples:
            >>> from qiskit_algorithms.optimizers import SPSA
            >>> algo = MyVQAlgorithm(
            ...     backend=AerSimulator(),
            ...     optimizer=SPSA(maxiter=150, blocking=True)
            ... )
        """
        super().__init__(backend)
        self.optimizer = optimizer
        self._optimal_params: Optional[np.ndarray] = None
        self.cost_history: List[float] = []
        
    @abstractmethod
    def cost_function(self, params: np.ndarray) -> float:
        """
        Evaluate the cost function for given parameters.
        
        Args:
            params: Circuit parameters
            
        Returns:
            float: Cost function value
        """
        pass
    
    @abstractmethod
    def get_initial_params(self) -> np.ndarray:
        """
        Generate initial parameters for the variational circuit.
        
        Returns:
            numpy array of initial parameter values
        """
        pass
    
    def optimize_parameters(self) -> np.ndarray:
        """
        Run classical optimization to find optimal parameters.
        
        Returns:
            numpy array of optimized parameters
        """
        if self.optimizer is None:
            raise ValueError("No optimizer specified")
            
        initial_params = self.get_initial_params()
        result = self.optimizer.minimize(self.cost_function, initial_params)
        self._optimal_params = result.x
        return self._optimal_params