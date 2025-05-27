"""
Optimization utilities for quantum algorithms.

This module provides functions for circuit optimization, parameter
optimization, and performance tuning.
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.transpiler import PassManager, InstructionDurations
from qiskit.transpiler.passes import (
    Optimize1qGates,
    CommutativeCancellation,
    CXCancellation,
    OptimizeSwapBeforeMeasure,
    RemoveDiagonalGatesBeforeMeasure,
    Depth,
    Size,
    FixedPoint,
    Collect2qBlocks,
    ConsolidateBlocks
)

from quactuary.quantum.quantum_types import (
    CircuitMetrics,
    OptimizationLevel,
    DEFAULT_OPTIMIZATION_LEVEL
)


def optimize_circuit_depth(circuit: QuantumCircuit,
                         optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL) -> QuantumCircuit:
    """
    Optimize circuit to reduce depth.
    
    Args:
        circuit: Circuit to optimize
        optimization_level: Optimization level (0-3)
        
    Returns:
        Optimized circuit
    """
    if optimization_level == 0:
        return circuit
        
    # Build custom pass manager for depth optimization
    passes = []
    
    # Level 1: Basic optimizations
    passes.extend([
        Optimize1qGates(),
        CXCancellation(),
    ])
    
    if optimization_level >= 2:
        # Level 2: More aggressive optimizations
        passes.extend([
            CommutativeCancellation(),
            OptimizeSwapBeforeMeasure(),
            RemoveDiagonalGatesBeforeMeasure(),
        ])
        
    if optimization_level >= 3:
        # Level 3: Block collection and consolidation
        passes.extend([
            Collect2qBlocks(),
            ConsolidateBlocks(),
        ])
        
    # Add convergence check
    passes.append(FixedPoint('depth'))
    
    pm = PassManager(passes)
    optimized = pm.run(circuit)
    
    return optimized


def reduce_circuit_width(circuit: QuantumCircuit,
                        target_qubits: Optional[int] = None) -> QuantumCircuit:
    """
    Attempt to reduce circuit width (number of qubits).
    
    Args:
        circuit: Circuit to optimize
        target_qubits: Target number of qubits
        
    Returns:
        Optimized circuit (may have same width if reduction not possible)
    """
    # Analyze qubit usage
    used_qubits = set()
    for inst in circuit.data:
        used_qubits.update(inst.qubits)
        
    num_used = len(used_qubits)
    
    if num_used < circuit.num_qubits:
        # Some qubits are unused, create smaller circuit
        new_circuit = QuantumCircuit(num_used)
        
        # Map old qubits to new qubits
        qubit_map = {old: new for old, new in 
                    zip(sorted(used_qubits), range(num_used))}
        
        # Copy gates with remapped qubits
        for inst in circuit.data:
            new_qubits = [qubit_map[q] for q in inst.qubits]
            new_circuit.append(inst.operation, new_qubits)
            
        return new_circuit
        
    return circuit


def optimize_parameterized_circuit(circuit: QuantumCircuit,
                                 param_values: np.ndarray,
                                 merge_threshold: float = 1e-10) -> Tuple[QuantumCircuit, np.ndarray]:
    """
    Optimize parameterized circuit by merging similar parameters.
    
    Args:
        circuit: Parameterized circuit
        param_values: Parameter values
        merge_threshold: Threshold for merging similar parameters
        
    Returns:
        Tuple of (optimized_circuit, reduced_parameters)
    """
    params = circuit.parameters
    n_params = len(params)
    
    if n_params != len(param_values):
        raise ValueError("Number of parameters doesn't match values")
        
    # Find similar parameters
    param_groups = []
    used = set()
    
    for i in range(n_params):
        if i in used:
            continue
            
        group = [i]
        for j in range(i + 1, n_params):
            if j not in used:
                if abs(param_values[i] - param_values[j]) < merge_threshold:
                    group.append(j)
                    used.add(j)
                    
        param_groups.append(group)
        
    # If no merging possible, return original
    if len(param_groups) == n_params:
        return circuit, param_values
        
    # Create new circuit with merged parameters
    new_params = ParameterVector('θ', len(param_groups))
    new_values = []
    
    # Map old parameters to new
    param_map = {}
    for i, group in enumerate(param_groups):
        # Use average value for group
        avg_value = np.mean([param_values[j] for j in group])
        new_values.append(avg_value)
        
        for j in group:
            param_map[params[j]] = new_params[i]
            
    # Rebuild circuit with new parameters
    new_circuit = circuit.assign_parameters(param_map)
    
    return new_circuit, np.array(new_values)


def estimate_optimization_speedup(original: QuantumCircuit,
                                optimized: QuantumCircuit) -> Dict[str, float]:
    """
    Estimate speedup from circuit optimization.
    
    Args:
        original: Original circuit
        optimized: Optimized circuit
        
    Returns:
        Dictionary with speedup metrics
    """
    orig_metrics = CircuitMetrics.from_circuit(original)
    opt_metrics = CircuitMetrics.from_circuit(optimized)
    
    return {
        "depth_reduction": 1 - (opt_metrics.depth / orig_metrics.depth),
        "gate_reduction": 1 - (opt_metrics.gate_count / orig_metrics.gate_count),
        "cnot_reduction": 1 - (opt_metrics.cnot_count / max(orig_metrics.cnot_count, 1)),
        "estimated_speedup": orig_metrics.depth / opt_metrics.depth,
    }


def find_optimal_decomposition(unitary: np.ndarray,
                             basis_gates: List[str],
                             max_error: float = 1e-10) -> QuantumCircuit:
    """
    Find optimal decomposition of a unitary into basis gates.
    
    Args:
        unitary: Unitary matrix to decompose
        basis_gates: Available basis gates
        max_error: Maximum allowed error
        
    Returns:
        Optimal circuit decomposition
    """
    from qiskit.quantum_info import Operator
    from qiskit.compiler import transpile
    
    # Create circuit from unitary
    n_qubits = int(np.log2(unitary.shape[0]))
    circuit = QuantumCircuit(n_qubits)
    circuit.unitary(unitary, range(n_qubits))
    
    # Try different optimization levels
    best_circuit = None
    best_depth = float('inf')
    
    for opt_level in range(4):
        decomposed = transpile(
            circuit,
            basis_gates=basis_gates,
            optimization_level=opt_level
        )
        
        if decomposed.depth() < best_depth:
            # Verify accuracy
            decomposed_op = Operator(decomposed)
            error = np.linalg.norm(decomposed_op.data - unitary)
            
            if error < max_error:
                best_circuit = decomposed
                best_depth = decomposed.depth()
                
    return best_circuit


def optimize_measurement_order(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Optimize measurement order to reduce circuit depth.
    
    Args:
        circuit: Circuit with measurements
        
    Returns:
        Circuit with optimized measurement order
    """
    # Separate circuit into pre-measurement and measurement parts
    pre_measure = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
    measurements = []
    
    for inst in circuit.data:
        if inst.operation.name == 'measure':
            measurements.append(inst)
        else:
            pre_measure.append(inst)
            
    if not measurements:
        return circuit
        
    # Analyze which qubits can be measured early
    # (qubits that are not used after certain point)
    qubit_last_use = {}
    for i, inst in enumerate(pre_measure.data):
        for qubit in inst.qubits:
            qubit_last_use[qubit] = i
            
    # Sort measurements by last use of qubit
    measurements.sort(key=lambda m: qubit_last_use.get(m.qubits[0], -1))
    
    # Rebuild circuit with optimized measurement order
    optimized = pre_measure.copy()
    for meas in measurements:
        optimized.append(meas)
        
    return optimized


class ParameterOptimizer:
    """
    Optimizer for variational circuit parameters.
    
    This class provides various optimization strategies for finding
    optimal parameters in variational quantum algorithms.
    """
    
    def __init__(self, method: str = "COBYLA", max_iter: int = 1000):
        """
        Initialize parameter optimizer.
        
        Args:
            method: Optimization method
            max_iter: Maximum iterations
        """
        self.method = method
        self.max_iter = max_iter
        self.history: List[Dict[str, Any]] = []
        
    def optimize(self, cost_function: Callable[[np.ndarray], float],
                initial_params: np.ndarray,
                bounds: Optional[List[Tuple[float, float]]] = None) -> Dict[str, Any]:
        """
        Optimize parameters to minimize cost function.
        
        Args:
            cost_function: Function to minimize
            initial_params: Starting parameters
            bounds: Parameter bounds
            
        Returns:
            Optimization result
        """
        from scipy.optimize import minimize
        
        # Track optimization history
        self.history = []
        
        def wrapped_cost(params):
            cost = cost_function(params)
            self.history.append({
                'params': params.copy(),
                'cost': cost,
                'iteration': len(self.history)
            })
            return cost
            
        # Run optimization
        result = minimize(
            wrapped_cost,
            initial_params,
            method=self.method,
            bounds=bounds,
            options={'maxiter': self.max_iter}
        )
        
        return {
            'optimal_params': result.x,
            'optimal_value': result.fun,
            'converged': result.success,
            'iterations': result.nit,
            'history': self.history
        }
        
    def get_convergence_plot_data(self) -> Tuple[List[int], List[float]]:
        """
        Get data for convergence plotting.
        
        Returns:
            Tuple of (iterations, costs)
        """
        if not self.history:
            return [], []
            
        iterations = [h['iteration'] for h in self.history]
        costs = [h['cost'] for h in self.history]
        
        return iterations, costs