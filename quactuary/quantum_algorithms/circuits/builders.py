"""
Circuit builder patterns for quantum algorithms.

This module provides builder classes and utilities for constructing
quantum circuits following best practices.
"""

from typing import List, Optional, Union, Callable
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
import numpy as np


class CircuitBuilder:
    """
    Builder pattern for constructing quantum circuits.
    
    This class provides a fluent interface for building quantum circuits
    with common patterns and best practices.
    """
    
    def __init__(self, num_qubits: int, num_classical: Optional[int] = None,
                 name: str = "circuit"):
        """
        Initialize circuit builder.
        
        Args:
            num_qubits: Number of quantum bits
            num_classical: Number of classical bits (default: same as qubits)
            name: Circuit name
        """
        self.qreg = QuantumRegister(num_qubits, 'q')
        self.creg = ClassicalRegister(
            num_classical or num_qubits, 'c'
        )
        self.circuit = QuantumCircuit(self.qreg, self.creg, name=name)
        
    def add_hadamard_layer(self, qubits: Optional[List[int]] = None) -> "CircuitBuilder":
        """
        Add Hadamard gates to specified qubits.
        
        Args:
            qubits: List of qubit indices (default: all qubits)
            
        Returns:
            self for method chaining
        """
        if qubits is None:
            qubits = list(range(self.circuit.num_qubits))
            
        for q in qubits:
            self.circuit.h(q)
            
        return self
    
    def add_rotation_layer(self, axis: str, params: Union[List[float], ParameterVector],
                          qubits: Optional[List[int]] = None) -> "CircuitBuilder":
        """
        Add rotation gates around specified axis.
        
        Args:
            axis: Rotation axis ('x', 'y', or 'z')
            params: Rotation parameters
            qubits: List of qubit indices (default: all qubits)
            
        Returns:
            self for method chaining
        """
        if qubits is None:
            qubits = list(range(self.circuit.num_qubits))
            
        if len(params) != len(qubits):
            raise ValueError(f"Number of parameters ({len(params)}) must match "
                           f"number of qubits ({len(qubits)})")
            
        rotation_gate = getattr(self.circuit, f'r{axis}')
        
        for q, param in zip(qubits, params):
            rotation_gate(param, q)
            
        return self
    
    def add_entangling_layer(self, pattern: str = "linear") -> "CircuitBuilder":
        """
        Add entangling gates following a specified pattern.
        
        Args:
            pattern: Entanglement pattern ('linear', 'circular', 'all_to_all')
            
        Returns:
            self for method chaining
        """
        n = self.circuit.num_qubits
        
        if pattern == "linear":
            for i in range(n - 1):
                self.circuit.cx(i, i + 1)
                
        elif pattern == "circular":
            for i in range(n - 1):
                self.circuit.cx(i, i + 1)
            if n > 2:
                self.circuit.cx(n - 1, 0)
                
        elif pattern == "all_to_all":
            for i in range(n):
                for j in range(i + 1, n):
                    self.circuit.cx(i, j)
                    
        else:
            raise ValueError(f"Unknown entanglement pattern: {pattern}")
            
        return self
    
    def add_barrier(self, qubits: Optional[List[int]] = None) -> "CircuitBuilder":
        """
        Add a barrier to specified qubits.
        
        Args:
            qubits: List of qubit indices (default: all qubits)
            
        Returns:
            self for method chaining
        """
        if qubits is None:
            self.circuit.barrier()
        else:
            self.circuit.barrier(qubits)
            
        return self
    
    def add_measurement(self, qubits: Optional[List[int]] = None,
                       clbits: Optional[List[int]] = None) -> "CircuitBuilder":
        """
        Add measurements to specified qubits.
        
        Args:
            qubits: List of qubit indices to measure (default: all)
            clbits: List of classical bit indices (default: same as qubits)
            
        Returns:
            self for method chaining
        """
        if qubits is None:
            qubits = list(range(self.circuit.num_qubits))
            
        if clbits is None:
            clbits = qubits[:self.circuit.num_clbits]
            
        self.circuit.measure(qubits, clbits)
        
        return self
    
    def add_custom_gate(self, gate_func: Callable, *args, **kwargs) -> "CircuitBuilder":
        """
        Add a custom gate or operation.
        
        Args:
            gate_func: Function that takes circuit as first argument
            *args: Additional arguments for gate_func
            **kwargs: Keyword arguments for gate_func
            
        Returns:
            self for method chaining
        """
        gate_func(self.circuit, *args, **kwargs)
        return self
    
    def build(self) -> QuantumCircuit:
        """
        Return the constructed circuit.
        
        Returns:
            The quantum circuit
        """
        return self.circuit
    

class ParameterizedCircuitBuilder(CircuitBuilder):
    """
    Specialized builder for parameterized quantum circuits.
    
    This builder is designed for variational algorithms that require
    parameterized circuits.
    """
    
    def __init__(self, num_qubits: int, num_params: int,
                 param_prefix: str = "θ", **kwargs):
        """
        Initialize parameterized circuit builder.
        
        Args:
            num_qubits: Number of qubits
            num_params: Number of parameters
            param_prefix: Prefix for parameter names
            **kwargs: Additional arguments for CircuitBuilder
        """
        super().__init__(num_qubits, **kwargs)
        self.params = ParameterVector(param_prefix, num_params)
        self.param_index = 0
        
    def get_next_param(self) -> Parameter:
        """
        Get the next available parameter.
        
        Returns:
            Parameter object
            
        Raises:
            IndexError: If all parameters have been used
        """
        if self.param_index >= len(self.params):
            raise IndexError("All parameters have been used")
            
        param = self.params[self.param_index]
        self.param_index += 1
        return param
    
    def add_parameterized_rotation_layer(self, axis: str,
                                        qubits: Optional[List[int]] = None) -> "ParameterizedCircuitBuilder":
        """
        Add parameterized rotation layer.
        
        Args:
            axis: Rotation axis ('x', 'y', or 'z')
            qubits: List of qubit indices (default: all qubits)
            
        Returns:
            self for method chaining
        """
        if qubits is None:
            qubits = list(range(self.circuit.num_qubits))
            
        params = [self.get_next_param() for _ in qubits]
        return super().add_rotation_layer(axis, params, qubits)
    
    def add_variational_layer(self, rotation_blocks: str = 'ry',
                             entanglement: str = 'linear') -> "ParameterizedCircuitBuilder":
        """
        Add a variational layer with rotations and entanglement.
        
        Args:
            rotation_blocks: Type of rotation gates ('rx', 'ry', 'rz')
            entanglement: Entanglement pattern
            
        Returns:
            self for method chaining
        """
        # Add rotation layer
        self.add_parameterized_rotation_layer(rotation_blocks[-1])
        
        # Add entangling layer
        self.add_entangling_layer(entanglement)
        
        return self