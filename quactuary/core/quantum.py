"""
Quantum actuarial integration module.

Centralizes quantum computing support for actuarial models, including backend management,
circuit construction, and execution.

Examples:
    >>> from quactuary.core.quantum import get_backend, QuantumActuarialModel
    >>> backend = get_backend()
    >>> model = QuantumActuarialModel(inforce_obj)
    >>> result = model.run()
"""

from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit_ibm_provider import IBMProvider

from quactuary.entities import Inforce, Portfolio

# Global backend manager instance
_backend = None


def get_backend():
    """
    Retrieve or initialize the global backend manager.

    Returns:
        BackendManager: Singleton backend manager for quantum execution.
    """
    global _backend
    if _backend is None:
        # Default to Aer simulator
        _backend = BackendManager(Aer.get_backend('aer_simulator'))
    return _backend


def set_backend(backend_type, provider=None, **kwargs):
    """
    Configure the global backend for simulation.

    Args:
        backend_type (str): Type of backend ('quantum' or 'classical').
        provider (Optional[str]): Specific quantum provider ('AerSimulator', 'IBMQ').
        **kwargs: Additional provider-specific settings.

    Returns:
        BackendManager: Updated backend manager instance.

    Raises:
        ValueError: If backend_type or provider is unsupported.

    Examples:
        >>> set_backend('quantum', provider='AerSimulator')
        >>> set_backend('quantum', provider='IBMQ', hub='ibm-q', token='abc')
    """
    backend = get_backend()

    if backend_type.lower() == 'quantum':
        if provider is None or provider.lower() == 'aersimulator':
            # Use Qiskit Aer simulator
            backend = Aer.get_backend('aer_simulator')
        elif provider.lower() == 'ibmq':
            # Handle IBMQ provider

            # Initialize the IBMQ provider with credentials from kwargs
            ibmq_provider = IBMProvider(**kwargs)
            # Get the least busy backend or specific backend if specified
            backend_name = kwargs.get('backend_name', None)
            if backend_name:
                backend = ibmq_provider.get_backend(backend_name)
            else:
                backend = None
                # backend = ibmq_provider.least_busy(
                #     filters=kwargs.get('filters', None))
        else:
            # Handle other providers as needed
            raise ValueError(f"Unsupported quantum provider: {provider}")
    elif backend_type.lower() == 'classical':
        # Implement classical simulation backend options
        import numpy as np

        # This is a placeholder - implement actual classical simulation backend
        backend = kwargs.get('backend', 'numpy')
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}")

    backend.set_backend(backend)
    return backend


class QuantumActuarialModel():
    """
    Quantum Actuarial model integrating classical actuarial data with quantum execution.

    Builds and runs quantum circuits representing claim portfolios.
    """
    def __init__(self, inforce, deductible=None, limit=None, **kw):
        """
        Initialize a QuantumActuarialModel.

        Args:
            inforce (Inforce or Portfolio): Policy inforce data.
            deductible (Optional[float]): Layer deductible.
            limit (Optional[float]): Layer limit.
            **kw: Additional parameters (ignored).

        Raises:
            TypeError: If inforce is not Inforce or Portfolio.
        """
        if isinstance(inforce, Inforce):
            self.portfolio = Portfolio([inforce])
        elif isinstance(inforce, Portfolio):
            self.portfolio = inforce
        else:
            raise TypeError("Need Inforce or Portfolio")

        # optional layer terms that sit above policy terms
        self.layer_deductible = deductible
        self.layer_limit = limit

        # Use the global backend manager
        self.backend = get_backend()

    def build_circuit(self):
        qc = QuantumCircuit()
        # call each bucket's state‑prep routine and ripple‑add payouts
        for bucket in self.portfolio:
            qc.compose(bucket._quantum_state(), inplace=True)
        # apply layer deductible/limit truncation oracle if needed …
        return qc

    def run(self):
        """
        Build and run the quantum circuit using the configured backend.

        Returns:
        -------
        - Result from running the circuit
        """
        circuit = self.build_circuit()
        return self.backend.run(circuit)


class BackendManager():
    """
    Manager for quantum and classical simulation backends.

    Handles backend assignment and execution interface.
    """
    def __init__(self, backend):
        """
        Initialize a BackendManager.

        Args:
            backend: Qiskit or classical backend instance.
        """
        self.backend = backend

    def set_backend(self, backend):
        """
        Update the active backend.

        Args:
            backend: New quantum or classical backend instance.
        """
        self.backend = backend

    def get_backend(self):
        """
        Retrieve the current backend.

        Returns:
            Current quantum or classical backend instance.
        """
        return self.backend

    def run(self, circuit):
        """
        Execute a quantum circuit on the active backend.

        Args:
            circuit (QuantumCircuit): Circuit to execute.

        Returns:
            Any: Execution result or job output.

        Examples:
            >>> backend = get_backend()
            >>> job = backend.run(qc)
            >>> result = job.result()
        """
        job = self.backend.run(circuit)
        return job.result()
