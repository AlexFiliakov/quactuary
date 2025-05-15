"""
Backend selection and execution module.

Provides plumbing to choose between quantum (Qiskit) and classical
execution backends. Abstracts backend management so user-facing models
receive numeric results regardless of execution mode.

Examples:
    >>> from quactuary.backend import get_backend, set_backend
    >>> manager = get_backend()
    >>> set_backend('quantum', provider='AerSimulator')
"""

from contextlib import contextmanager

from qiskit_aer import Aer
from qiskit_ibm_provider import IBMProvider

# Global backend manager instance
_backend = None


class BackendManager():
    """
    Manager for quantum and classical backends.

    Manages backend assignments and executes quantum circuits or classical simulations.
    """
    def __init__(self, backend):
        """
        Initialize a BackendManager.

        Args:
            backend: Backend instance (e.g., Qiskit backend or custom classical engine).
        """
        self.backend = backend

    def set_backend(self, backend):
        """
        Override the active backend.

        Args:
            backend: New backend instance.
        """
        self.backend = backend

    def get_backend(self):
        """
        Retrieve the active backend.

        Returns:
            The current backend instance.
        """
        return self.backend

    def run(self, circuit):
        """
        Execute a circuit on the active backend.

        Args:
            circuit: QuantumCircuit or equivalent object to run.

        Returns:
            Result: Execution result or job output.

        Examples:
            >>> result = backend_manager.run(qc)
        """
        job = self.backend.run(circuit)
        return job.result()


def get_backend():
    """
    Retrieve or initialize the global BackendManager.

    Returns:
        BackendManager: Singleton backend manager instance.
    """
    global _backend
    if _backend is None:
        # Default to Aer simulator
        _backend = BackendManager(Aer.get_backend('aer_simulator'))
    return _backend


def set_backend(mode, provider=None, **kwargs):
    """
    Configure the global execution backend.

    Args:
        mode (str): Type of execution ('quantum' or 'classical').
        provider (Optional[str]): Quantum provider name ('AerSimulator', 'IBMQ').
        **kwargs: Additional settings (e.g., backend credentials, simulation parameters).

    Returns:
        The newly assigned backend instance.

    Raises:
        ImportError: If Qiskit or packaging dependencies are missing for quantum mode.
        ValueError: If mode or provider is unsupported.

    Examples:
        >>> set_backend('quantum', provider='AerSimulator')
        >>> set_backend('classical', backend='numpy')
    """
    backend_manager = get_backend()

    if mode.lower() == 'quantum':
        # Lazy import: Only require Qiskit when quantum mode is used.
        try:
            import qiskit
        except ImportError:
            raise ImportError("Qiskit is required for quantum backend. "
                                "Please install it with:\n\npip install qiskit==1.4.2.")

        # Ensure the installed version meets our requirements.
        try:
            from packaging import version
        except ImportError:
            raise ImportError("Please install the 'packaging' package to check Qiskit version with:"
                            "\n\npip install packaging")
        
        if version.parse(qiskit.__version__) != version.parse("1.4.2"):
            raise ImportError("Quantum mode requires Qiskit version 1.4.2 exactly. "
                                "Please upgrade it with:\n\npip install qiskit==1.4.2 --force-reinstall")

        if provider is None or provider.lower() == 'aersimulator':
            # Use Qiskit Aer simulator
            new_backend = Aer.get_backend('aer_simulator')
        elif provider.lower() == 'ibmq':
            # Use IBM's quantum provider.
            from qiskit_ibm_provider import IBMProvider
            backend_name = kwargs.get('instance', None)
            if backend_name:
                ibmq_provider = IBMProvider(**kwargs)
                new_backend = ibmq_provider.get_backend(backend_name)
            else:
                new_backend = None
                # Optionally, you could set new_backend to the provider's least_busy backend.
        else:
            raise ValueError(f"Unsupported quantum provider: {provider}")
    elif mode.lower() == 'classical':
        # This is a placeholder - implement actual classical simulation backend
        new_backend = kwargs.get('backend', provider)
    else:
        raise ValueError(f"Unsupported backend type: {mode}")

    backend_manager.set_backend(new_backend)
    return new_backend


@contextmanager
def use_backend(mode, provider=None, **kwargs):
    """
    Context manager for temporary backend configuration.

    Args:
        mode (str): Execution mode override ('quantum' or 'classical').
        provider (Optional[str]): Quantum provider name.
        **kwargs: Additional backend settings.

    Yields:
        BackendManager: Manager with temporary backend applied.

    Examples:
        >>> with use_backend('quantum', provider='AerSimulator') as mgr:
        ...     result = mgr.run(circuit)
    """
    global _backend
    # Store the original backend manager instance
    original_backend = _backend
    try:
        # Set the temporary backend
        set_backend(mode, provider, **kwargs)
        yield get_backend()
    finally:
        # Restore the original backend manager instance
        _backend = original_backend
