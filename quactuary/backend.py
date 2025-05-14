"""
This module provides the plumbing for selecting quantum vs classical execution
and interfacing with Qiskit or other quantum frameworks.

Provides both global and per-model mechanisms to select the backend,
defaulting to classical execution unless explicitly turned on or available.

Even if the backend is quantum, the output format and type remains the same:
the user gets Python/numeric results, not quantum objects.
"""

from contextlib import contextmanager

from qiskit_aer import Aer
from qiskit_ibm_provider import IBMProvider

# Global backend manager instance
_backend = None


class BackendManager():
    def __init__(self, backend):
        """
        Initialize the Backend Manager.
        Parameters:
        ----------
        - `backend`: The quantum backend to use.
        """
        self.backend = backend

    def set_backend(self, backend):
        """
        Set the quantum backend.
        Parameters:
        ----------
        - `backend`: The quantum backend to use.
        """
        self.backend = backend

    def get_backend(self):
        """
        Get the current quantum backend.
        Returns:
        -------
        - `backend`: The current quantum backend.
        """
        return self.backend

    def run(self, circuit):
        """
        Run the quantum circuit on the specified backend.
        Parameters:
        ----------
        - `circuit`: The quantum circuit to run.
        """
        job = self.backend.run(circuit)
        return job.result()


def get_backend():
    """
    Get the global backend manager instance.
    Returns:
    -------
    - `BackendManager`: The global backend manager instance.
    """
    global _backend
    if _backend is None:
        # Default to Aer simulator
        _backend = BackendManager(Aer.get_backend('aer_simulator'))
    return _backend


def set_backend(mode, provider=None, **kwargs):
    """
    Set the global backend configuration.

    Parameters:
    ----------
    - `backend_type`: String identifier for the backend type ('quantum', 'classical', etc.)
    - `provider`: Optional provider name (e.g., 'AerSimulator', 'IBMQ', etc.)
    - `**kwargs`: Additional keyword arguments for backend configuration

    Examples:
    --------
    >>> import quactuary
    >>> quactuary.set_backend('quantum', provider='AerSimulator')
    >>> quactuary.set_backend('quantum', provider='IBMQ', hub='ibm-q', token='my-token')
    >>> quactuary.set_backend('classical', num_simulations=1_000_000)
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
            ibmq_provider = IBMProvider(**kwargs)
            backend_name = kwargs.get('backend_name', None)
            if backend_name:
                new_backend = ibmq_provider.get_backend(backend_name)
            else:
                new_backend = None
                # Optionally, you could set new_backend to the provider's least_busy backend.
        else:
            raise ValueError(f"Unsupported quantum provider: {provider}")
    elif mode.lower() == 'classical':
        # This is a placeholder - implement actual classical simulation backend
        new_backend = kwargs.get('backend', 'numpy')
    else:
        raise ValueError(f"Unsupported backend type: {mode}")

    backend_manager.set_backend(new_backend)
    return new_backend


@contextmanager
def use_backend(mode, provider=None, **kwargs):
    """
    Context manager to temporarily set a backend.

    Usage:
    ------
    with use_backend('quantum', provider='AerSimulator'):
        # computations here use the temporary backend
        res = layer.compute_excess_loss()
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
