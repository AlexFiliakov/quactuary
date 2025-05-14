"""
This module provides the plumbing for selecting quantum vs classical execution
and interfacing with Qiskit or other quantum frameworks.

Provides both global and per-model mechanisms to select the backend,
defaulting to classical execution unless explicitly turned on or available.
"""

from qiskit_aer import Aer
from qiskit_ibm_provider import IBMProvider

# Global backend manager instance
_backend = None


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
    """
    backend = get_backend()

    if mode.lower() == 'quantum':
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
    elif mode.lower() == 'classical':
        # This is a placeholder - implement actual classical simulation backend
        backend = kwargs.get('backend', 'numpy')
    else:
        raise ValueError(f"Unsupported backend type: {mode}")

    backend.set_backend(backend)
    return backend


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
