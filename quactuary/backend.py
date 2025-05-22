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

from __future__ import annotations

import copy
from contextlib import contextmanager

from qiskit.providers import Backend, BackendV1
from qiskit_aer import Aer
from qiskit_ibm_provider import IBMProvider

# Global backend manager instance
global _backend


class ClassicalBackend():
    """
    Indicates that classical algorithms should be used.
    """
    version = 0


class BackendManager():
    """
    Manager for quantum and classical backends.

    Manages backend assignments and executes quantum circuits or classical simulations.
    """

    def __init__(self, backend: Backend | BackendV1 | ClassicalBackend) -> None:
        """
        Initialize a BackendManager.

        Args:
            backend: Backend instance (e.g., Qiskit backend or custom classical engine).
        """
        self.backend = backend
        if isinstance(backend, Backend) or isinstance(backend, BackendV1):
            self.backend_type = 'quantum'
        elif isinstance(backend, ClassicalBackend):
            self.backend_type = 'classical'
        else:
            raise ValueError(
                "Unsupported backend type. Must be a Qiskit or classical backend.")

    def __copy__(self) -> BackendManager:
        """
        Create a shallow copy of the BackendManager.

        Returns:
            A new instance of BackendManager with the same backend.
        """
        return BackendManager(self.backend)

    def __deepcopy__(self, memo) -> BackendManager:
        """
        Create a deep copy of the BackendManager.

        Args:
            memo: Dictionary to keep track of already copied objects.

        Returns:
            A new instance of BackendManager with the same backend.
        """
        error_message = "Deep copy is not implemented because Qiskit doesn't support it. Use shallow copy instead."
        raise NotImplementedError(error_message)
        # return BackendManager(self.backend.__deepcopy__(memo))

    def __str__(self) -> str:
        return f"BackendManager(backend_type={self.backend_type}, backend={self.backend})"

    def __repr__(self) -> str:
        return str(self)

    def copy(self, deep=False) -> BackendManager:
        """
        Create a shallow copy of the BackendManager.

        Returns:
            A new instance of BackendManager with the same backend.
        """
        if not deep:
            return copy.copy(self)
        else:
            return copy.deepcopy(self)

    def set_backend(self, manager: BackendManager) -> None:
        """
        Override the active backend.

        Args:
            manager: New BackendManager instance.
        """
        if isinstance(manager.backend, Backend) or \
                isinstance(manager.backend, BackendV1):
            self.backend_type = 'quantum'
        elif isinstance(manager.backend, ClassicalBackend):
            self.backend_type = 'classical'
        else:
            raise ValueError(
                "Unsupported backend type. Must be a Qiskit or classical backend.")
        self.backend = manager.backend


def get_backend() -> BackendManager:
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


def set_backend(mode='quantum', provider=None, **kwargs) -> BackendManager:
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
        >>> set_backend('classical')
    """
    global _backend

    if mode.lower() in ('quantum', 'q', 'qiskit', 'aer', 'ibmq'):
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
            aer_backend = Aer.get_backend('aer_simulator')
            new_backend = BackendManager(aer_backend)
        elif provider.lower() == 'ibmq':
            # Use IBM's quantum provider.
            from qiskit_ibm_provider import IBMProvider
            backend_name = kwargs.get('instance', None)
            if backend_name:
                ibmq_provider = IBMProvider(**kwargs)
                ibmq_backend = ibmq_provider.get_backend(backend_name)
                new_backend = BackendManager(ibmq_backend)
            else:
                raise NotImplementedError(
                    "Alternate IBM Quantum backend is not implemented yet.")
                # Optionally, you could set new_backend to the provider's least_busy backend.
        else:
            raise ValueError(f"Unsupported quantum provider: {provider}")
    elif mode.lower() in ('classical', 'c'):
        new_backend = BackendManager(ClassicalBackend())
    else:
        raise ValueError(f"Unsupported backend type: {mode}")

    _backend = new_backend
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
    original_backend = get_backend().copy()
    try:
        # Set the temporary backend
        set_backend(mode, provider, **kwargs)
        yield get_backend()
    finally:
        # Restore the original backend manager instance
        _backend = original_backend
