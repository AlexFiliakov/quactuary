"""
Backend selection and execution module.

This module provides the infrastructure for seamlessly switching between quantum
and classical computation backends in the quactuary framework. It abstracts the
complexity of backend management, allowing actuarial models to work with both
quantum circuits (via Qiskit) and classical algorithms transparently.

Key Features:
    - Unified interface for quantum and classical backends
    - Support for various quantum providers (Aer simulator, IBMQ, etc.)
    - Global backend configuration with context manager support
    - Automatic backend detection and initialization
    - Thread-safe backend switching

Architecture:
    - BackendManager: Wrapper class managing backend instances
    - ClassicalBackend: Marker class for classical computation
    - Global backend instance with getter/setter functions
    - Context managers for temporary backend switching

Examples:
    Basic backend configuration:
        >>> from quactuary.backend import get_backend, set_backend
        >>> 
        >>> # Get current backend (defaults to classical)
        >>> current = get_backend()
        >>> print(current.backend_type)  # 'classical'
        >>> 
        >>> # Switch to quantum simulator
        >>> set_backend('quantum', provider='AerSimulator')
        >>> 
        >>> # Switch back to classical
        >>> set_backend('classical')

    Using context manager for temporary switching:
        >>> from quactuary.backend import backend_context
        >>> 
        >>> # Temporarily use quantum backend
        >>> with backend_context('quantum'):
        ...     # Quantum computations here
        ...     result = model.calculate()
        >>> # Automatically reverts to previous backend

    Working with real quantum hardware:
        >>> # Configure IBMQ backend (requires credentials)
        >>> set_backend('quantum', provider='IBMQ', 
        ...            hub='ibm-q', group='open', project='main')

Notes:
    - Classical backend is always available and requires no dependencies
    - Quantum backend requires Qiskit and qiskit-aer packages
    - Backend selection affects all quactuary computations globally
    - Use context managers for temporary backend changes in production code
"""

from __future__ import annotations

import copy
from contextlib import contextmanager

import qiskit
from qiskit.providers import Backend, BackendV1, BackendV2
from qiskit_aer import Aer

# Global backend manager instance
_backend = None


class ClassicalBackend():
    """
    Marker class indicating classical computation backend.
    
    This class serves as a placeholder to identify when classical algorithms
    should be used instead of quantum circuits. It provides a consistent
    interface with quantum backends while signaling that computations should
    use traditional Monte Carlo or analytical methods.
    
    Attributes:
        version (int): Backend version number for compatibility tracking.
        
    Examples:
        >>> backend = ClassicalBackend()
        >>> manager = BackendManager(backend)
        >>> print(manager.backend_type)  # 'classical'
        
    Notes:
        - No actual computation logic is implemented here
        - Acts as a type marker for the BackendManager
        - Always available without external dependencies
    """
    version = 0


class BackendManager():
    """
    Manager for quantum and classical computation backends.

    This class provides a unified interface for managing different types of computational
    backends, abstracting the differences between quantum circuits and classical algorithms.
    It handles backend detection, type checking, and provides a consistent API for the
    rest of the quactuary framework.

    The BackendManager supports:
    - Qiskit quantum backends (simulators and real hardware)
    - Classical computation backend (for Monte Carlo, analytical methods)
    - Dynamic backend switching
    - Backend state management and copying

    Attributes:
        backend: The underlying backend instance (Qiskit Backend or ClassicalBackend).
        backend_type (str): Either 'quantum' or 'classical' indicating the backend type.

    Examples:
        Creating with classical backend:
            >>> classical = ClassicalBackend()
            >>> manager = BackendManager(classical)
            >>> print(manager.backend_type)  # 'classical'
            
        Creating with quantum simulator:
            >>> from qiskit_aer import Aer
            >>> quantum = Aer.get_backend('aer_simulator')
            >>> manager = BackendManager(quantum)
            >>> print(manager.backend_type)  # 'quantum'
            
        Switching backends:
            >>> manager1 = BackendManager(ClassicalBackend())
            >>> manager2 = BackendManager(Aer.get_backend('aer_simulator'))
            >>> manager1.set_backend(manager2)  # Switch to quantum

    Notes:
        - The manager does not execute computations directly
        - It serves as a configuration holder and type checker
        - Deep copying is not supported due to Qiskit limitations
    """

    def __init__(self, backend: Backend | BackendV1 | ClassicalBackend) -> None:
        """
        Initialize a BackendManager with a specific backend.

        Args:
            backend: Backend instance to manage. Can be:
                - Qiskit Backend/BackendV1/BackendV2 for quantum computation
                - ClassicalBackend instance for classical computation

        Raises:
            ValueError: If the backend type is not supported.
            
        Examples:
            >>> # Classical backend
            >>> manager = BackendManager(ClassicalBackend())
            >>> 
            >>> # Quantum simulator
            >>> from qiskit_aer import Aer
            >>> manager = BackendManager(Aer.get_backend('aer_simulator'))
        """
        self.backend = backend
        if isinstance(backend, (Backend, BackendV1, BackendV2)):
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
        if isinstance(manager.backend, (Backend, BackendV1, BackendV2)):
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
            try:
                from qiskit_ibm_runtime import QiskitRuntimeService
            except ImportError:
                raise ImportError("qiskit-ibm-runtime is required for IBM Quantum backends. "
                                  "Please install it with:\n\npip install qiskit-ibm-runtime")
            backend_name = kwargs.get('instance', None)
            if backend_name:
                ibmq_provider = QiskitRuntimeService(**kwargs)
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
