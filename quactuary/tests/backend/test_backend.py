from contextlib import contextmanager

import pytest
from qiskit.providers import Backend, BackendV1, BackendV2
try:
    from qiskit_ibm_runtime import QiskitRuntimeService
except ImportError:
    QiskitRuntimeService = None

import quactuary.backend as backend
from quactuary.backend import ClassicalBackend


def test_get_backend_returns_backend_manager():
    manager = backend.get_backend()
    assert isinstance(manager, backend.BackendManager)


def test_unsupported_backend_type():
    with pytest.raises(ValueError):
        backend.BackendManager(
            backend='unsupported_backend')  # type: ignore[attr-defined]
    manager = backend.BackendManager(ClassicalBackend())
    manager.backend = 'unsupported_backend'  # type: ignore[attr-defined]
    with pytest.raises(ValueError):
        backend.get_backend().set_backend(
            manager)  # type: ignore[attr-defined]


def test_copy_backend_manager():
    manager = backend.get_backend()
    orig_backend = manager.backend
    copy_manager = manager.copy()
    assert isinstance(copy_manager, backend.BackendManager)
    assert copy_manager.backend == manager.backend
    assert copy_manager.backend_type == manager.backend_type
    assert copy_manager is not manager
    assert copy_manager.backend is orig_backend
    with pytest.raises(NotImplementedError):
        deepcopy_manager = manager.copy(deep=True)


def test_set_backend_classical_sets_backend():
    backend.set_backend('classical', backend='test_backend')
    manager = backend.get_backend()
    assert manager.backend_type == 'classical'
    assert isinstance(manager.backend, backend.ClassicalBackend)


def test_set_backend_invalid_mode():
    with pytest.raises(ValueError):
        backend.set_backend('invalid_mode')


def test_set_backend_quantum_aersimulator():
    backend.set_backend('quantum', 'aersimulator')
    mgr = backend.get_backend()
    assert mgr.backend_type == 'quantum'
    assert isinstance(mgr.backend, (Backend, BackendV1, BackendV2))


def test_set_backend_quantum_defaults_to_aersimulator():
    backend.set_backend('quantum')
    mgr = backend.get_backend()
    assert mgr.backend_type == 'quantum'
    assert isinstance(mgr.backend, (Backend, BackendV1, BackendV2))


@pytest.mark.skip(reason="TODO: implement working IBM connection")
def test_set_backend_ibmq_provider_specific():
    backend.set_backend('quantum', provider='ibmq',
                        instance='ibmq_qasm_simulator')
    b = backend.get_backend().backend
    assert isinstance(b, backend.QiskitRuntimeService)
    assert b.backend_name == 'ibmq_qasm_simulator'


@pytest.mark.skip(reason="TODO: implement working IBM connection")
def test_set_backend_ibmq_provider_default():
    backend.set_backend('quantum', provider='ibmq')
    b = backend.get_backend().backend
    assert isinstance(b, backend.QiskitRuntimeService)
    pytest.fail("TODO: detect least busy backend and set it as default")


def test_set_backend_invalid_quantum_provider():
    with pytest.raises(ValueError):
        backend.set_backend('quantum', provider='notarealprovider')


def test_use_backend_context_manager_restores_backend():
    backend.set_backend()
    assert backend.get_backend().backend_type == 'quantum'
    with backend.use_backend('classical', backend='temp_backend') as temp:
        assert backend.get_backend().backend_type == 'classical'
    # After context, should restore original
    assert backend.get_backend().backend_type == 'quantum'
