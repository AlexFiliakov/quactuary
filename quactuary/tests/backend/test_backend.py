from contextlib import contextmanager

import pytest
from qiskit_aer.backends import AerSimulator

from quactuary import backend


class DummyBackend:
    def __init__(self):
        self.ran = False

    def run(self, circuit):
        self.ran = True
        return DummyResult()


class DummyResult:
    def result(self):
        return 'dummy_result'


def test_backend_manager_set_and_get_backend():
    dummy = DummyBackend()
    manager = backend.BackendManager(dummy)
    assert manager.get_backend() is dummy
    dummy2 = DummyBackend()
    manager.set_backend(dummy2)
    assert manager.get_backend() is dummy2


def test_backend_manager_run():
    dummy = DummyBackend()
    manager = backend.BackendManager(dummy)
    result = manager.run('fake_circuit')
    assert result == 'dummy_result'
    assert dummy.ran


def test_get_backend_returns_backend_manager():
    b = backend.get_backend()
    assert isinstance(b, backend.BackendManager)


def test_set_backend_classical_sets_backend():
    backend.set_backend('classical', backend='test_backend')
    b = backend.get_backend()
    assert b.get_backend() == 'test_backend'


def test_set_backend_invalid_mode():
    with pytest.raises(ValueError):
        backend.set_backend('invalid_mode')


def test_set_backend_quantum_aersimulator():
    backend.set_backend('quantum', 'aersimulator')
    b = backend.get_backend()
    assert isinstance(b.get_backend(), AerSimulator)


def test_set_backend_quantum_defaults_to_aersimulator():
    backend.set_backend('quantum')
    b = backend.get_backend()
    assert isinstance(b.get_backend(), AerSimulator)


@pytest.mark.skip(reason="TODO: implement working IBM connection")
def test_set_backend_ibmq_provider_specific():
    backend.set_backend('quantum', provider='ibmq',
                        instance='ibmq_qasm_simulator')
    b = backend.get_backend()
    assert isinstance(b.get_backend(), backend.IBMProvider)
    assert b.get_backend().backend_name == 'ibmq_qasm_simulator'


@pytest.mark.skip(reason="TODO: implement working IBM connection")
def test_set_backend_ibmq_provider_default():
    backend.set_backend('quantum', provider='ibmq')
    b = backend.get_backend()
    assert isinstance(b.get_backend(), backend.IBMProvider)
    pytest.fail("TODO: detect least busy backend and set it as default")


def test_set_backend_invalid_quantum_provider():
    with pytest.raises(ValueError):
        backend.set_backend('quantum', provider='notarealprovider')


def test_use_backend_context_manager_restores_backend():
    orig = backend.get_backend()
    with backend.use_backend('classical', backend='temp_backend') as temp:
        assert temp.get_backend() == 'temp_backend'
    # After context, should restore original
    assert backend.get_backend() == orig
